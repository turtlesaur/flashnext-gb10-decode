#!/usr/bin/env python3
"""H4.1 v2: PLE (engram) row-gather strategies, with PERSISTENT thread pools.

v1 created a ThreadPoolExecutor per call, which dominated the high-thread-count numbers.
Production (vllm_ple_mmap.py) builds one pool in __init__, so a persistent pool is the
correct comparison. Strategies:
  memmap  - np.memmap fancy indexing; misses are synchronous page faults (what ships today)
  pread   - os.pread per row through the page cache (explicit I/O, same caching)
  odirect - os.preadv with O_DIRECT into aligned buffers (bypasses the page cache)
  seq     - one pread per contiguous run after sorting (coalesces adjacent rows)
Row distributions: uniform (worst case) and zipf (realistic n-gram skew).
Usage: ple_gather_bench2.py [rows_per_step] [steps]
"""
import os, sys, json, struct, time, mmap, random, glob, ctypes
import numpy as np
from concurrent.futures import ThreadPoolExecutor

SNAP = os.environ.get("PLE_SNAP") or os.path.expanduser(
    "~/.cache/huggingface/hub/models--<org>--Qwen3.8-Flash-Next-NVFP4/snapshots/<revision>")
ROW = 160
R = int(sys.argv[1]) if len(sys.argv) > 1 else 180
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 25

shards = sorted(glob.glob(SNAP + "/model-plefp8-*.safetensors"))
meta = []
for p in shards:
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    k = [k for k in hdr if k != "__metadata__"][0]
    meta.append((os.path.realpath(p), hdr[k]["data_offsets"][0] + 8 + n, hdr[k]["shape"][0]))
total_rows = sum(m[2] for m in meta)
starts = np.cumsum([0] + [m[2] for m in meta])
print(f"{len(meta)} shards, {total_rows/1e6:.1f}M rows x {ROW} B; R={R}, {STEPS} steps")

def locate(gid):
    si = int(np.searchsorted(starts, gid, side="right")) - 1
    return si, meta[si][1] + (gid - starts[si]) * ROW

mms = [np.memmap(p, dtype=np.uint8, mode="r") for p, _, _ in meta]
fds = [os.open(p, os.O_RDONLY) for p, _, _ in meta]
try:
    fdd = [os.open(p, os.O_RDONLY | os.O_DIRECT) for p, _, _ in meta]
except OSError:
    fdd = None

POOLS = {}
def pool(n):
    if n not in POOLS:
        POOLS[n] = ThreadPoolExecutor(n)
        list(POOLS[n].map(lambda _: None, range(n)))  # force thread creation
    return POOLS[n]

def s_memmap(ids, nt):
    def one(g):
        si, o = locate(g); return bytes(mms[si][o:o + ROW])
    return list(pool(nt).map(one, ids))

def s_pread(ids, nt):
    def one(g):
        si, o = locate(g); return os.pread(fds[si], ROW, o)
    return list(pool(nt).map(one, ids))

def s_odirect(ids, nt):
    def one(g):
        si, o = locate(g); a = o & ~4095
        buf = mmap.mmap(-1, 8192)
        os.preadv(fdd[si], [buf], a)
        return bytes(buf[o - a:o - a + ROW])
    return list(pool(nt).map(one, ids))

def s_seq(ids, nt):
    """Coalesce sorted ids into runs; one pread per run (<=64 KiB)."""
    runs, cur = [], [ids[0], ids[0]]
    for g in ids[1:]:
        if g - cur[1] <= 25 and (g - cur[0]) * ROW < 65536:  # within 4 KiB-ish
            cur[1] = g
        else:
            runs.append(tuple(cur)); cur = [g, g]
    runs.append(tuple(cur))
    def one(run):
        a, b = run; si, o = locate(a); return os.pread(fds[si], (b - a + 1) * ROW, o)
    out = list(pool(nt).map(one, runs))
    return out, len(runs)

rng = np.random.default_rng(0)
def ids_uniform():
    return sorted(rng.integers(0, total_rows, R).tolist())
# zipf-like: 80 % of lookups from a 1 % hot region, 20 % uniform
HOT = int(total_rows * 0.01)
hot_base = rng.integers(0, total_rows - HOT)
def ids_zipf():
    nh = int(R * 0.8)
    a = rng.integers(hot_base, hot_base + HOT, nh)
    b = rng.integers(0, total_rows, R - nh)
    return sorted(np.concatenate([a, b]).tolist())

def bench(name, fn, gen, nt, warm=False):
    ids = gen(); ts = []
    for _ in range(STEPS):
        if not warm: ids = gen()
        t0 = time.perf_counter(); fn(ids, nt); ts.append(time.perf_counter() - t0)
    ts = sorted(ts)
    print(f"  {name:<26} t={nt:<4} {'warm' if warm else 'cold':<5} median {ts[len(ts)//2]*1e3:7.2f} ms  p90 {ts[int(len(ts)*0.9)]*1e3:7.2f}  min {ts[0]*1e3:7.2f}")
    return ts[len(ts) // 2]

print("\n=== uniform random rows (worst case), cold ===")
best = {}
for nt in (8, 16, 32, 64, 128):
    best[("memmap", nt)] = bench("memmap fancy", s_memmap, ids_uniform, nt)
for nt in (8, 16, 32, 64, 128, 180):
    best[("pread", nt)] = bench("pread", s_pread, ids_uniform, nt)
if fdd:
    for nt in (32, 128):
        bench("pread O_DIRECT", s_odirect, ids_uniform, nt)
for nt in (32, 128):
    bench("pread coalesced runs", lambda i, n: s_seq(i, n)[0], ids_uniform, nt)

print("\n=== zipf-ish rows (80 % from a 1 % hot region), cold ===")
for nt in (8, 32, 128):
    bench("memmap fancy", s_memmap, ids_zipf, nt)
    bench("pread", s_pread, ids_zipf, nt)

print("\n=== same row set repeated (fully warm) ===")
for nt in (8, 32):
    bench("memmap fancy", s_memmap, ids_uniform, nt, warm=True)
    bench("pread", s_pread, ids_uniform, nt, warm=True)

print("\nsummary: best cold memmap %.2f ms, best cold pread %.2f ms" % (
    min(v for (k, _), v in best.items() if k == "memmap"),
    min(v for (k, _), v in best.items() if k == "pread")))
