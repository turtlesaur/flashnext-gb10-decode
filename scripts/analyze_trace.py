#!/usr/bin/env python3
"""Summarize a torch-profiler Chrome trace: GPU kernel time by name, busy fraction, per-step attribution."""
import json, sys, gzip, collections, os, glob
path = sys.argv[1]
if os.path.isdir(path):
    cands = sorted(glob.glob(os.path.join(path, "**", "*.json*"), recursive=True), key=os.path.getmtime)
    path = cands[-1]
print("trace:", path, f"{os.path.getsize(path)/1e6:.1f} MB")
op = gzip.open if path.endswith(".gz") else open
with op(path, "rt") as f:
    data = json.load(f)
ev = data["traceEvents"] if isinstance(data, dict) else data
gpu = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
cpu = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("cpu_op",)]
if not gpu:
    print("no GPU events; cats:", collections.Counter(e.get("cat") for e in ev).most_common(10)); sys.exit()
t0 = min(e["ts"] for e in gpu); t1 = max(e["ts"] + e["dur"] for e in gpu)
span = t1 - t0
iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in gpu)
busy = 0; cs, ce = iv[0]
for s, e in iv[1:]:
    if s > ce:
        busy += ce - cs; cs, ce = s, e
    else:
        ce = max(ce, e)
busy += ce - cs
tot = sum(e["dur"] for e in gpu)
by = collections.defaultdict(lambda: [0, 0.0])
for e in gpu:
    b = by[e["name"]]; b[0] += 1; b[1] += e["dur"]
markers = {n: c for n, (c, d) in by.items() if any(k in n.lower() for k in ("rejection", "ple", "sampl", "topk", "argmax"))}
print(f"GPU span {span/1e3:.1f} ms, busy(union) {busy/1e3:.1f} ms ({100*busy/span:.1f}%), sum(kernels) {tot/1e3:.1f} ms")
steps = int(sys.argv[2]) if len(sys.argv) > 2 else None
if steps:
    print(f"steps={steps}: {span/1e3/steps:.2f} ms/step wall, {busy/1e3/steps:.2f} ms/step GPU-busy")
print("\nmarker kernel counts:", dict(sorted(markers.items(), key=lambda x: -x[1])[:8]))
print(f"\n{'total ms':>9} {'%':>6} {'count':>7} {'avg us':>8}  kernel")
rows = sorted(by.items(), key=lambda x: -x[1][1])
for n, (c, d) in rows[:45]:
    print(f"{d/1e3:9.2f} {100*d/tot:6.1f} {c:7d} {d/c:8.1f}  {n[:110]}")
fam = collections.defaultdict(float)
FAM = [("gemm", "GEMM(dense cublas)"), ("cutlass", "cutlass"), ("nvfp4", "nvfp4/moe"), ("moe", "moe"), ("fp4", "fp4"),
       ("gdn", "gdn"), ("gated_delta", "gdn"), ("recurrent", "gdn"), ("qsa", "qsa attn"), ("attn", "attn"), ("flash", "attn"),
       ("topk", "topk"), ("ple", "ple"), ("sampl", "sampling"), ("rejection", "sampling"), ("memcpy", "memcpy"), ("memset", "memset"),
       ("elementwise", "elementwise"), ("reduce", "reduce/norm"), ("norm", "reduce/norm"), ("triton", "triton-other"), ("copy", "copy"), ("cat", "copy")]
def family(n):
    l = n.lower()
    for k, f in FAM:
        if k in l: return f
    return "other"
for n, (c, d) in by.items(): fam[family(n)] += d
print("\nby family:")
for f, d in sorted(fam.items(), key=lambda x: -x[1]):
    print(f"  {d/1e3:9.2f} ms {100*d/tot:5.1f}%  {f}")
if cpu:
    byc = collections.defaultdict(float)
    for e in cpu: byc[e["name"]] += e["dur"]
    print("\ntop CPU ops (inclusive, ms):")
    for n, d in sorted(byc.items(), key=lambda x: -x[1])[:15]: print(f"  {d/1e3:9.1f}  {n[:100]}")
