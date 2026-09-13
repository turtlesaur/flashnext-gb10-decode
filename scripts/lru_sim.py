#!/usr/bin/env python3
"""H4.4: simulate hot-row cache hit rates for the PLE (engram) table from logged row ids.

Input: one or more .npz files written by ple_log_patch.py (arrays: ids, lens). Each call's
ids are the table rows a decode/prefill step needed. Simulates LRU (per-call granularity: a
call's rows are looked up together, then inserted) at several cache sizes and reports the
hit rate per row and per call, plus how many distinct rows the trace touched.
Usage: lru_sim.py /research/ple_log/*.npz [--row-bytes 160]
"""
import sys, glob, argparse
from collections import OrderedDict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--row-bytes", type=int, default=160)
ap.add_argument("--sizes-gb", default="0.5,1,2,4,8,16")
a = ap.parse_args()

calls = []
for pat in a.files:
    for f in sorted(glob.glob(pat)):
        d = np.load(f)
        ids, lens = d["ids"], d["lens"]
        off = 0
        for n in lens:
            calls.append(ids[off:off + n]); off += n
tot_rows = sum(len(c) for c in calls)
distinct = len(np.unique(np.concatenate(calls))) if calls else 0
print(f"{len(calls)} calls, {tot_rows} row lookups, {distinct} distinct rows, "
      f"{np.mean([len(c) for c in calls]):.1f} rows/call; unique rows per call: "
      f"{np.mean([len(np.unique(c)) for c in calls]):.1f}")
sizes = [float(s) for s in a.sizes_gb.split(",")]
for gb in sizes:
    cap = int(gb * 1e9 / a.row_bytes)
    lru = OrderedDict(); hits = misses = 0; call_hits = 0
    for c in calls:
        u = np.unique(c); h = 0
        for r in u.tolist():
            if r in lru:
                lru.move_to_end(r); h += 1
            else:
                lru[r] = None
                if len(lru) > cap: lru.popitem(last=False)
        hits += h; misses += len(u) - h
        if h == len(u): call_hits += 1
    print(f"cache {gb:5.1f} GB ({cap/1e6:6.1f}M rows): row hit rate {hits/(hits+misses):.3f}, "
          f"all-hit calls {call_hits/len(calls):.3f}, mean misses/call {misses/len(calls):.1f}")
# frequency skew: what share of lookups do the top-N rows cover (upper bound for a static hot set)
allids = np.concatenate(calls)
vals, counts = np.unique(allids, return_counts=True)
order = np.argsort(-counts); cum = np.cumsum(counts[order]) / len(allids)
for n in (100_000, 1_000_000, 5_000_000, 25_000_000):
    if n < len(cum): print(f"top {n/1e6:5.1f}M rows cover {cum[n-1]:.3f} of lookups (static oracle)")
