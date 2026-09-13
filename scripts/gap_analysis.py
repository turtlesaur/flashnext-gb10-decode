#!/usr/bin/env python3
"""Attribute the non-kernel time in a decode step: GPU gaps and what the CPU is doing in them."""
import json, gzip, sys, collections
p = sys.argv[1]
ev = json.load(gzip.open(p, "rt"))["traceEvents"]
gpu = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
cpu = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "cpu_op"]
gpu.sort(key=lambda e: e["ts"])
iv = [(e["ts"], e["ts"] + e["dur"]) for e in gpu]
merged = []
cs, ce = iv[0]
for s, e in iv[1:]:
    if s > ce: merged.append((cs, ce)); cs, ce = s, e
    else: ce = max(ce, e)
merged.append((cs, ce))
gaps = [(merged[i][1], merged[i+1][0] - merged[i][1]) for i in range(len(merged) - 1)]
gaps = [(t, d) for t, d in gaps if d > 50]  # ignore sub-50us noise
tot_gap = sum(d for _, d in gaps)
span = merged[-1][1] - merged[0][0]
print(f"span {span/1e3:.0f} ms, gaps >50us: {len(gaps)}, total {tot_gap/1e3:.0f} ms ({100*tot_gap/span:.1f}% of span)")
gaps.sort(key=lambda x: -x[1])
print(f"largest gaps (us): {[round(d) for _, d in gaps[:15]]}")
# what CPU ops span each big gap
byname = collections.defaultdict(lambda: [0, 0.0])
for t, d in gaps[:400]:
    mid = t + d / 2
    best = None
    for c in cpu:
        if c["ts"] <= mid <= c["ts"] + c["dur"]:
            if best is None or c["dur"] < best["dur"]: best = c
    key = best["name"] if best else "(no cpu_op covering the gap)"
    byname[key][0] += 1; byname[key][1] += d
print("\nCPU op covering each GPU gap (innermost), by total gap time:")
for n, (c, d) in sorted(byname.items(), key=lambda x: -x[1][1])[:18]:
    print(f"  {d/1e3:8.1f} ms  x{c:4d}  {n[:95]}")
