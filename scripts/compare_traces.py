#!/usr/bin/env python3
"""Compare two torch traces normalised by decode-step count.

Steps are counted from `_rejection_kernel` (one per speculative decode step).
Usage: compare_traces.py LABEL_A=path_a LABEL_B=path_b
"""
import json, gzip, collections, sys


def load(p):
    ev = json.load(gzip.open(p, "rt"))["traceEvents"]
    g = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    steps = sum(1 for e in g if "_rejection_kernel" in e["name"])
    if steps == 0:
        steps = sum(1 for e in g if "ArgMaxOps" in e["name"]) or 1
    iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in g)
    m = []
    cs, ce = iv[0]
    for s, e in iv[1:]:
        if s > ce:
            m.append((cs, ce)); cs, ce = s, e
        else:
            ce = max(ce, e)
    m.append((cs, ce))
    busy = sum(b - a for a, b in m)
    span = m[-1][1] - m[0][0]
    return g, steps, busy, span


def fam(n):
    if "gemvx" in n: return "gemvx(M=1)"
    if "wmma" in n: return "wmma(dense)"
    if "BlockScaled" in n: return "nvfp4_moe"
    if "fused_moe" in n or "MoeFCGemm" in n: return "bf16_moe"
    if "nvjet" in n: return "nvjet"
    if "gdn" in n or "conv1d" in n: return "gdn"
    if "qsa" in n: return "qsa"
    return "other"


res = {}
order = []
for arg in sys.argv[1:]:
    lab, p = arg.split("=", 1)
    order.append(lab)
    g, steps, busy, span = load(p)
    d = collections.defaultdict(float)
    for e in g:
        d[fam(e["name"])] += e["dur"]
    tot = sum(e["dur"] for e in g)
    res[lab] = (steps, busy, span, d, tot)
    print(f"{lab}: {steps} decode steps | union-busy {busy/1e3/steps:6.2f} ms/step | "
          f"span {span/1e3/steps:6.2f} ms/step | kernel-sum {tot/1e3/steps:6.2f} ms/step")

a, b = order[0], order[1]
keys = sorted(set(res[a][3]) | set(res[b][3]), key=lambda k: -res[a][3].get(k, 0))
print()
print("%-14s %14s %14s %9s" % ("family", a + " ms/st", b + " ms/st", "delta"))
for k in keys:
    x = res[a][3].get(k, 0) / 1e3 / res[a][0]
    y = res[b][3].get(k, 0) / 1e3 / res[b][0]
    print("%-14s %14.2f %14.2f %9.2f" % (k, x, y, y - x))
for lab, idx, name in ((None, 4, "kernel-sum"), (None, 1, "union-busy"), (None, 2, "span")):
    x = res[a][idx] / 1e3 / res[a][0]
    y = res[b][idx] / 1e3 / res[b][0]
    print("%-14s %14.2f %14.2f %9.2f" % (name, x, y, y - x))
