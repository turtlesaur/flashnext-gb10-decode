#!/usr/bin/env python3
"""H1.4 analysis: expert-set overlap between adjacent tokens from expert_route_log.py output.

For each layer, walks the logged token stream and measures the number of DISTINCT experts
touched by windows of w consecutive tokens (w = 1..8), compared with the random-routing
expectation E[union] = E * (1 - (1 - k/E)^w). Reports bytes per verify pass implied by the
measured union (1 expert = 3 matrices x (640x2560/2 + scales) ~ 2.76 MB in NVFP4).
Usage: route_overlap.py /research/route_log/*.npz
"""
import sys, glob, collections
import numpy as np

E, K = 512, 10
BYTES_PER_EXPERT = 3 * (640 * 2560 // 2 + 640 * 160)  # fp4 packed + fp8 group scales, per layer
per_layer = collections.defaultdict(list)
for pat in sys.argv[1:]:
    for f in sorted(glob.glob(pat)):
        d = np.load(f)
        names, lens, ids = d["names"], d["lens"], d["ids"]
        off = 0
        for n, l in zip(names, lens):
            per_layer[str(n)].append(ids[off:off + l]); off += l
layers = sorted(per_layer)
print(f"{len(layers)} layers logged; tokens per layer ~{np.mean([sum(x.shape[0] for x in v) for v in per_layer.values()]):.0f}")
print(f"{'w':>3} {'measured union':>15} {'random expectation':>19} {'GB/verify (48 layers)':>22}")
for w in range(1, 9):
    unions = []
    for name in layers:
        stream = np.concatenate(per_layer[name])  # [T, k]
        for i in range(0, len(stream) - w + 1, max(1, w)):
            unions.append(len(np.unique(stream[i:i + w])))
    m = float(np.mean(unions)) if unions else float("nan")
    rnd = E * (1 - (1 - K / E) ** w)
    print(f"{w:>3} {m:15.2f} {rnd:19.2f} {m * BYTES_PER_EXPERT * 48 / 1e9:22.2f}")
# per-layer skew: how concentrated is routing (top-32 experts' share of assignments)?
print("\nrouting concentration (share of assignments to the 32 most used experts), first/mid/last layers:")
for name in [layers[0], layers[len(layers) // 2], layers[-1]] if layers else []:
    stream = np.concatenate(per_layer[name]).reshape(-1)
    cnt = np.bincount(stream, minlength=E); share = np.sort(cnt)[::-1][:32].sum() / cnt.sum()
    print(f"  {name}: {share:.3f}")
