#!/usr/bin/env python3
"""H2.1: NVFP4 fused-MoE kernel efficiency at Flash-Next decode shapes on GB10 (sm_121).

Model shape per MoE layer: 512 experts, top-10 routing, hidden 2560, intermediate 640,
NVFP4 weights (group size 16) with fp8 group scales. 48 layers.

For M = 1..8 tokens (a speculative verify pass is M = 1 + num_speculative_tokens), measures
each available backend and compares with the memory-bandwidth roofline implied by the number
of DISTINCT experts the routing actually selects (which is what must be read from memory).

Usage (sidecar alongside the running engine, or standalone):
  docker run --rm --gpus all --ipc=host -v ~/research:/research --entrypoint python3 <image> \
      /research/moe_kernel_bench.py
"""
import os, sys, time, itertools
import torch

os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
E, TOPK, H, I = 512, 10, 2560, 640
DEV = "cuda"
BYTES_PER_EXPERT = 3 * (I * H // 2) + 3 * (I * H // 16)   # fp4 packed + fp8 group-16 scales
print(f"device {torch.cuda.get_device_name(0)}  cap {torch.cuda.get_device_capability(0)}")
print(f"{E} experts, top-{TOPK}, H={H}, I={I}; {BYTES_PER_EXPERT/1e6:.2f} MB per expert per layer "
      f"({E*BYTES_PER_EXPERT/1e9:.2f} GB per layer, {48*E*BYTES_PER_EXPERT/1e9:.1f} GB for 48 layers)")

def graph_time(fn, iters=10, reps=5):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(2): fn()
    torch.cuda.synchronize()
    try:
        with torch.cuda.graph(g, stream=s):
            for _ in range(iters): fn()
    except Exception:
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(iters * reps): fn()
        torch.cuda.synchronize(); return (time.perf_counter() - t) / (iters * reps)
    torch.cuda.synchronize(); best = 1e9
    for _ in range(reps):
        torch.cuda.synchronize(); t = time.perf_counter(); g.replay(); torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t) / iters)
    return best

# ---- distinct-expert statistics under uniform random routing (H1.4 baseline) ----
print("\ndistinct experts selected by M tokens (uniform random routing, 2000 trials):")
gen = torch.Generator(device="cpu").manual_seed(0)
for M in (1, 2, 3, 4, 5, 6, 8):
    d = [len(torch.randperm(E, generator=gen)[:TOPK * M].unique()) if False else
         len(torch.stack([torch.randperm(E, generator=gen)[:TOPK] for _ in range(M)]).unique())
         for _ in range(200)]
    mean = sum(d) / len(d)
    print(f"  M={M}: {mean:6.1f} distinct  -> {mean*BYTES_PER_EXPERT*48/1e9:5.2f} GB per 48-layer pass "
          f"(roofline at 240 GB/s: {mean*BYTES_PER_EXPERT*48/240e9*1e3:6.2f} ms)")

# ---- actual kernels ----
try:
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    import vllm.model_executor.layers.fused_moe as fm
except Exception as exc:
    print("vllm fused_moe import failed:", exc); sys.exit(1)

def make_nvfp4_weights():
    w13 = torch.randint(0, 255, (E, 2 * I, H // 2), dtype=torch.uint8, device=DEV)
    w2 = torch.randint(0, 255, (E, H, I // 2), dtype=torch.uint8, device=DEV)
    w13_s = torch.ones((E, 2 * I, H // 16), dtype=torch.float8_e4m3fn, device=DEV)
    w2_s = torch.ones((E, H, I // 16), dtype=torch.float8_e4m3fn, device=DEV)
    g1 = torch.ones(E, dtype=torch.float32, device=DEV)
    g2 = torch.ones(E, dtype=torch.float32, device=DEV)
    return w13, w13_s, w2, w2_s, g1, g2

def try_flashinfer_b12x(M):
    from flashinfer.fused_moe import b12x_fused_moe
    w13, w13_s, w2, w2_s, g1, g2 = make_nvfp4_weights()
    x = torch.randn(M, H, dtype=torch.bfloat16, device=DEV)
    sel = torch.stack([torch.randperm(E, device=DEV)[:TOPK] for _ in range(M)]).to(torch.int32)
    sc = torch.softmax(torch.randn(M, TOPK, device=DEV), -1).to(torch.float32)
    sfa = torch.ones(1, dtype=torch.float32, device=DEV)
    def fn():
        b12x_fused_moe(x, w13, w13_s, w2, w2_s, sel, sc, E, TOPK,
                       w1_alpha=g1, w2_alpha=g2, fc2_input_scale=sfa, input_global_scale=sfa,
                       output_dtype=torch.bfloat16, activation="silu", quant_mode="nvfp4")
    fn(); torch.cuda.synchronize()
    return graph_time(fn)

def try_flashinfer_cutlass(M):
    from flashinfer import fp4_quantize
    from flashinfer.fused_moe import cutlass_fused_moe
    w13, w13_s, w2, w2_s, g1, g2 = make_nvfp4_weights()
    x = torch.randn(M, H, dtype=torch.bfloat16, device=DEV)
    sel = torch.stack([torch.randperm(E, device=DEV)[:TOPK] for _ in range(M)]).to(torch.int32)
    sc = torch.softmax(torch.randn(M, TOPK, device=DEV), -1).to(torch.float32)
    one = torch.ones(1, dtype=torch.float32, device=DEV)
    out = torch.empty(M, H, dtype=torch.bfloat16, device=DEV)
    xq, xs = fp4_quantize(x, one)
    def fn():
        cutlass_fused_moe(xq, sel, sc, w13, w2, torch.bfloat16, output=out,
                          quant_scales=[one, w13_s.view(torch.uint8), one, w2_s.view(torch.uint8), one, one],
                          input_sf=xs)
    fn(); torch.cuda.synchronize()
    return graph_time(fn)

print("\nfused-MoE kernel time per LAYER (us), and the implied 48-layer cost (ms):")
print(f"{'M':>3} {'backend':<22} {'us/layer':>10} {'ms x48':>9} {'GB/s effective':>16}")
for M in (1, 2, 4, 5, 6):
    for name, fnmaker in (("flashinfer_b12x", try_flashinfer_b12x), ("flashinfer_cutlass", try_flashinfer_cutlass)):
        try:
            t = fnmaker(M)
            distinct = min(E, TOPK * M)
            gb = distinct * BYTES_PER_EXPERT
            print(f"{M:>3} {name:<22} {t*1e6:10.1f} {t*48*1e3:9.2f} {gb/t/1e9:16.1f}")
        except Exception as exc:
            print(f"{M:>3} {name:<22} {'FAIL':>10}  {str(exc)[:80]}")
        torch.cuda.empty_cache()
