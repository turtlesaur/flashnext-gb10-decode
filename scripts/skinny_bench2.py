import torch, time, random
dev = "cuda"; torch.manual_seed(0); random.seed(0)
p = torch.cuda.get_device_properties(0)
print("device:", p.name, "L2 cache MB:", getattr(p, "L2_cache_size", 0) / 1e6, "SMs:", p.multi_processor_count)
flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)  # 256 MB > any L2
def graph_time(fn, iters=10, reps=5):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(2): fn()
    torch.cuda.synchronize()
    with torch.cuda.graph(g, stream=s):
        for _ in range(iters): fn()
    torch.cuda.synchronize(); best = 1e9
    for _ in range(reps):
        torch.cuda.synchronize(); t = time.perf_counter(); g.replay(); torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t) / iters)
    return best
t_flush = graph_time(lambda: flush.fill_(1))
print(f"flush (256MB write): {t_flush*1e6:.0f} us -> {flush.numel()/t_flush/1e9:.0f} GB/s write")
rd = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, device=dev)
t_rd = graph_time(lambda: rd.sum())
print(f"read 256MB fp32 sum: {t_rd*1e6:.0f} us -> {rd.numel()*4/t_rd/1e9:.0f} GB/s read")
shapes = [("gdn in_proj_qkv", 10240, 2560, 36), ("gdn in_proj_z", 6144, 2560, 36), ("gdn out_proj", 2560, 6144, 36),
          ("attn q_proj", 12288, 2560, 12), ("attn kv_proj", 1024, 2560, 12), ("attn o_proj", 2560, 6144, 12),
          ("indexer qk", 640, 2560, 12), ("shared gate_up", 1280, 2560, 48), ("shared down", 2560, 640, 48),
          ("hc mix down", 320, 10240, 96), ("hc mix up", 10240, 320, 96), ("router", 512, 2560, 48),
          ("lm_head", 248320, 2560, 1)]
Ms = [1, 2, 4, 5, 6]
print(f"\n{'layer (bf16 cuBLAS, L2-flushed)':<32}{'N':>7}{'K':>6}" + "".join(f"{'M='+str(m):>13}" for m in Ms) + "   [us / GB/s]")
tot = {m: 0.0 for m in Ms}
order = shapes[:]; random.shuffle(order)
res = {}
for name, N, K, mult in order:
    W = torch.randn(N, K, dtype=torch.bfloat16, device=dev)
    for M in Ms:
        xx = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
        def fn():
            flush.fill_(1); torch.nn.functional.linear(xx, W)
        t = graph_time(fn) - t_flush
        res[(name, M)] = t
    del W
for name, N, K, mult in shapes:
    row = f"{name:<32}{N:>7}{K:>6}"
    for M in Ms:
        t = res[(name, M)]; row += f"{t*1e6:7.1f}/{N*K*2/t/1e9:4.0f} "; tot[M] += t * mult
    print(row)
ideal = sum(N * K * 2 * m for _, N, K, m in shapes) / 230e9 * 1e3
print("\nSum over one verify pass of dense side-layer bf16 GEMMs (x layer counts), L2-flushed:")
for M in Ms: print(f"  M={M}: {tot[M]*1e3:6.2f} ms   (ideal at 230 GB/s: {ideal:.1f} ms)")
print("\nfp8 weight variants (10240x2560):")
W8 = torch.randn(10240, 2560, device=dev).to(torch.float8_e4m3fn)
for M in [1, 4]:
    xx = torch.randn(M, 2560, dtype=torch.bfloat16, device=dev)
    def fn():
        flush.fill_(1); torch.nn.functional.linear(xx, W8.to(torch.bfloat16))
    t = graph_time(fn) - t_flush
    print(f"  naive fp8->bf16 cast + gemm M={M}: {t*1e6:.0f} us (weight {W8.numel()/1e6:.0f} MB -> {W8.numel()/t/1e9:.0f} GB/s effective)")
xf = torch.randn(16, 2560, device=dev).to(torch.float8_e4m3fn); sa = torch.tensor(1.0, device=dev); sb = torch.tensor(1.0, device=dev)
def fn():
    flush.fill_(1); torch._scaled_mm(xf, W8.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
t = graph_time(fn) - t_flush
print(f"  _scaled_mm fp8xfp8 M=16: {t*1e6:.0f} us -> {W8.numel()/t/1e9:.0f} GB/s effective")
