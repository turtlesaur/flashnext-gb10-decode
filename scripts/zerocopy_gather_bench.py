#!/usr/bin/env python3
"""H3.1: zero-copy GPU gather from pinned host memory on GB10 (unified memory).

Compares, for R random 160-byte rows out of a T-row table:
  A. today's path: ids on GPU -> D2H -> CPU gather from a host array -> pinned staging -> H2D -> expand
  B. GPU gather kernel reading the pinned host table directly through its device pointer
  C. GPU gather from a device-resident copy of the same table (upper bound)
Also reports random-row and sequential read bandwidth of host memory as seen by the GPU.
Run: docker run --rm --gpus all -v ~/research:/research --entrypoint python3 <image> /research/zerocopy_gather_bench.py
"""
import time, sys
import numpy as np
import torch
from torch.utils.cpp_extension import load_inline

ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 25_000_000   # 25M rows x 160 B = 4 GB host table
R = int(sys.argv[2]) if len(sys.argv) > 2 else 180
ROW = 160
dev = "cuda"

cpp_src = r"""
torch::Tensor gather_from_ptr(int64_t table_ptr, torch::Tensor idx, torch::Tensor out, int64_t row_bytes);
int64_t host_to_device_ptr(int64_t host_ptr);
"""
cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

__global__ void gather_rows(const unsigned char* __restrict__ table, const long* __restrict__ idx,
                            unsigned char* __restrict__ out, long row_bytes) {
    long r = blockIdx.x;
    const uint4* src = reinterpret_cast<const uint4*>(table + idx[r] * row_bytes);
    uint4* dst = reinterpret_cast<uint4*>(out + r * row_bytes);
    for (int i = threadIdx.x; i < row_bytes / 16; i += blockDim.x) dst[i] = src[i];
}

torch::Tensor gather_from_ptr(int64_t table_ptr, torch::Tensor idx, torch::Tensor out, int64_t row_bytes) {
    const unsigned char* table = reinterpret_cast<const unsigned char*>(table_ptr);
    int rows = idx.size(0);
    gather_rows<<<rows, 32, 0, c10::cuda::getCurrentCUDAStream()>>>(
        table, idx.data_ptr<long>(), out.data_ptr<unsigned char>(), row_bytes);
    return out;
}

int64_t host_to_device_ptr(int64_t host_ptr) {
    void* d = nullptr;
    cudaError_t e = cudaHostGetDevicePointer(&d, reinterpret_cast<void*>(host_ptr), 0);
    TORCH_CHECK(e == cudaSuccess, "cudaHostGetDevicePointer failed: ", cudaGetErrorString(e));
    return reinterpret_cast<int64_t>(d);
}
"""
ext = load_inline(name="zc_gather2", cpp_sources=cpp_src, cuda_sources=cuda_src,
                  functions=["gather_from_ptr", "host_to_device_ptr"],
                  extra_cuda_cflags=["-O3"], verbose=False)
print("device:", torch.cuda.get_device_name(0))
host = torch.empty((ROWS, ROW), dtype=torch.uint8, pin_memory=True)
host.view(-1)[::4096] = 7  # touch pages
print(f"host pinned table: {host.numel()/1e9:.2f} GB, {ROWS/1e6:.1f}M rows")
dptr = ext.host_to_device_ptr(host.data_ptr())
print("device ptr for host table:", hex(dptr), "(identity => single address space)" if dptr == host.data_ptr() else "(remapped)")
host_np = host.numpy()
out = torch.empty((R, ROW), dtype=torch.uint8, device=dev)
stage = torch.empty((R, ROW), dtype=torch.uint8, pin_memory=True)

def timeit(fn, n=50):
    for _ in range(5): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n

def new_ids(n=R, hi=ROWS):
    return torch.randint(0, hi, (n,), dtype=torch.int64, device=dev)

def path_a():
    ids = new_ids()
    ids_np = ids.cpu().numpy()
    rows = host_np[ids_np]
    stage.numpy()[:] = rows
    out.copy_(stage, non_blocking=True)
tA = timeit(path_a)

def path_b():
    ext.gather_from_ptr(dptr, new_ids(), out, ROW)
tB = timeit(path_b)

dev_rows = min(ROWS, 6_000_000)
dtab = torch.empty((dev_rows, ROW), dtype=torch.uint8, device=dev)
def path_c():
    ext.gather_from_ptr(dtab.data_ptr(), new_ids(dev_rows and R, dev_rows), out, ROW)
tC = timeit(path_c)

print(f"\nR={R} rows per step:")
print(f"  A  today  (D2H ids + CPU gather + H2D): {tA*1e6:8.0f} us")
print(f"  B  zero-copy GPU gather from pinned host: {tB*1e6:8.0f} us   ({tA/tB:.1f}x faster than A)")
print(f"  C  GPU gather from device-resident table: {tC*1e6:8.0f} us   (upper bound)")

ids = new_ids(); ext.gather_from_ptr(dptr, ids, out, ROW); torch.cuda.synchronize()
ref = torch.from_numpy(host_np[ids.cpu().numpy()])
print("  B output matches a CPU gather:", bool((out.cpu() == ref).all()))

print("\nscaling of the zero-copy gather (random rows out of the 4 GB host table):")
for r in (64, 256, 1024, 4096, 16384, 65536):
    o = torch.empty((r, ROW), dtype=torch.uint8, device=dev)
    t = timeit(lambda: ext.gather_from_ptr(dptr, new_ids(r), o, ROW), 20)
    print(f"  R={r:6d}: {t*1e6:8.0f} us -> {r*ROW/t/1e9:6.2f} GB/s random-row read from host")

big = torch.empty(1 << 30, dtype=torch.uint8, pin_memory=True)
bptr = ext.host_to_device_ptr(big.data_ptr())
nrow = (1 << 30) // ROW
o = torch.empty((nrow, ROW), dtype=torch.uint8, device=dev)
ids_seq = torch.arange(nrow, dtype=torch.int64, device=dev)
t = timeit(lambda: ext.gather_from_ptr(bptr, ids_seq, o, ROW), 5)
print(f"\nGPU sequential read of 1 GiB pinned host memory: {(1<<30)/t/1e9:.1f} GB/s")
d = torch.empty(1 << 30, dtype=torch.uint8, device=dev)
t = timeit(lambda: d.copy_(big, non_blocking=True), 5)
print(f"cudaMemcpy H2D of the same 1 GiB:               {(1<<30)/t/1e9:.1f} GB/s")
