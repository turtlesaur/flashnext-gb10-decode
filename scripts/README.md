# Scripts

## Harness and orchestration

| file | what it does |
|---|---|
| `bench_steps.py` | the benchmark. Reads the engine's Prometheus counters around each request, so per-step time and acceptance are independent of client overhead. |
| `exp.sh` | launch an isolated engine (own port, cgroup memory cap) with a given configuration. |
| `entrypoint_patched.sh` | inject the patches into the container's writable layer, then exec the normal entrypoint. |
| `run_bench_monitored.sh` | benchmark while sampling GPU utilisation, disk I/O, thread placement and major page faults. |
| `sanitize.py` | build this public tree from the private one and fail if anything host-specific survives. |

## Analysis

| file | what it does |
|---|---|
| `compare_traces.py` | compare two profiler traces normalised by decode-step count. |
| `gap_analysis.py` | find the gaps *between* GPU kernels and attribute each to the CPU op spanning it. |
| `analyze_trace.py` | single-trace kernel breakdown by name and family. |
| `byte_budget.py` | per-class weight byte budget straight from the checkpoint's safetensors headers. |
| `make_figures.py` | regenerate the figures in `docs/figures`. |

## Machine characterisation

| file | what it measures |
|---|---|
| `skinny_bench2.py` | cuBLAS at the model's exact GEMM shapes, L2 flushed, CUDA-graph timed. |
| `zerocopy_gather_bench.py` | GPU reads of pinned host memory vs device memory. |
| `ple_gather_bench2.py` | engram row-gather strategies: memmap page faults vs `pread` vs `O_DIRECT`. |
| `mincore.py` | page-cache residency of the engram table. |
| `cubin_audit.sh` | which GPU architectures the loaded kernels actually target. |

## Supporting the open questions

`token_freq.py` builds the frequency-ranked vocabulary (rebuild it on your own traffic).
`route_overlap.py` and `lru_sim.py` analyse logs from the instrumentation patches; see
`patches/README.md` for their status. `moe_kernel_bench.py` computes the expert-footprint
arithmetic; its direct kernel-timing half does not yet bind to the fused MoE APIs correctly.
