# Measurement log

Raw records, newest last. Every number is from the box (NVIDIA GB10, sm_121, 121 GB unified).

---

## 2026-09-12 — Memory system characterisation

Script: `scripts/skinny_bench2.py`, run in the vLLM image, L2 flushed (256 MB) between timings,
CUDA-graph replay timing, best of 5.

```
device: NVIDIA GB10   L2 cache 25.17 MB   SMs 48
flush (256MB write):            199 GB/s
read 256MB fp32 sum:            241 GB/s
D2D copy 1 GiB:                 179 GB/s (read+write)
```

cuBLAS bf16 GEMM at the model's shapes, us and GB/s of weight traffic:

```
layer                        N      K       M=1        M=2        M=4        M=5        M=6
gdn in_proj_qkv          10240   2560  302.6/173  229.9/228  229.5/228  230.1/228  230.3/228
gdn in_proj_z             6144   2560  188.6/167  144.1/218  141.4/222  142.7/220  145.1/217
gdn out_proj              2560   6144  138.0/228  153.8/205  150.9/208  152.7/206  150.9/208
attn q_proj              12288   2560  362.0/174  279.1/225  274.3/229  274.3/229  273.9/230
attn kv_proj              1024   2560   32.1/163   22.7/231   24.5/214   24.2/217   27.3/192
attn o_proj               2560   6144  138.5/227  149.9/210  149.3/211  150.8/209  148.9/211
indexer qk                 640   2560   16.2/202   15.2/216   16.0/205   16.4/200   17.1/191
shared gate_up            1280   2560   40.0/164   31.9/205   29.2/225   30.0/219   30.2/217
shared down               2560    640   15.7/209   12.3/266   13.9/236   11.9/275   14.1/233
hc mix down                320  10240   30.4/216   30.6/214   32.3/203   31.0/211   31.0/211
hc mix up                10240    320   28.5/230   30.2/217   29.1/226   29.7/221   27.9/235
router                     512   2560   10.5/249   13.1/200   14.5/180   12.6/209   12.8/205
lm_head                 248320   2560 7290.0/174 5413.4/235 5429.8/234 5434.6/234 5446.1/233
```

Sum over one verify pass of all dense side-layer GEMMs, multiplied by layer counts:

```
M=1: 45.36 ms    M=2: 38.61 ms    M=4: 38.44 ms    M=5: 38.38 ms    M=6: 38.39 ms
(ideal at 230 GB/s: 37.0 ms)
```

**Finding A**: M=1 costs 18 % more than M=2 for the same bytes. Every MTP draft pass runs at
M=1. `lm_head` alone: 7.29 ms at M=1 vs 5.41 ms at M>=2.

fp8 weight paths on the same 10240x2560 shape:

```
naive fp8->bf16 cast + gemm, M=1:  579 us  (45 GB/s effective)   <- dequant dominates
naive fp8->bf16 cast + gemm, M=4:  599 us  (44 GB/s effective)
torch._scaled_mm fp8 x fp8, M=16:  127 us  (206 GB/s effective)  <- viable
```

**Finding B**: a naive dequant-then-GEMM is 4.5x worse than bf16. fp8 only pays off through a
real fp8 GEMM (`_scaled_mm` or the blockwise-fp8 CUTLASS path).

---

## 2026-09-12 — Unified memory is not uniform (H3.1)

Script: `scripts/zerocopy_gather_bench.py`. A 4 GB pinned host table, a custom CUDA gather
kernel reading it through the device pointer from `cudaHostGetDevicePointer`.

```
device ptr for host table: 0xed5ea0000000 (identical to the host pointer => one address space)

R=180 rows per step:
  A  today (D2H ids + CPU gather + H2D):        32 us
  B  zero-copy GPU gather from pinned host:     25 us   (1.3x faster than A)
  C  GPU gather from a device-resident table:    8 us   (upper bound)
  B output matches a CPU gather: True

random-row read bandwidth from host memory, as seen by the GPU:
  R=64:     0.75 GB/s     R=1024:   1.57 GB/s     R=16384:  1.63 GB/s
  R=256:    1.24 GB/s     R=4096:   1.64 GB/s     R=65536:  1.65 GB/s

GPU sequential read of 1 GiB pinned host memory:  77.3 GB/s
cudaMemcpy H2D of the same 1 GiB:                 59.1 GB/s
```

**Finding C**: the GPU reads pinned host memory at 77 GB/s versus 241 GB/s for its own
allocations — a 3.1x penalty on physically identical LPDDR5X. `cudaMemcpy` H2D is worse still
at 59 GB/s. Offloading weights to "host" memory on GB10 is not free.

**Finding D**: the transfer half of the engram lookup is *not* a bottleneck — 25-32 us against
3-40 ms of disk I/O. Optimisation effort belongs entirely on the I/O path, not the copy.

---

## 2026-09-12 — Engram gather strategies (H4.1)

Script: `scripts/ple_gather_bench2.py`, run in the image with the real 10-shard, 25M-row,
51 GB table, persistent thread pools (v1 rebuilt a pool per call and had to be discarded).

```
=== uniform random rows (worst case), cold ===
  memmap fancy   t=8    median 40.97 ms   p90 50.13
  memmap fancy   t=32   median 31.38 ms   p90 35.09      <- production setting
  memmap fancy   t=128  median 26.28 ms   p90 31.58
  pread          t=8    median  3.92 ms   p90  4.69      <- best
  pread          t=32   median  5.56 ms   p90  6.44
  pread          t=180  median  5.70 ms   p90  6.89
  pread O_DIRECT t=32   median  7.32 ms
  pread coalesced runs  median  5.40 ms

=== zipf-ish (80 % of lookups from a 1 % hot region), cold ===
  memmap fancy   t=8    median  8.06 ms   p90 14.46
  memmap fancy   t=128  median  6.78 ms   p90  8.55
  pread          t=8    median  4.47 ms   p90  5.39
  pread          t=128  median  4.73 ms   p90  5.70

=== same rows repeated (fully warm) ===
  memmap fancy   t=32   median  3.03 ms
  pread          t=32   median  3.86 ms
```

**Finding E**: on cold rows, explicit `pread` beats page-faulting memmap by **6.7x** at its best
config and **8x** against the production setting. On warm rows memmap is 27 % faster (no
syscall). Optimal thread counts move in opposite directions for the two strategies.

**Finding F**: `O_DIRECT` is worse than buffered `pread` (7.3 vs 5.6 ms at the same thread
count) because each 160-byte row still forces a 4 KiB aligned read and forfeits all caching.

For reference, the production engine logs `gather 4.00 ms/op, op 5.74 ms/op` on real traffic,
consistent with the warm-to-skewed regime.

---

## 2026-09-12 — Expert footprint arithmetic

Per MoE layer: 512 experts, top-10 routing, hidden 2560, intermediate 640, NVFP4 weights with
fp8 group-16 scales = **2.76 MB per expert per layer**, 1.42 GB per layer, 67.9 GB for 48 layers.

Distinct experts selected by M tokens under *uniform random* routing, and the implied cost:

| M | distinct experts | GB per 48-layer pass | ms at 240 GB/s |
|---|---|---|---|
| 1 | 10.0 | 1.33 | 5.5 |
| 2 | 19.8 | 2.63 | 11.0 |
| 3 | 29.5 | 3.91 | 16.3 |
| 4 | 38.7 | 5.14 | 21.4 |
| 5 | 48.2 | 6.40 | 26.7 |
| 6 | 57.2 | 7.60 | 31.7 |
| 8 | 74.7 | 9.91 | 41.3 |

This is the *upper bound*. Published measurements on other MoEs report real routing has strong
temporal locality (1.25-1.42x the single-token footprint at batch 1 rather than (K+1)x). If that
holds here, a 4-token verify costs ~7 ms of expert traffic rather than 21 ms. Measuring it on
this model is the open item; the MTP depth sweep bounds it end-to-end without instrumentation.

---

## 2026-09-12 — Baseline, MTP=3 (E1)

Fresh engine on port 9115, `--load-format safetensors`, cpuset 5-9,15-19 (big cores), cgroup
capped at 100 GB. Boot: 12 min (weights ~11 min, torch.compile 22.4 s, graph capture 13 s,
engine init 104 s). KV pool 644,732 tokens in 18.63 GiB.

```
case    tok/s   ms/step  tok/step  acceptance by position
prose    24.3     87.0     2.11    0.606 / 0.366 / 0.141
code     36.8     87.7     3.23    0.849 / 0.763 / 0.634
repeat   43.6     89.3     3.90    1.000 / 0.987 / 0.922
```

Identical `tok/step` and per-position acceptance to the production lane measured earlier
(greedy is deterministic), with step time 87 ms here versus 95 ms on the production lane — the
production lane shares the box with other services.

**Finding G**: step time is essentially constant across content classes (87.0 / 87.7 / 89.3 ms).
Throughput differences come entirely from acceptance. This confirms the step is fixed work:
one M=4 verify plus three M=1 draft passes, regardless of how many drafts survive.

Kernel selection at boot: `FLASHINFER_CUTLASS` NVFP4 MoE backend, `cuda` GDN decode kernel,
Triton/FLA GDN prefill, FlashAttention 2 for the vision tower, PIECEWISE CUDA graphs.

---

## 2026-09-12 — Where the 87 ms decode step actually goes (E2, torch profiler)

This build registers `/start_profile` only when a `ProfilerConfig` is passed on the command
line (`--profiler-config '{"profiler":"torch","torch_profiler_dir":...}'`). The documented
`VLLM_TORCH_PROFILER_DIR` environment variable alone leaves the routes 404.

Trace: steady-state code generation, MTP=3, greedy. GPU span 2314.7 ms, **busy 94.2 %**,
sum of kernels 2306 ms. Profiling inflates the step from 87.2 to 109.9 ms, so the table below
reports each kernel family's **share of GPU time** and applies that share to the clean 87 ms.

| kernel family | share of GPU time | ms of the 87 ms step |
|---|---:|---:|
| `cutlass_80_wmma_*_bf16_16x16_*` dense bf16 GEMMs (M>=2) | 44.6 % | **38.8** |
| `internal::gemvx::kernel` — cuBLAS GEMV, i.e. all M=1 work | 24.5 % | **21.3** |
| `cutlass ... Sm120ArrayTmaWarpSpecializedBlockScaled` — NVFP4 grouped MoE | 20.7 % | **18.0** |
| `fused_moe ... Sm80` — bf16 MoE in the MTP draft layer | 2.4 % | 2.1 |
| `gdn_decode_post_conv_mtp_kernel` | 1.3 % | 1.1 |
| QSA sparse attention (`_qsa_*`) | 0.9 % | 0.8 |
| everything else (norms, routing, elementwise, sampling) | 5.6 % | 4.9 |
| GPU idle | — | **5.0** |

Call counts confirm the attribution. Over the traced window: 1201 `gemvx` calls, 1920 NVFP4
grouped-GEMM calls (48 MoE layers x 2 GEMMs per verify), 300 `persistent_topk` calls
(12 full-attention layers), 684 GDN decode calls.

### Finding H — the draft passes are almost entirely `lm_head`

`gemvx` is cuBLAS's matrix-vector path, taken only at M=1, and the only M=1 work in a step is
the three MTP draft passes. Per draft pass there are ~20 GEMV calls (the MTP layer's
projections plus `lm_head`). Weight bytes say `lm_head` dominates that list by an order of
magnitude: 1.27 GB versus ~150 MB for the whole MTP layer, and the standalone microbenchmark
measured `lm_head` at M=1 at 7.29 ms.

So of the 21.3 ms/step spent at M=1, roughly **22 ms is three full-vocabulary `lm_head`
reads** — 3 x 7.29 ms — and under 1 ms per pass is everything else. Adding the verify pass's
own `lm_head` read at M=4 (5.4 ms), the output head accounts for **~27 ms of an 87 ms step,
31 % of decode, for a model whose entire active body is 6B parameters.**

H1.1 predicted 25-29 ms/step. Confirmed.

### Finding I — routing locality on this model is weak

The NVFP4 grouped MoE costs 18.0 ms per step for a single M=4 verify pass. At the measured
240 GB/s that is 4.3 GB of expert traffic. Uniform random routing over 512 experts with top-10
predicts 38.7 distinct experts per layer = 5.14 GB = 21.4 ms.

Measured traffic is therefore only ~15 % below the random-routing bound. Published results on
other MoEs report much stronger temporal locality (1.25-1.42x the single-token footprint rather
than (K+1)x). **That does not reproduce here**: with 512 experts and top-10 routing, four
consecutive tokens really do touch most of ~38 distinct experts. Expert traffic on this model
scales close to linearly with speculative depth, which makes depth genuinely expensive.

Caveat: this bound cannot separate "many distinct experts" from "inefficient kernel" on its own.
An eager-mode routing log (`patches/expert_route_log.py`) separates them directly.

### Finding J — GPU idle matches the engram stall

5.8 % of the window is GPU-idle, i.e. ~5.0 ms per step. The engram (PLE) custom op is a CPU-side
gather that blocks the stream, and the engine's own counters report `gather 4.00 ms/op`. The
idle time and the gather time agree, which makes the engram gather the whole of the non-kernel
cost of a step — and fixing it (Finding E: `pread` instead of page faults; or hoisting it off
the critical path) is worth up to 5 ms/step.

### Implied opportunity

| change | expected ms/step | mechanism |
|---|---:|---|
| baseline | 87 | |
| FR-Spec 32K draft head | **-19** | 3 draft `lm_head` reads: 1.27 GB -> 0.17 GB each |
| engram gather: `pread` + prefetch | -4 | removes the 5 ms stall |
| fp8 side layers (hybrid checkpoint) | -12 est. | halves part of the 38.8 ms dense GEMM |
| total | **~52** | 1.7x single-stream |


---

## 2026-09-12 — FR-Spec draft head: 20.5 ms of GPU work removed (E3/E4)

A frequency-ranked draft vocabulary was built from 13 M tokens of mixed corpus (vLLM +
transformers Python sources, 15 Gutenberg books, markdown docs) with
`scripts/token_freq.py`. Coverage of the top-K tokens, ranked on the union corpus:

| K | code | prose | docs |
|---|---|---|---|
| 8,192 | 0.9650 | 0.8846 | 0.8130 |
| 16,384 | 0.9882 | 0.9535 | 0.9069 |
| **32,768** | **0.9984** | **0.9922** | **0.9791** |
| 65,536 | 1.0000 | 1.0000 | 1.0000 |

At K=32,768 the draft head becomes a [32768, 2560] bf16 slice: **168 MB instead of 1.27 GB.**

Correctness of the slice was verified directly against the checkpoint's `lm_head.weight`:
sliced logits are bit-identical to the full logits restricted to those ids
(`torch.equal` -> True), and argmax over the subset agrees with argmax-over-restricted-full at
1.000.

### GPU time, measured by profiler, per decode step

| kernel family | baseline | FR-Spec 32K | delta |
|---|---:|---:|---:|
| `wmma` dense bf16 GEMMs | 40.22 ms | 39.92 ms | -0.3 |
| **`gemvx` (all M=1 work)** | **22.12 ms** | **4.56 ms** | **-17.6** |
| NVFP4 grouped MoE | 19.14 ms | 17.96 ms | -1.2 |
| bf16 MoE (MTP layer) | 3.11 ms | 2.32 ms | -0.8 |
| GDN / QSA / other | 7.56 ms | 6.87 ms | -0.7 |
| **sum of GPU kernel time** | **92.15 ms** | **71.63 ms** | **-20.5** |

GEMV calls longer than 4 ms — the full-vocabulary `lm_head` reads — fall from **61 calls /
501.6 ms** to **1 call / 9.1 ms**. The mechanism works exactly as intended.

### But wall-clock barely moved, and that is the interesting part

Measured step time only improved from 87.7 ms to 84.4 ms. Removing 20.5 ms of GPU work bought
3 ms of wall clock, because the step stopped being GPU-bound: GPU busy falls from ~94 % to
~84 %, exposing roughly 13 ms per step of host-side serial work that had previously been hidden
underneath the GEMMs. The largest component of that is the engram gather, which runs as a
blocking CPU-side custom op on the decode critical path.

**Finding K**: on this system the output head and the engram gather are *coupled* bottlenecks.
Fixing either alone yields a fraction of its apparent value; the head optimisation only pays
off once the engram gather is taken off the critical path. This is the kind of interaction a
pure kernel-level analysis misses, and it is why the earlier byte-budget projection
(87 -> 68 ms from FR-Spec alone) was wrong.

### A silent failure worth documenting

The first FR-Spec attempt patched only the `Qwen3_8FlashNextMTPProposer` subclass. The instrumented
counters reported `mapped=0 unmapped=0`: neither sampling path ran, because that module is not
imported in the engine process the way the drafter is constructed. The drafter therefore emitted
*reduced-vocabulary indices as if they were token ids* — proposing token 5 when it meant the 5th
most frequent token.

The failure is invisible in the output: the target model verifies over the full vocabulary and
rejects the garbage, so generated text stays correct. Only acceptance length gives it away,
falling from 3.23 to 1.16 on code. Any FR-Spec-style implementation needs an explicit
mapped/unmapped counter, because "wrong draft ids" and "drafts that happen to be rejected" are
indistinguishable from the output side.


---

## 2026-09-12 — Speculation is the single biggest lever (E7, MTP=0)

Speculative decoding disabled entirely (`--speculative-config` omitted):

```
case    tok/s   ms/step  tok/step
prose    15.0     66.5     1.00
code     15.1     66.2     1.00
repeat   15.0     66.5     1.00
```

Against the MTP=3 baseline (24.3 / 36.8 / 43.6 tok/s), the model's own MTP head is worth
**1.6x on prose, 2.4x on code, 2.9x on verbatim copy**. It also confirms that step time is
content-independent: without speculation every case lands within 0.3 ms of 66.3 ms.

### Reconciling the cost model

A step at MTP=3 does one 4-token verify plus three 1-token drafts. Measured cost of that over
MTP=0 is **87.7 - 66.3 = 21.4 ms**. The naive byte model predicts far more:

| term | naive prediction |
|---|---:|
| dense side layers, M=1 -> M=4 | **-6.9 ms** (M=4 is *cheaper*: the M=1 GEMV penalty disappears) |
| expert traffic, 10 -> 38.7 distinct experts (random routing) | +15.9 ms |
| three draft passes (`lm_head` 7.29 ms + MTP layer ~0.9 ms each) | +24.6 ms |
| **total** | **+33.6 ms** vs 21.4 measured |

The gap closes only if a 4-token verify touches roughly **22 distinct experts per layer, not
38.7** — i.e. routing does have meaningful temporal locality on this model, around 2.2x the
single-token footprint rather than the 3.9x that independent routing implies. This revises
Finding I, which was derived from a trace taken on a cold engram cache (see below).

Note the sign of the first term: because cuBLAS is 18 % slower at M=1 than at M>=2, **widening
the verify batch makes the dense two-thirds of the model cheaper.** Speculation on this machine
is not only a token-per-step win, it also moves the dense GEMMs onto a faster kernel path.

### Caution about profiler traces on this system

The baseline trace (E2) was captured shortly after boot with a cold engram page cache. Its
per-step totals (121 ms of kernel time, 115 ms union-busy) are ~30 % above the same
configuration measured without the profiler (87.7 ms). The inflation is not CPU overhead: the
GPU *kernels themselves* are slower, because NVMe reads filling the page cache contend for the
same LPDDR5X that feeds the GPU. On a unified-memory machine, background I/O steals GPU
bandwidth. Traces intended for kernel accounting must be taken on a warmed cache.

### Open question

FR-Spec removes 19-23 ms/step of measured GPU kernel time (61 `lm_head` GEMV calls of ~8 ms
each fall to 1), acceptance is preserved within 1-3 %, and yet end-to-end throughput is
unchanged: 36.8 -> 36.2 tok/s on code, step 87.7 -> 86.4 ms. Either the removed work was not on
the critical path, or an equal amount of host-side serial time is exposed once it goes. The gap
analysis attributes the largest GPU idle to `vllm::ple_mmap_lookup` (2.2 ms per call, one per
step) and `aten::copy_`, which together are only ~4 ms/step - not enough to explain 20.
Resolving this is the next priority; E8 tests it by spending the freed headroom on deeper
speculation (MTP=5), which should be nearly free if the headroom is real.

---

## 2026-09-12 — Speculative depth is constrained by a block-size divisibility assertion (E8)

`MTP=5` fails to boot:

```
Setting attention block size to 1616 tokens to ensure that attention page size is >= mamba page size.
AssertionError: QSA ring capacity 12 must divide the attention block size 1616
```

The QSA raw-key ring capacity is tied to the speculative width (8 at MTP=3, 12 at MTP=5), while
the attention block size is chosen independently to cover the Mamba page size, which also grows
with the speculative width (800 at MTP=3, 1616 at MTP=5). Nothing makes the two divide.

So the usable speculative depths are not a free parameter: only those where the auto-selected
block size happens to be a multiple of `2*(k+1)` will start. The workaround is an explicit
`--block-size` that satisfies both (for k=5: 1632 = 12 x 136 = 16 x 102). Worth reporting
upstream, since the failure is a bare assertion at startup with no suggested remedy.

---

## 2026-09-13 — Methodology: two confounds that invalidate cross-time comparisons

Re-running the *identical* baseline configuration hours later produced 106.2 ms/step on code
against 87.7 ms/step earlier, with bit-identical acceptance (3.23 tokens/step, per-position
0.849/0.763/0.634 — greedy decoding is deterministic, so the model did exactly the same work).
Two effects account for it, and both must be controlled.

### Confound 1: engram page-cache warmth (worth up to ~18 ms/step)

The engine's own counters, sampled every 30 s from a cold start:

```
op 11.63 ms/op, gather 9.80 ms/op     <- first traffic after container start
op  7.78 ms/op, gather 5.86 ms/op
op  3.30 ms/op, gather 1.55 ms/op
op  2.12 ms/op, gather 0.29 ms/op     <- steady state, after ~3 passes of the suite
```

The engram gather blocks the CUDA stream, so this lands directly on step time. Steady state
took three full passes of the benchmark suite (~9 requests per case). **Any measurement taken
before the gather stabilises below ~0.5 ms/op is measuring page-cache state, not the change
under test.**

### Confound 2: the desktop session (worth ~20 %)

With a GNOME session logged in and compositing (gnome-shell at 55 % CPU, Xorg and
gnome-remote-desktop active, all sharing the same LPDDR5X), warm steady-state throughput is
30.4 tok/s on code. With the box idle after a fresh power cycle the same build measured
36.8 tok/s. That is a **21 % difference from the desktop alone**, on a machine where the GUI's
GPU usage is only ~400 MB and a few percent utilisation — because what it actually costs is
memory bandwidth, the resource decode is bound by.

**Protocol adopted**: every A/B is run back-to-back within the same session, after the engram
gather has stabilised, and reported with the paired control measured the same hour. Numbers
from different hours are not comparable on this machine.

---

## 2026-09-13 — Warm, paired profiler comparison: FR-Spec removes 25.75 ms/step (E9 vs E6)

Both traces warm-cache, 27 decode steps each, same prompt and decode settings.

| kernel family | baseline ms/step | FR-Spec ms/step | delta |
|---|---:|---:|---:|
| `wmma` dense bf16 GEMMs | 53.52 | 51.18 | -2.34 |
| **`gemvx` (all M=1 work)** | **28.25** | **5.82** | **-22.43** |
| NVFP4 grouped MoE | 24.05 | 23.42 | -0.63 |
| bf16 MoE (MTP layer) | 3.42 | 3.13 | -0.29 |
| GDN / QSA / nvjet / other | 9.76 | 9.13 | -0.63 |
| **kernel-sum** | **118.98** | **92.68** | **-26.31** |
| **union-busy** | **112.33** | **86.58** | **-25.75** |

Crucially, **both configurations are GPU-bound**: union-busy (112.33 / 86.58) matches the wall
step time measured during the same run (109.1 / 86.7). There is no large hidden host-side
stall. The earlier "FR-Spec saves GPU time but not wall time" puzzle was an artefact of
comparing an FR-Spec run against a baseline measured under different box conditions.

The `gemvx` figure lands where the standalone microbenchmark predicted: baseline 28.25 ms/step
is three draft `lm_head` reads at ~7.3 ms plus the MTP layer's own M=1 projections; FR-Spec's
5.82 ms is the same minus 3 x 1.27 GB of weight traffic.

---

## 2026-09-13 — HEADLINE: FR-Spec gives +19-24 % single-stream throughput (E9 vs E10, paired)

Both configurations measured back-to-back in the same session, on the same box state, each
warmed to a stable engram gather (0.26-0.29 ms/op), median of the third pass. Greedy decoding,
300 generated tokens, thinking off.

| case | baseline tok/s | FR-Spec tok/s | gain | baseline ms/step | FR-Spec ms/step | tok/step base -> FR-Spec |
|---|---:|---:|---:|---:|---:|---|
| prose | 20.0 | **24.8** | **+24.0 %** | 105.4 | 84.6 | 2.11 -> 2.10 (-0.5 %) |
| code | 30.4 | **36.2** | **+19.1 %** | 106.2 | 86.3 | 3.23 -> 3.12 (-3.4 %) |
| verbatim copy | 35.8 | **43.6** | **+21.8 %** | 108.8 | 88.2 | 3.90 -> 3.85 (-1.3 %) |

Step time falls by 19.9 ms on code, against the 25.75 ms of GPU work the paired profiler
comparison says was removed — the rest is absorbed by the slightly lower acceptance, which
costs a few more steps for the same 300 tokens.

**The trade is strongly favourable**: a 0.5-3.4 % reduction in accepted tokens per step buys a
19 % reduction in the cost of every step. Output remains exactly correct by construction — the
target model still verifies over the full 248,320-token vocabulary, and rejection sampling is
untouched. Only the drafter's candidate set is restricted.

### Why it is worth so much more here than in published results

The llama.cpp prototype of the same idea on Qwen3.6-27B measured an 85 % reduction in draft
`lm_head` kernel time but only +1.4-3.1 % end to end. The difference is the **head-to-body byte
ratio**. That model is dense: the body dominates each step, so the head is a rounding error.
Flash-Next activates ~6B parameters per token but shares a 1.27 GB output head, and at
speculative depth 3 that head is read four times per step. It is 28 ms of a 106 ms step.

The generalisable statement: *FR-Spec's value scales with (vocabulary x hidden x
(1 + draft depth)) divided by active body bytes.* Sparse-MoE models with large vocabularies and
native multi-token-prediction heads are exactly where it pays, and that describes most of the
current frontier-model architectures being run on single devices.

### Configuration

```
VLLM_MTP_DRAFT_VOCAB=/research/vocab/draft_vocab_top32768.npy
```
with `patches/frspec_mtp.py` injected into `Qwen3_8FlashNextMTP.compute_logits` by
`entrypoint_patched.sh`. The draft head is a [32768, 2560] bf16 slice of `lm_head` (168 MB),
built once at first use; logits are scattered back to full vocabulary width against a -inf
buffer so that every downstream sampling path sees correct token ids without further patching.

---

## 2026-09-13 — The SM12x fused MoE kernel is unreachable on this model (E11)

`--moe-backend flashinfer_b12x` selects FlashInfer's SM120/SM121-specific fused MoE
(`b12x_fused_moe`, one kernel for dispatch + both GEMMs + SwiGLU + reduction), which is
excluded from vLLM's auto-selection pending an upstream CUTLASS SM121 guard. It is the obvious
candidate for the 23.4 ms/step spent in the NVFP4 grouped GEMM.

It binds correctly to the main MoE:

```
Using 'FLASHINFER_B12X' NvFp4 MoE backend out of potential backends: [...]
```

and then the engine dies ten minutes later, during the MTP draft layer:

```
ValueError: moe_backend='flashinfer_b12x' is not supported for unquantized MoE.
Expected one of ['triton', 'batched_triton', 'flashinfer_trtllm', 'flashinfer_cutlass', ...]
```

**Root cause**: the checkpoint quantises the 48 main MoE layers to NVFP4 but leaves the MTP
draft layer's experts in bf16 (`mtp.layers.0.mlp.experts.gate_up_proj` 3.36 GB +
`down_proj` 1.68 GB). `--moe-backend` is a single global setting, so asking for an
NVFP4-only kernel breaks the bf16 layer that shares the flag.

Two ways out, both worth doing:
1. **Per-layer backend selection** — the MoE oracle already resolves a backend per layer from
   the quant config; the user-facing override should be resolvable per layer too. Upstreamable.
2. **Quantise the MTP experts to NVFP4** offline. That unblocks b12x *and* removes ~0.25 GB of
   weight traffic per step in its own right (the bf16 MTP MoE currently costs 3.13 ms/step).

Until then the SM12x fused kernel cannot be evaluated on this model, and the 23.4 ms of expert
GEMM stays on the generic CUTLASS path.

---

## 2026-09-13 — HEADLINE: fp8 side layers + FR-Spec = +50 % single-stream (E12)

The paired profiler comparison left dense bf16 GEMMs as the largest remaining term: 51.2 ms of
an 86.6 ms step. The upstream "hybrid" recipe rewrites the side layers that every decoded token
reads in full — GDN in/out projections, QSA q/k/v/o, shared experts, ~15 GiB of bf16 — as
blockwise fp8-e4m3 (128x128 blocks, DeepSeek layout), leaving the NVFP4 experts untouched.

Prepared with the upstream serving repo's `prepare-hybrid.sh` (not part of this repo) against the snapshot: 300 tensors across
4 shards converted, **worst per-tensor max relative error 0.0354**, +12 GB disk, ~6 min. The
converted snapshot is a sibling directory of relative symlinks, so the original is untouched.

All three configurations measured in the same session, warmed to a stable engram gather
(0.30 ms/op), median of the third pass, greedy, 300 tokens:

| config | prose | code | verbatim | tokens/step (code) |
|---|---:|---:|---:|---:|
| baseline | 20.0 | 30.4 | 35.8 | 3.23 |
| + FR-Spec 32K | 24.8 | 36.2 | 43.6 | 3.12 |
| **+ FR-Spec + fp8 side layers** | **30.9** | **45.5** | **52.2** | **3.30** |
| **total gain** | **+54.5 %** | **+49.7 %** | **+45.8 %** | |

Step time on code: 106.2 -> 86.3 -> **72.5 ms**.

Two things worth noting.

**Acceptance went up, not down.** Tokens per step rose from 3.12 (FR-Spec alone) to 3.30 with
fp8 side layers, above even the 3.23 baseline. Draft and target share the same quantised side
layers, so the perturbation is common-mode: it changes what the model says slightly, but does
not desynchronise the drafter from the target. The combination therefore recovers the small
acceptance cost FR-Spec introduced.

**The levers compose.** FR-Spec attacks the M=1 GEMV term (28.3 -> 5.8 ms/step), fp8 attacks
the M=4 dense GEMM term. They target disjoint parts of the step, and the measured combination
(-33.7 ms) is close to the sum of the parts (-19.9 and ~-14).

Validity checks: box state matched the paired baseline (gnome-shell 58 % CPU, load ~1.9, SM
clock 2424 MHz, 51 C); the loaded snapshot really contains 300 fp8 tensors in its side-layer
shards; output on a held-out prompt is coherent and correct. A proper quality evaluation
(top-1 agreement against bf16 over a fixed prompt set, plus verbatim-recall and needle tests)
is still owed before recommending fp8 for production.

---

## 2026-09-13 — Full ablation: the levers compose additively (E13)

fp8 side layers measured alone, same session and protocol as the other cells:

```
case    tok/s   ms/step  tok/step
prose    24.1     91.5     2.21
code     36.7     92.9     3.41
repeat   41.5     95.1     3.95
```

Completing the 2x2:

| config | step time (code) | delta | tok/step |
|---|---:|---:|---:|
| baseline | 106.2 ms | — | 3.23 |
| FR-Spec only | 86.3 ms | -19.9 | 3.12 |
| fp8 only | 92.9 ms | -13.3 | 3.41 |
| both | 72.5 ms | **-33.7** | 3.30 |

Sum of the individual savings is -33.2 ms; measured together, -33.7 ms. **They compose to
within 1.5 %.** The two act on disjoint terms — FR-Spec on the M=1 GEMV work in the three draft
passes, fp8 on the M=4 dense GEMMs of the verify pass — so on a purely bandwidth-bound decoder
the step-time model is additive in bytes moved. That is worth stating explicitly because it
means the remaining levers can be evaluated independently and their gains summed, which is not
true once a step becomes latency- or launch-bound.

fp8 alone is also the only change so far that *raises* acceptance (3.23 -> 3.41).
