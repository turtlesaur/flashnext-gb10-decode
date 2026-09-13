# Results summary

All numbers from one NVIDIA GB10 (DGX Spark / GX10, sm_121, 121 GB LPDDR5X unified) serving
Qwen3.8-Flash-Next-NVFP4 under vLLM 0.1.dev20073, MTP speculative decoding at
depth 3, greedy, 300 generated tokens, batch 1.

Every row in the main table was measured **in the same session, back-to-back, after the engram
gather had warmed to a steady 0.30 ms/op**. On this machine that protocol is not optional: the
page-cache state is worth up to 18 ms/step and an active desktop session is worth 21 %.

## Single-stream throughput (tokens/s)

| config | prose | code | verbatim copy | ms/step (code) | tokens/step (code) |
|---|---:|---:|---:|---:|---:|
| no speculation (MTP=0) | 15.0 | 15.1 | 15.0 | 66.3 | 1.00 |
| **baseline** (MTP=3) | 20.0 | 30.4 | 35.8 | 106.2 | 3.23 |
| FR-Spec 32K draft head | 24.8 | 36.2 | 43.6 | 86.3 | 3.12 |
| fp8 side layers | 24.1 | 36.7 | 41.5 | 92.9 | 3.41 |
| **FR-Spec + fp8 side layers** | **30.9** | **45.5** | **52.2** | **72.5** | 3.30 |
| gain over baseline | **+54.5 %** | **+49.7 %** | **+45.8 %** | -31.7 % | |

### The two levers compose additively

| config | step time | delta vs baseline |
|---|---:|---:|
| baseline | 106.2 ms | |
| FR-Spec only | 86.3 ms | -19.9 |
| fp8 only | 92.9 ms | -13.3 |
| both | 72.5 ms | **-33.7** (sum of parts: -33.2) |

They attack disjoint terms of the step — FR-Spec the M=1 GEMV work in the draft passes, fp8 the
M=4 dense GEMMs in the verify pass — so their savings add with 1.5 % error. That is the useful
engineering statement: on a bandwidth-bound decoder, byte reductions in different parts of the
step compose, and the step-time model is simply additive in bytes moved.

Note also that **fp8 raises acceptance**, from 3.23 to 3.41 tokens/step. Draft and target share
the quantised side layers, so the perturbation is common-mode and does not desynchronise them;
the slightly different text it produces happens to be marginally more predictable. This is why
the combination (3.30) recovers most of the acceptance FR-Spec alone gave up (3.12).

Output correctness is preserved by construction for FR-Spec: the target model still verifies
over the full 248,320-token vocabulary and rejection sampling is untouched. The fp8 side layers
do change the model's numerics (worst per-tensor max relative error 0.0354) and still owe a
formal quality evaluation.

## Where the 106 ms baseline step goes

From paired torch-profiler traces, warm cache, 27 decode steps each.

| kernel family | baseline | + FR-Spec | what it is |
|---|---:|---:|---|
| `wmma` dense bf16 GEMMs | 53.5 ms | 51.2 ms | GDN, hyper-connections, attention, shared experts, verify `lm_head` |
| `gemvx` (all M=1 work) | 28.3 ms | 5.8 ms | the three draft passes — almost entirely `lm_head` |
| NVFP4 grouped MoE | 24.1 ms | 23.4 ms | 10 of 512 experts per token, 48 layers |
| bf16 MoE (MTP layer) | 3.4 ms | 3.1 ms | the draft layer's experts, left unquantised in the checkpoint |
| GDN / QSA / other | 9.8 ms | 9.1 ms | |
| **union-busy** | **112.3 ms** | **86.6 ms** | both GPU-bound: this matches wall time |

The single largest surprise: a model with **6B active parameters** spends **28 ms per step
re-reading its 1.27 GB output head**, because the MTP drafter shares it and reads it once per
draft at batch 1.

## Machine characterisation

| quantity | measured |
|---|---|
| GPU read bandwidth, device memory | 241 GB/s (vendor: 273) |
| GPU read of pinned *host* memory, zero-copy | 77 GB/s |
| `cudaMemcpy` host-to-device | 59 GB/s |
| cuBLAS bf16 GEMM efficiency, M>=2 | 205-230 GB/s (~96 % of roofline) |
| cuBLAS bf16 GEMM efficiency, M=1 | 163-174 GB/s (**18 % penalty**) |
| engram row gather, cold, page-fault memmap (ships today) | 26-41 ms / 180 rows |
| engram row gather, cold, `pread` with 8 threads | **3.9 ms** / 180 rows |
| engram row gather, warm | ~3 ms (memmap) / 4 ms (pread) |

## Negative and blocked results

- **`flashinfer_b12x`** (the SM120/SM121-specific fused MoE) binds correctly to the NVFP4
  experts and then kills the engine on the MTP layer's bf16 experts, because `--moe-backend`
  is a single global setting. Needs a per-layer override or an NVFP4 MTP checkpoint.
- **MTP=5** does not boot: `QSA ring capacity 12 must divide the attention block size 1616`.
  Speculative depth is only usable where the auto-chosen block size is a multiple of `2*(k+1)`.
- **Naive fp8** (dequantise to bf16, then GEMM) is 4.5x *slower* than staying in bf16. fp8 only
  pays through a real fp8 GEMM path.
- **`O_DIRECT`** is worse than buffered `pread` for the engram gather (7.3 vs 5.6 ms): a
  160-byte row still forces a 4 KiB aligned read and forfeits all caching.
- **Zero-copy GPU gather** from pinned host memory works and is correct, but saves nothing that
  matters (25 us vs 32 us) — the engram cost is entirely disk I/O, not transfer.

## Reproduction

```bash
# on the box
cd ~/research
PATCHED=1 HYBRID=1 MTP=3 CPUSET=5-9,15-19 \
  ENV_EXTRA="-e VLLM_MTP_DRAFT_VOCAB=/research/vocab/draft_vocab_top32768.npy" \
  bash exp.sh my-run
# warm to steady state, then measure
for r in 1 2 3; do python3 bench_steps.py --base http://127.0.0.1:9115 \
  --cases prose,code,repeat --reps 2 --max-tokens 300 --label my-run-p$r; done
```
