# Why a 6B-active MoE spends a quarter of every decode step in its output head

A measurement study of single-stream decoding for **Qwen3.8-Flash-Next** — a 176B-parameter
hybrid linear-attention MoE with ~6B active parameters and a 51 GB n-gram ("engram") embedding
table served from NVMe — on a single **NVIDIA GB10** (DGX Spark class, sm_121, 121 GB unified
LPDDR5X), under vLLM with the model's native MTP speculative decoding.

## Read this first: what is and is not new here

**The headline optimisation in this repo is not novel.** Two existing projects already ship a
reduced draft vocabulary for this model, both using 65,536 tokens:

- [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX) added `DRAFT_VOCAB`
  on **2026-09-08** and made it the **default** (+20 % decode, identical outputs).
- [jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000) ("Pennyroyal")
  ships FR-Spec with a 65,536-token map on RTX PRO 6000 (+9.7-16.8 % single-request decode).

Both predate this work. The build measured here is pinned to the Flash-DGX tree at
**2026-09-06**, two days before that feature landed, which is why the baseline still reads the
full vocabulary — verified directly: `src/patch_mtp_draft_vocab.py` is absent from the pinned
clone, and the profiler shows four full-head reads per step.

So treat the draft-vocabulary result below as an **independent replication**: +19.1 % on code
against their +20 %, arrived at separately, which is worth something as corroboration but is not
a discovery.

What this repo adds that I have not found elsewhere:

1. **A kernel-level account of *why* it is worth ~20 %** — the output head is 28.3 ms of a
   106 ms step, and exactly which cuBLAS calls disappear. Both upstream projects report the
   end-to-end number without the decomposition.
2. **K=32,768 may be enough** — half the vocabulary both projects use. Measured: 0.5-3.4 %
   acceptance cost on three workloads. Caveat in [COMPARISON.md](COMPARISON.md): their
   vocabulary is built more robustly than this one, so this is not a clean claim that half of
   *their* set would do.
3. **The two levers compose additively to within 1.5 %**, measured as a full 2x2 ablation.
4. **`pread` beats the shipping page-fault gather 6.7x on cold engram rows.** Flash-DGX's own
   documentation notes the I/O mechanism is not specified; this measures it.
5. **"Unified" memory is not uniform**: 241 vs 77 vs 59 GB/s on one physical pool.
6. **Two benchmarking confounds worth ~20 % each**, one of which (a logged-in desktop session)
   I have not seen documented anywhere.
7. **Negative results**, including a silent failure mode that leaves output correct while
   quietly destroying acceptance.

---

## The measurements

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/throughput-dark.svg">
  <img alt="Single-stream throughput by configuration" src="docs/figures/throughput-light.svg">
</picture>

| config | prose | code | verbatim copy | ms/step (code) |
|---|---:|---:|---:|---:|
| no speculation | 15.0 | 15.1 | 15.0 | 66.3 |
| **baseline** (MTP depth 3, full draft vocabulary) | 20.0 | 30.4 | 35.8 | 106.2 |
| reduced draft vocabulary (32K) | 24.8 | 36.2 | 43.6 | 86.3 |
| fp8 side layers | 24.1 | 36.7 | 41.5 | 92.9 |
| **both** | **30.9** | **45.5** | **52.2** | **72.5** |
| | **+54.5 %** | **+49.7 %** | **+45.8 %** | **-31.7 %** |

## 1. Where the step actually goes

Decode here is bandwidth-bound: GPU utilisation sits at 92-96 % and achievable read bandwidth is
241 GB/s, so step time is essentially "bytes moved / bandwidth".

The model activates ~6B parameters per token, but its MTP drafter **shares the target's
248,320 x 2560 bf16 output head — 1.27 GB** — and at speculative depth 3, batch 1, that head is
read **four times per step**: once to verify, once per draft.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/step-anatomy-dark.svg">
  <img alt="Decode step broken down by kernel family" src="docs/figures/step-anatomy-light.svg">
</picture>

Every draft runs at M=1, landing in cuBLAS's GEMV path: **61 GEMV calls of ~8 ms each** in a
27-step trace, 501 of the 553 ms spent in GEMV. Restricting the drafter's candidate set to the
32,768 most frequent tokens — the target still verifies over the full vocabulary, so the output
distribution is unchanged — collapses that to **one call, 9.1 ms**:

| | baseline | reduced 32K |
|---|---:|---:|
| draft-pass GEMV time | 28.3 ms/step | **5.8 ms/step** |
| GEMV calls > 4 ms per trace | 61 (501.6 ms) | 1 (9.1 ms) |
| tokens/step, code | 3.23 | 3.12 |
| throughput, code | 30.4 tok/s | **36.2 tok/s** |

**On vocabulary size.** Both upstream projects use K=65,536. At K=32,768 corpus coverage is
99.8 % on code and 99.2 % on prose and acceptance falls only 0.5-3.4 %. But their vocabulary is
constructed more carefully than this one — theirs unconditionally includes every special and
added token plus the 256 byte-fallback ids, this one is corpus frequency plus a handful of
specials — so this is **not** a clean claim that half of their set would do. See
[COMPARISON.md](COMPARISON.md).

**On when this matters at all.** A prototype of the same idea on a dense 27B model in another
engine reported an 85 % cut in draft-head kernel time but only **+1.4-3.1 %** end-to-end. The
value scales as

```
  vocabulary x hidden x (1 + draft depth)   /   active body bytes
```

which is large exactly for **sparse MoE models with large vocabularies and native
multi-token-prediction heads**, and small for dense ones. That ratio, not the technique, is what
decides whether it is worth doing.

## 2. The two levers compose additively

The remaining bulk of the step is dense bf16 GEMMs. Converting the side layers every token reads
in full to blockwise fp8-e4m3 removes another 13 ms. **That recipe is Flash-DGX's `MODE=hybrid`,
measured here, not invented here.**

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/ablation-dark.svg">
  <img alt="Ablation waterfall of step time" src="docs/figures/ablation-light.svg">
</picture>

Measured alone the savings are -19.9 ms and -13.3 ms; together, **-33.7 ms** against a sum of
parts of -33.2 ms. They add to within 1.5 %, because they act on disjoint terms (M=1 GEMV in the
drafts, M>=2 GEMM in the verify) of a step whose cost is just bytes moved. That is the practical
payoff of establishing bandwidth-boundedness: remaining levers can be evaluated independently
and their gains summed.

fp8 also *raises* acceptance, 3.23 to 3.41 tokens/step — drafter and target share the quantised
layers, so the perturbation is common-mode and does not desynchronise them.

Pennyroyal goes further and quantises the **output head** online as well. Given the head is 28 ms
of the step here, that looks like the largest remaining lever on this hardware, and it is the
obvious next thing to measure.

## 3. "Unified" memory is not uniform

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/bandwidth-dark.svg">
  <img alt="Effective bandwidth by access path" src="docs/figures/bandwidth-light.svg">
</picture>

One physical LPDDR5X pool, three access paths. `cudaHostGetDevicePointer` returns the
**identical pointer** — a single address space — and the GPU still reads host-allocated memory
at under a third of device-local speed. "It's unified, just put it in host memory" costs 3-4x.

Related: cuBLAS reaches 205-230 GB/s (~96 % of roofline) at M>=2 but only 163-174 GB/s at M=1,
so **widening the verify batch makes the dense two-thirds of the model cheaper**. Speculation
pays twice on this hardware: once in tokens per step, once in kernel efficiency.

## 4. The engram gather is a page-fault problem

The 51 GB n-gram table is mmapped from NVMe and gathered every step. The shipping implementation
uses `np.memmap` fancy indexing, inline and single-threaded for decode-sized batches, so every
non-resident row becomes a **serialized synchronous page fault**.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/engram-gather-dark.svg">
  <img alt="Engram gather strategies" src="docs/figures/engram-gather-light.svg">
</picture>

Explicit `pread` on 8 threads is **6.7x faster on cold rows** and 0.9 ms slower when everything
is resident, so the right design is a hybrid: memmap for resident rows, batched `pread` for
misses. The *transfer* half of the lookup is irrelevant — 25-32 us against 3-40 ms of I/O — so a
zero-copy GPU gather, which works and is correct, buys nothing.

## 5. Two confounds that make cross-time comparison meaningless

Re-running the **identical** baseline hours later gave 106.2 ms/step against 87.7 ms earlier,
with bit-identical acceptance (greedy decoding is deterministic, so the model did exactly the
same work).

1. **Engram page-cache warmth**, up to ~18 ms/step: the gather goes 11.6 -> 9.8 -> 5.9 -> 1.6 ->
   0.3 ms/op over roughly three passes of the suite. Flash-DGX documents a cold region being
   "2-3x slower"; this quantifies the settling curve and how long it takes.
2. **A logged-in desktop session**, ~21 %. Its GPU footprint is ~400 MB and a few percent
   utilisation, but compositing competes for the memory bandwidth decode is bound by.

A profiler trace on a cold cache shows the GPU **kernels themselves** ~30 % slower, because NVMe
reads filling the page cache steal bandwidth. One early conclusion here ("the optimisation
removes GPU time but not wall time") was purely an artefact of comparing a warm experiment
against a cold baseline. See [METHOD.md](METHOD.md).

## Negative and blocked results

- **`flashinfer_b12x`**, the SM120/SM121-specific fused MoE kernel, binds to the NVFP4 experts
  and then kills the engine on the MTP layer's bf16 experts — `--moe-backend` is a single global
  setting. Needs a per-layer override. (Flash-DGX hit the mirror image of this with NVIDIA's
  checkpoint, whose MTP experts are blockwise fp8, and shipped a loader shim for it.)
- **Speculative depth 5** does not boot: `QSA ring capacity 12 must divide the attention block
  size 1616`. Only depths where the auto-chosen block size is a multiple of `2*(k+1)` work.
- **Naive fp8** (dequantise to bf16, then GEMM) is **4.5x slower** than bf16. fp8 pays only
  through a real fp8 GEMM path.
- **`O_DIRECT`** loses to buffered `pread` (7.3 vs 5.6 ms): a 160-byte row still forces a 4 KiB
  aligned read and gives up all caching.
- **A silent failure mode.** The first draft-vocabulary implementation patched a sampling path
  that was never called, so the drafter emitted reduced-vocabulary *indices as token ids*.
  Output stayed correct — the target rejects them — and only acceptance collapsed, 3.23 to 1.16.
  Any implementation of this needs an explicit mapped/unmapped counter; the output is not a
  sufficient test.

## Layout

```
patches/frspec_mtp.py     reduced draft vocabulary (independent implementation)
patches/ple_pread.py      engram gather via threaded pread instead of page faults
scripts/bench_steps.py    harness: reads the engine's own counters, not client timing
scripts/compare_traces.py paired profiler comparison, normalised per decode step
scripts/gap_analysis.py   attributes GPU idle to the CPU op spanning it
COMPARISON.md             how this differs from the two shipping implementations
RESULTS.md                every number, with conditions
FINDINGS.md               the full measurement log, in the order it happened
METHOD.md                 the benchmarking protocol and why it is mandatory
```

## Status and caveats

A measurement study from one machine, not a library. The patches are monkey-patches against one
vLLM build, written to make an effect measurable rather than to be merge-ready — if you want
these optimisations in production, use
[Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX), which ships them properly and has more
of them.

The fp8 side layers change numerics (worst per-tensor max relative error 0.0354) and this study
owes a formal quality evaluation — top-1 agreement against bf16, verbatim recall, long-context
needle tests — before recommending them; Flash-DGX reports an unchanged tournament score for the
same conversion, which is stronger evidence than anything here.

See [NOTICE.md](NOTICE.md) for full attribution.
