# Attribution

This is a measurement study layered on other people's work. None of it would have run at all
without the following, and where a finding here refines or contradicts one of them, that is a
consequence of having their working system to measure, not a criticism of it.

## The serving stack

- **vLLM** (Apache-2.0) — the engine. All patches here are runtime monkey-patches against one
  build; nothing is forked.
- **The Qwen3.8-Flash-Next community serving repositories** for DGX Spark / GB10, which solved
  the hard problem of making this model run on a single device at all:
  - serving the 51 GB n-gram ("PLE") embedding table from NVMe via `mmap`, which is what makes
    the model fit beside a usable KV cache;
  - the prefix-caching block-size fix, without which a cache hit silently restored an all-zero
    Mamba state;
  - a deterministic sparse-attention top-k kernel, replacing a GB10 kernel that dropped
    candidates non-deterministically;
  - the blockwise-fp8 side-layer conversion and the NVFP4-plus-fp8 dispatch that this study
    measures as its second optimisation;
  - the fp8 KV-cache path for the sparse-attention kernels;
  - GB10-specific fixes to the linear-attention kernels, including the shared-memory gate that
    otherwise silently halves tile sizes on sm_121.

  The fp8 side-layer result in this repo is a **measurement of their recipe**, not a new
  technique. The preparation script and the conversion tool are theirs, and their quality
  evidence for it (an unchanged agentic-tournament score) is stronger than anything measured
  here.

## The ideas

- **FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative
  Sampling** (ACL 2025, arXiv:2502.14856) — the frequency-ranked draft vocabulary.

- **Prior art for this exact model, which this study did not know about while measuring and
  therefore re-derived.** Both predate the work here and both should be considered the
  reference implementations:
  - `blazux/qwen3.8-Flash-DGX` shipped `DRAFT_VOCAB` on **2026-09-08** and made it the
    default: 65,536 tokens, +20 % decode, identical outputs, with
    `src/patch_mtp_draft_vocab.py` and a `tools/build_draft_vocab.py` for rebuilding the set.
    The build measured in this repo is pinned to that tree at 2026-09-06, two days earlier,
    which is the only reason the baseline still reads the full vocabulary.
  - `jpezzulli/sglang-rtxpro6000` ("Pennyroyal") ships FR-Spec with a 65,536-token map on
    RTX PRO 6000 / SM120, reporting +9.7-16.8 % single-request decode, and goes further by
    quantising the output head itself online.

  The draft-vocabulary number in this repo is therefore an **independent replication**
  (+19.1 % against their +20 %), not a discovery. What is added is the kernel-level
  decomposition explaining the size of the effect, evidence that half their vocabulary size is
  sufficient, and the observation about when the technique pays at all: in proportion to the
  output head's share of per-step bytes, which is large for sparse-MoE models with native
  multi-token-prediction heads and small for dense ones.
- A community prototype of the same idea for a native MTP head in another inference engine
  measured +1.4-3.1 % end-to-end on a dense 27B model. That result is the useful contrast for
  the +19 % measured here, and the reason this repo frames the effect as a byte-ratio argument.
- Work on speculative decoding for mixture-of-experts models — expert-aware draft selection,
  utility-driven adaptive draft length, and measurements of expert temporal locality — informed
  the expert-footprint analysis, including the finding that locality on this model is weaker
  than published figures for other MoEs.
- Independent DGX Spark benchmarking that established the bandwidth-ceiling framing (bytes read
  per token divided into achievable bandwidth) and separately found the CuteDSL fp4 path
  unsupported at compute capability 12.1.

## Corpus

The frequency-ranked vocabulary was built from open-source Python sources and public-domain
texts from Project Gutenberg. `scripts/token_freq.py` rebuilds it for any corpus; a deployment
should rank on its own traffic mix.

## Licence

Apache-2.0, matching the upstream repositories this builds on. See LICENSE.
