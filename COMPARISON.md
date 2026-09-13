# Comparison with the existing implementations

Two projects already ship a reduced draft vocabulary for this model. This file compares them
with the implementation in `patches/frspec_mtp.py`, which was written without knowledge of
either and therefore serves as an independent check on the design.

| | [Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX) `DRAFT_VOCAB` | [Pennyroyal](https://github.com/jpezzulli/sglang-rtxpro6000) FR-Spec | this repo |
|---|---|---|---|
| engine | vLLM | SGLang | vLLM |
| hardware | GB10 / DGX Spark | RTX PRO 6000, SM120 | GB10 / DGX Spark |
| shipped | **2026-09-08, on by default** | v2.5.0 | 2026-09-13 (after both) |
| vocabulary size | 65,536 | 65,536 | 32,768 |
| reported gain | **+20 % decode** | +9.7-16.8 % single-request decode | **+19.1 % code, +24.0 % prose** |
| output equivalence | identical outputs | — | unchanged distribution by construction |

The three numbers agree. That is the main value of this repo's version of the result: two
independent implementations on the same hardware, and a third on different hardware, land on
roughly the same ~20 %.

## The implementations converged on the same design

Flash-DGX's `src/patch_mtp_draft_vocab.py` and this repo's `patches/frspec_mtp.py` were written
independently and arrived at the same three decisions:

| decision | Flash-DGX | this repo |
|---|---|---|
| where to hook | the MTP class's `compute_logits` | the MTP class's `compute_logits` |
| how to slice | `w.index_select(0, ids).contiguous()` | `lm_head_weight.index_select(0, ids).contiguous()` |
| what to return | **full-width logits**, `new_full(..., -inf)` then `index_copy_(1, ids, red)` | **full-width logits**, `new_full(..., -inf)` then `index_copy_(1, _IDS, small)` |

The third one matters and is worth calling out, because it is not the obvious implementation.
The naive approach returns reduced-width `[T, K]` logits and maps the argmax back to token ids
afterwards — which requires patching whichever sampling path the drafter actually takes, and a
miss is **silent**: the drafter then proposes index *i* as if it were token *i*, output stays
correct because the target rejects the garbage, and only acceptance collapses. This repo
shipped that bug first (acceptance fell 3.23 -> 1.16 with no other symptom) and only then
arrived at the scatter-to-full-width design that Flash-DGX had already chosen. Scattering into
a `-inf` buffer costs ~0.5 MB of write per row against the 1.27 GB of weight reads it avoids,
and makes the reduction invisible to every downstream consumer.

## Where Flash-DGX is better

**Vocabulary construction.** Theirs (`tools/build_draft_vocab.py`) is a four-tier priority set:

1. corpus tokens ranked by frequency,
2. **all special and added tokens, unconditionally**,
3. **the first 256 byte-fallback ids**,
4. lowest remaining ids, using BPE merge order as a frequency proxy.

This repo's `scripts/token_freq.py` does only (1), plus a handful of explicitly named special
tokens. Tiers (2) and (3) are the robustness argument: chat templates, tool-call markers and
thinking delimiters must always be proposable, and byte-fallback ids are what the tokenizer
reaches for on text unlike the corpus. A corpus-frequency-only set can silently lose acceptance
on traffic that does not resemble the corpus.

**Consequence for the K=32,768 claim in this repo.** Because the two sets are built
differently, "32,768 is enough" is *not* a clean claim that half of their vocabulary would do.
What is measured here is that a 32,768-token, corpus-frequency-ranked set costs only 0.5-3.4 %
acceptance **on three workloads that resemble its corpus**. Testing K=32,768 with their
construction, on traffic unlike the corpus, is the experiment that would settle it, and it has
not been run.

## Where this repo adds something

- **The kernel-level decomposition.** Both projects report the end-to-end gain; neither
  publishes why it is that size. Here: the output head is 28.3 ms of a 106 ms step, visible as
  61 cuBLAS GEMV calls of ~8 ms in a 27-step trace, falling to one call at 9.1 ms. That is also
  what makes the head-to-body byte-ratio argument concrete — and predicts correctly that the
  same technique is worth only +1.4-3.1 % on a dense 27B model.
- **Mapped/unmapped counters** in the patch, logged periodically. Motivated by the silent bug
  above; Flash-DGX's design cannot hit that bug, but any reduced-width implementation can.
- **An optional fp8 draft head** (`VLLM_MTP_DRAFT_HEAD_DTYPE=fp8`), which would take the slice
  from 168 MB to 84 MB. **Implemented but never benchmarked** — treat it as untested.
- **The 2x2 ablation** showing the draft-vocabulary and fp8-side-layer savings add to within
  1.5 %.

## Pennyroyal's online fp8 is the biggest lever this study has not tested

Pennyroyal quantises "selected otherwise-BF16 projections, HyperConnection mix weights and the
output head" online, reporting +15.8-28.3 % decode. Flash-DGX's `MODE=hybrid` covers the GDN
in/out projections, sparse-attention q/k/v/o and shared experts — but **not** the
hyper-connection mix weights or the output head.

From this study's own byte budget, read once per verify pass:

| tensor group | bytes | covered by Flash-DGX hybrid? |
|---|---:|---|
| hyper-connection mix weights | 1.28 GB | no |
| output head | 1.27 GB | no |

That is ~2.5 GB per step, roughly **11 ms at the measured 241 GB/s**, sitting outside the
hybrid conversion. It is the clearest remaining target on this hardware, and Pennyroyal already
demonstrates it works. Extending the blockwise-fp8 conversion to those two groups is the next
experiment.

## Practical recommendation

For production use of this model on a DGX Spark, use
[Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX) directly. It ships these optimisations
properly, has more of them, has a stronger quality evidence base (a 17-scenario agentic
tournament rather than three synthetic workloads), and is maintained. This repo is a
measurement study whose patches exist to isolate effects, not to be deployed.
