# Method

How every number in this repo was produced, and the two confounds that make the protocol
mandatory rather than fussy.

## The harness

`scripts/bench_steps.py` drives the engine's OpenAI endpoint and reads the engine's **own
Prometheus counters** immediately before and after each request:

- `spec_decode_num_drafts_total` — one per decode step, so step counts are exact rather than
  inferred from timings
- `spec_decode_num_accepted_tokens_total` and the per-position breakdown — acceptance length
- `generation_tokens_total` — tokens actually produced
- `request_decode_time_seconds_sum` / `request_prefill_time_seconds_sum` — engine-side timing

Reporting per-step time as a counter delta rather than client wall time keeps HTTP, streaming
and tokenizer overhead out of the measurement, and makes acceptance directly observable — which
turned out to matter, because the only symptom of one bug was an acceptance change.

Decoding is **greedy** (`temperature=0`) throughout, so a given configuration produces exactly
the same tokens every run. Tokens-per-step is therefore not a sampled statistic: when it moves,
the model's behaviour genuinely changed.

Three workloads, chosen to span the acceptance range that speculative decoding lives or dies on:

| case | prompt | what it exercises |
|---|---|---|
| `prose` | open-ended creative writing | low acceptance (~2.1 tokens/step) |
| `code` | write a module to spec | middling acceptance (~3.2) |
| `repeat` | reproduce given code with one edit | near-ceiling acceptance (~3.9) |

## The protocol

1. Launch the configuration in an isolated container (`scripts/exp.sh`) with a cgroup memory
   cap, on its own port, leaving any production lane untouched.
2. Wait for health, then run the suite **three times**, watching the engine's
   `PLE mmap stats ... gather X ms/op` line.
3. Report the **third pass**, and only once the gather has stabilised below ~0.5 ms/op.
4. Record box state alongside the result: load average, desktop-session CPU, SM clock.
5. Measure the paired control **in the same session**. Never compare against a number from
   another hour.

## Why step 5 is not optional

Re-running the identical baseline configuration hours apart produced 87.7 and 106.2 ms/step —
a 21 % spread — with **bit-identical acceptance and per-position accept rates**. The model did
exactly the same work both times. Two effects account for it.

### Engram page-cache warmth: up to ~18 ms/step

The n-gram table is mmapped from NVMe and gathered on the decode critical path. Sampled every
30 s from a cold container start:

```
op 11.63 ms/op, gather 9.80 ms/op     <- first traffic
op  7.78 ms/op, gather 5.86 ms/op
op  3.30 ms/op, gather 1.55 ms/op
op  2.12 ms/op, gather 0.29 ms/op     <- steady state, after ~3 passes
```

Anything measured before this stabilises is measuring page-cache state.

### A logged-in desktop session: ~21 %

With a GNOME session compositing (shell at ~55 % CPU, X server and remote-desktop daemon
active), warm steady-state throughput is 30.4 tok/s on code; on an otherwise idle box the same
build measured 36.8. The desktop's *GPU* footprint is ~400 MB and a few percent utilisation —
what it actually costs is **memory bandwidth**, which is the resource decode is bound by.

### Consequence for profiling

A torch-profiler trace captured on a cold engram cache shows the GPU **kernels themselves**
~30 % slower than the same kernels warm, because NVMe reads filling the page cache contend for
the same LPDDR5X that feeds the GPU. Traces intended for kernel accounting must be warm. One
early conclusion in this work ("the optimisation removes GPU time but not wall time") was
purely an artefact of comparing a warm experiment against a cold-cache baseline.

## Profiling

The vLLM build here registers `/start_profile` only when a profiler config is passed on the
command line; the documented `VLLM_TORCH_PROFILER_DIR` environment variable alone leaves the
routes returning 404.

```
--profiler-config '{"profiler":"torch","torch_profiler_dir":"...","torch_profiler_with_stack":false}'
```

`scripts/compare_traces.py` normalises two traces by **decode-step count**, taken from the
rejection-sampler kernel (one call per speculative step), and reports per-step GPU time by
kernel family plus union-busy time. Union-busy is the honest denominator: summing kernel
durations double-counts anything overlapping across streams.

`scripts/gap_analysis.py` inverts the question — it finds the gaps *between* GPU kernels and
attributes each to the innermost CPU operation spanning it, which is how the engram gather was
identified as the dominant stall.

## Validating an A/B

A single A-then-B pair cannot separate the effect from drift on this machine. The stronger
form, in `aba_validate.sh`, is **A / B / A**: baseline, treatment, baseline again, all in one
unattended session. The two A measurements bound the drift; the effect is credible only if it
is large compared with `|A1 - A2|`.

## Machine characterisation microbenchmarks

Reported separately because they are properties of the hardware, not of the model:

- `scripts/skinny_bench2.py` — cuBLAS at the model's exact GEMM shapes, **L2 flushed between
  timings** (256 MB write) and timed by CUDA-graph replay, best of 5. Without the flush a
  1.27 GB weight matrix still partly hits a 25 MB L2 and the numbers flatter themselves.
- `scripts/zerocopy_gather_bench.py` — a hand-written CUDA gather kernel reading pinned host
  memory through its device pointer, against the same gather done device-side.
- `scripts/ple_gather_bench2.py` — engram row gather strategies. Version 1 of this benchmark
  built a fresh thread pool per call, which dominated the high-thread-count results and had to
  be discarded; the shipping engine reuses one pool, so a persistent pool is the correct
  comparison.
