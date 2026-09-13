# Patches

Runtime monkey-patches against one vLLM build, injected into a container's writable layer by
`scripts/entrypoint_patched.sh`. The engine is never forked and the image is never modified.

Each is written to be readable and to make an effect measurable, not to be merge-ready.

## Validated — these produced the results in RESULTS.md

| file | what it does | status |
|---|---|---|
| `frspec_mtp.py` | frequency-ranked draft vocabulary for the MTP head | **measured, +19-24 %**; output-preserving by construction |
| `ple_pread.py` | engram row gather via threaded `pread` instead of page faults | correct and 6.7x faster on cold rows in isolation, but the paired end-to-end A/B was **inconclusive** (~2 ms/step slower warm). Needs a hybrid policy before it is a clear win. |

## Instrumentation — written, not yet used for a published result

These exist to answer open questions in the study. They load and are syntactically checked, but
**no result in this repo depends on them**, and they have not been exercised end to end.

| file | intended use |
|---|---|
| `expert_route_log.py` | log MoE top-k routing per layer (eager mode) to measure expert-set overlap between consecutive tokens; analysed by `scripts/route_overlap.py` |
| `ple_log_patch.py` | log engram row ids per step, to size a hot-row cache (`scripts/lru_sim.py`) and to validate a CPU reimplementation of the n-gram hash |
| `draftpad.py` | pad the draft batch from M=1 to M=2 to dodge the cuBLAS GEMV penalty; subsumed by `frspec_mtp.py` for the output head, may still help the draft layer's own projections |
