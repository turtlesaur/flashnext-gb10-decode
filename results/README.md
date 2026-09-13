# Raw data

`raw_runs.csv` — every individual benchmark repetition behind the tables in RESULTS.md:

| column | meaning |
|---|---|
| `run` | configuration label; the trailing `-pN` is the warm-up pass number (report pass 3) |
| `case` | workload: `prose`, `code`, `repeat` (verbatim copy) |
| `rep` | repetition within that pass |
| `tok_s` | generated tokens / decode wall time |
| `ms_per_step` | decode wall time / engine decode steps |
| `tok_per_step` | accepted tokens per speculative step |
| `accept_len` | mean acceptance length from the engine's counters |

Step counts and acceptance come from the engine's own Prometheus counters, not from client
timing. Decoding is greedy, so `tok_per_step` is exact rather than sampled: within one
configuration it repeats identically across runs, and any change in it is a real behaviour
change rather than noise.

Runs labelled `e1`-`e13` are the exploratory sequence in FINDINGS.md, in order. Only runs
measured back-to-back in one session with a warm engram cache are comparable to each other —
see METHOD.md, which explains why that matters by about 20 % on this machine.
