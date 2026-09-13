#!/usr/bin/env python3
"""Single-stream, step-level benchmark against a vLLM OpenAI endpoint.

Per-step time comes from vLLM's own metrics deltas (spec_decode_num_drafts_total ==
engine steps for a single stream), so the numbers are independent of client overhead.
"""
import json, time, urllib.request, argparse, re, statistics, os
ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:9114")
ap.add_argument("--model", default="qwen3.8-flash-next")
ap.add_argument("--max-tokens", type=int, default=300)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--cases", default="prose,code,repeat")
ap.add_argument("--label", default="")
ap.add_argument("--think", default="off")
ap.add_argument("--temperature", type=float, default=0.0)
a = ap.parse_args()

KEYS = ["spec_decode_num_drafts_total", "spec_decode_num_draft_tokens_total",
        "spec_decode_num_accepted_tokens_total", "generation_tokens_total", "prompt_tokens_total",
        "request_decode_time_seconds_sum", "request_prefill_time_seconds_sum"]
def metrics():
    raw = urllib.request.urlopen(a.base + "/metrics", timeout=10).read().decode()
    d = {}
    for k in KEYS:
        m = re.search(r"^vllm:%s\{[^}]*\} ([0-9.e+]+)$" % re.escape(k), raw, re.M)
        d[k] = float(m.group(1)) if m else 0.0
    for p in range(10):
        m = re.search(r'^vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*position="%d"[^}]*\} ([0-9.e+]+)$' % p, raw, re.M)
        if m: d["pos%d" % p] = float(m.group(1))
    return d

PROSE = "Write a detailed, vivid 600-word short story about a lighthouse keeper who discovers a message in a bottle. Use rich descriptive prose."
CODE = "Write a complete Python module implementing an LRU cache with TTL expiry, thread safety, and hit/miss statistics. Include docstrings and a small self-test at the bottom. Output only code."
UNIT = ("def process_record_{i}(payload, *, strict=True, retries=3):\n"
        "    if payload is None:\n        raise ValueError('payload_{i} must not be None')\n"
        "    result = {{'id': payload.get('id'), 'ts': payload.get('timestamp')}}\n"
        "    for key in ('temperature', 'humidity', 'pressure'):\n        raw = payload.get(key)\n"
        "        result[key] = float(raw) if raw is not None else None\n    return result\n\n")
REPEAT = "".join(UNIT.format(i=i) for i in range(10)) + "\n\nReproduce the code above EXACTLY, character for character, changing only retries=3 to retries=5. Output only the code."
FILLER = ("The quick brown fox jumps over the lazy dog near the riverbank at dawn. "
          "Sensors recorded ambient temperature, humidity and barometric pressure. ")
LONGCTX = (FILLER * 3000) + "\n\nWrite a detailed 300-word description of a storm at sea."  # ~40K tokens
CASES = {"prose": PROSE, "code": CODE, "repeat": REPEAT, "ctx40k": LONGCTX}

def run(prompt):
    body = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": a.max_tokens,
            "temperature": a.temperature, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": a.think == "on"}}
    req = urllib.request.Request(a.base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; n = 0; usage = None; text = ""
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"): continue
            payload = line[5:].strip()
            if payload == "[DONE]": break
            d = json.loads(payload)
            if d.get("usage"): usage = d["usage"]
            ch = d.get("choices") or []
            if ch:
                delta = ch[0].get("delta", {}) or {}
                piece = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning") or ""
                if piece:
                    if ttft is None: ttft = time.time() - t0
                    n += 1; text += piece
    return {"wall": time.time() - t0, "ttft": ttft or 0, "chunks": n, "usage": usage, "text": text}

rows = []
for case in a.cases.split(","):
    for rep in range(a.reps):
        m0 = metrics(); r = run(CASES[case]); m1 = metrics()
        dl = lambda k: m1.get(k, 0) - m0.get(k, 0)
        steps, drafted, acc, gen = dl("spec_decode_num_drafts_total"), dl("spec_decode_num_draft_tokens_total"), dl("spec_decode_num_accepted_tokens_total"), dl("generation_tokens_total")
        if steps == 0: steps = gen
        pos = [round(dl("pos%d" % p) / max(steps, 1), 3) for p in range(10) if "pos%d" % p in m1]
        dw = r["wall"] - r["ttft"]
        row = {"case": case, "rep": rep, "gen_tokens": gen, "steps": steps, "tok_per_step": gen / max(steps, 1),
               "accept_len": 1 + acc / max(steps, 1), "draft_acc_rate": acc / max(drafted, 1), "pos_acc": pos,
               "decode_wall_s": dw, "ms_per_step_wall": 1000 * dw / max(steps, 1),
               "ms_per_step_engine": 1000 * dl("request_decode_time_seconds_sum") / max(steps, 1),
               "tok_s_wall": gen / max(dw, 1e-9), "ttft_s": r["ttft"], "prefill_s_engine": dl("request_prefill_time_seconds_sum"),
               "prompt_tokens": (r["usage"] or {}).get("prompt_tokens"), "text_head": r["text"][:120], "text": r["text"]}
        rows.append(row)
        print(f"{case:<7} r{rep} gen={gen:4.0f} steps={steps:4.0f} tok/step={row['tok_per_step']:.2f} acc={row['accept_len']:.2f} pos={pos} "
              f"step={row['ms_per_step_wall']:.1f}ms(wall)/{row['ms_per_step_engine']:.1f}ms(eng) tok/s={row['tok_s_wall']:.1f} ttft={row['ttft_s']:.2f}s", flush=True)
print("\nSUMMARY (median over reps)")
for case in a.cases.split(","):
    rs = [r for r in rows if r["case"] == case]
    med = lambda k: statistics.median(r[k] for r in rs)
    print(f"  {case:<8} tok/s={med('tok_s_wall'):6.1f}  ms/step={med('ms_per_step_wall'):6.1f}  tok/step={med('tok_per_step'):.2f}  acc_len={med('accept_len'):.2f}")
if a.label:
    os.makedirs(os.path.expanduser("~/research/results"), exist_ok=True)
    p = os.path.expanduser(f"~/research/results/{a.label}-{time.strftime('%Y%m%d-%H%M%S')}.json")
    json.dump({"label": a.label, "args": vars(a), "rows": rows}, open(p, "w"), indent=1); print("saved", p)
