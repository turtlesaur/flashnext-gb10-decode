#!/usr/bin/env python3
"""Token frequency / coverage analysis for a frequency-ranked draft vocabulary (FR-Spec style).

Corpus = code (vllm + transformers .py sources inside the image) + prose (Gutenberg texts in /research/corpus).
Outputs coverage of top-K tokens and saves ranked id lists to /research/vocab/.
"""
import os, glob, collections, json, sys, time
import numpy as np
SNAP = "/hf/hub/models--<org>--Qwen3.8-Flash-Next-NVFP4/snapshots/<revision>"
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(SNAP)
print("vocab size (tokenizer):", len(tok), "model vocab 248320")
def files(pattern, limit_mb):
    out, tot = [], 0
    for f in sorted(glob.glob(pattern, recursive=True)):
        try: s = os.path.getsize(f)
        except OSError: continue
        if s < 1000: continue
        out.append(f); tot += s
        if tot > limit_mb * 1e6: break
    return out
corpora = {
    "code": files("/usr/local/lib/python3.12/dist-packages/vllm/**/*.py", 25) + files("/usr/local/lib/python3.12/dist-packages/transformers/models/**/*.py", 15),
    "prose": files("/research/corpus/*.txt", 40),
    "docs": files("/usr/local/lib/python3.12/dist-packages/**/*.md", 10),
}
counts = {}
for name, fl in corpora.items():
    c = collections.Counter(); n = 0; t0 = time.time()
    for f in fl:
        try: txt = open(f, encoding="utf-8", errors="ignore").read()
        except Exception: continue
        for i in range(0, len(txt), 200000):
            ids = tok.encode(txt[i:i + 200000], add_special_tokens=False)
            c.update(ids); n += len(ids)
    counts[name] = c
    print(f"{name}: {len(fl)} files, {n/1e6:.2f}M tokens, {len(c)} distinct, {time.time()-t0:.0f}s")
allc = collections.Counter()
for c in counts.values(): allc.update(c)
os.makedirs("/research/vocab", exist_ok=True)
def coverage(c, ranked, K):
    top = set(ranked[:K]); tot = sum(c.values())
    return sum(v for k, v in c.items() if k in top) / tot
ranked_all = [k for k, _ in allc.most_common()]
print("\ncoverage of top-K (ranked on the union corpus), per corpus:")
print(f"{'K':>8}" + "".join(f"{n:>10}" for n in counts))
for K in [4096, 8192, 16384, 32768, 65536, 131072]:
    print(f"{K:>8}" + "".join(f"{coverage(counts[n], ranked_all, K):10.4f}" for n in counts))
print("\ncoverage of top-K ranked on the SAME corpus (upper bound):")
for K in [8192, 16384, 32768, 65536]:
    print(f"{K:>8}" + "".join(f"{coverage(counts[n], [k for k,_ in counts[n].most_common()], K):10.4f}" for n in counts))
for K in [16384, 32768, 65536]:
    ids = np.array(ranked_all[:K], dtype=np.int64)
    # always include special tokens (eos etc.) so the draft can propose them
    specials = [i for i in [tok.eos_token_id, tok.pad_token_id, 248044, 248046, 248045] if i is not None]
    ids = np.unique(np.concatenate([ids, np.array(specials, dtype=np.int64)]))
    np.save(f"/research/vocab/draft_vocab_top{K}.npy", ids)
    print(f"saved /research/vocab/draft_vocab_top{K}.npy ({len(ids)} ids)")
json.dump({"ranked_all": ranked_all[:200000]}, open("/research/vocab/ranked_all.json", "w"))
