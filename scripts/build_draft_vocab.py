#!/usr/bin/env python3
"""Build a frequency-ranked draft vocabulary for the MTP drafter.

This replaces the earlier corpus-frequency-only builder. It adopts the four-tier construction
used by blazux/qwen3.8-Flash-DGX's tools/build_draft_vocab.py, because a corpus-only set can
silently lose acceptance on traffic that does not resemble the corpus:

  1. every special and added token, unconditionally  - chat templates, tool-call markers and
     thinking delimiters must always be proposable, however rare they are in a corpus
  2. the first 256 ids (byte fallback)               - what the tokenizer reaches for on text
     unlike anything it has seen
  3. corpus tokens, ranked by frequency
  4. lowest remaining ids, as a BPE-merge-order frequency proxy, to fill to K

Output: a sorted int64 .npy of token ids, plus a coverage report per corpus.

    python build_draft_vocab.py --snapshot <path> --corpus <dir> [...] --n 65536 --out ids.npy
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

import numpy as np

TEXT_EXT = {".txt", ".md", ".py", ".json", ".jsonl", ".log", ".rst", ".c", ".h", ".cpp",
            ".js", ".ts", ".sh", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".html", ".css"}


def read_corpus(paths, limit_mb):
    files, total = [], 0
    for p in paths:
        if os.path.isfile(p):
            cand = [p]
        else:
            cand = [f for f in glob.glob(os.path.join(p, "**", "*"), recursive=True)
                    if os.path.isfile(f) and os.path.splitext(f)[1].lower() in TEXT_EXT]
        for f in sorted(cand):
            try:
                sz = os.path.getsize(f)
            except OSError:
                continue
            if sz < 200:
                continue
            files.append(f)
            total += sz
            if total > limit_mb * 1e6:
                return files, total
    return files, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True, help="model snapshot dir (for the tokenizer)")
    ap.add_argument("--corpus", nargs="*", default=[], help="files or directories")
    ap.add_argument("--n", type=int, default=65536)
    ap.add_argument("--limit-mb", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.snapshot)
    cfg = json.load(open(os.path.join(a.snapshot, "config.json")))
    vocab = cfg.get("text_config", cfg).get("vocab_size")
    print(f"tokenizer {len(tok)} entries, model vocab_size {vocab}, target K={a.n}")

    # tier 1: every special / added token
    special = set()
    for attr in ("all_special_ids",):
        special.update(int(i) for i in getattr(tok, attr, []) or [])
    try:
        special.update(int(i) for i in tok.get_added_vocab().values())
    except Exception:
        pass
    for extra in (cfg.get("bos_token_id"), cfg.get("eos_token_id"), cfg.get("pad_token_id"),
                  cfg.get("image_token_id"), cfg.get("video_token_id"),
                  cfg.get("vision_start_token_id"), cfg.get("vision_end_token_id")):
        if isinstance(extra, int):
            special.add(extra)
        elif isinstance(extra, list):
            special.update(int(i) for i in extra)
    special = {i for i in special if 0 <= i < vocab}
    print(f"tier 1: {len(special)} special/added tokens")

    # tier 2: byte fallback
    byte_fallback = set(range(min(256, vocab)))
    print(f"tier 2: {len(byte_fallback)} byte-fallback ids")

    # tier 3: corpus frequency
    counts = Counter()
    per_corpus = {}
    if a.corpus:
        files, total = read_corpus(a.corpus, a.limit_mb)
        print(f"tier 3: {len(files)} files, {total/1e6:.1f} MB")
        for f in files:
            try:
                txt = open(f, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            local = Counter()
            for i in range(0, len(txt), 200000):
                local.update(tok.encode(txt[i:i + 200000], add_special_tokens=False))
            counts.update(local)
            key = os.path.splitext(f)[1].lower() or "other"
            per_corpus.setdefault(key, Counter()).update(local)
        print(f"        {sum(counts.values())/1e6:.2f}M tokens, {len(counts)} distinct")

    chosen = list(special | byte_fallback)
    seen = set(chosen)
    for tid, _ in counts.most_common():
        if len(chosen) >= a.n:
            break
        if tid not in seen and 0 <= tid < vocab:
            chosen.append(tid); seen.add(tid)
    n_after_corpus = len(chosen)
    # tier 4: BPE merge order proxy
    tid = 0
    while len(chosen) < a.n and tid < vocab:
        if tid not in seen:
            chosen.append(tid); seen.add(tid)
        tid += 1
    print(f"tier 3 added {n_after_corpus - len(special | byte_fallback)}, "
          f"tier 4 filled {len(chosen) - n_after_corpus}")

    ids = np.unique(np.array(sorted(chosen), dtype=np.int64))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    np.save(a.out, ids)
    print(f"saved {a.out}: {ids.size} ids, max {ids.max()}")

    if counts:
        keep = set(ids.tolist())
        print("\ncoverage of this set, by corpus file type:")
        for ext, c in sorted(per_corpus.items(), key=lambda kv: -sum(kv[1].values())):
            tot = sum(c.values())
            hit = sum(v for k, v in c.items() if k in keep)
            print(f"  {ext:<8} {tot/1e6:7.2f}M tokens   coverage {hit/tot:.5f}")
        tot = sum(counts.values())
        hit = sum(v for k, v in counts.items() if k in keep)
        print(f"  {'ALL':<8} {tot/1e6:7.2f}M tokens   coverage {hit/tot:.5f}")


if __name__ == "__main__":
    sys.exit(main())
