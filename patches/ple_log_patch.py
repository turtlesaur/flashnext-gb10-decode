"""Log PLE (engram) lookups: token ids in, table row ids out, per custom-op call.

Feeds H4.2 (validate a CPU reimplementation of the n-gram hash against the GPU ids),
H4.4 (LRU hot-cache sizing from the row-id stream) and D2 (speculative prefetch design).

Activate with VLLM_PLE_LOG_DIR=/research/ple_log. Wiring: append to the image's
vllm_ple_mmap.py (the module that owns the placeholder embedding and the custom op):
    import os as _os
    if _os.environ.get("VLLM_PLE_LOG_DIR"):
        from ple_log_patch import apply as _ple_log_apply
        _ple_log_apply(_MmapNgramEmbedding)
and the layer hook is installed lazily from inside the embedding forward.
"""
import logging
import os
import threading
import time

import numpy as np
import torch

logger = logging.getLogger("vllm.ple_log")
_DIR = os.environ.get("VLLM_PLE_LOG_DIR")
_FLUSH_EVERY = int(os.environ.get("VLLM_PLE_LOG_FLUSH", "200"))
_lock = threading.Lock()
_buf = {"ids": [], "n_tokens": [], "t": []}
_ctx = {"input_ids": None, "qsl": None, "ngram_context": None, "calls": 0}


def _flush():
    if not _buf["ids"]:
        return
    os.makedirs(_DIR, exist_ok=True)
    path = os.path.join(_DIR, f"ple-{int(time.time())}-{os.getpid()}.npz")
    lens = np.array([len(x) for x in _buf["ids"]], dtype=np.int64)
    np.savez_compressed(
        path,
        ids=np.concatenate(_buf["ids"]) if _buf["ids"] else np.zeros(0, np.int64),
        lens=lens,
        n_tokens=np.array(_buf["n_tokens"], dtype=np.int64),
        t=np.array(_buf["t"], dtype=np.float64),
        **({"input_ids": _ctx["input_ids"], "qsl": _ctx["qsl"], "ngram_context": _ctx["ngram_context"]}
           if _ctx["input_ids"] is not None else {}),
    )
    n = len(_buf["ids"])
    _buf["ids"].clear(); _buf["n_tokens"].clear(); _buf["t"].clear()
    logger.info("PLE log: flushed %d calls to %s", n, path)


def apply(embedding_cls) -> None:
    if getattr(embedding_cls, "_ple_log_patched", False):
        return
    orig_forward = embedding_cls.forward

    def forward(self, ids: torch.Tensor):
        out = orig_forward(self, ids)
        try:
            ids_np = ids.detach().to("cpu").numpy().reshape(-1).astype(np.int64)
            with _lock:
                _buf["ids"].append(ids_np)
                _buf["n_tokens"].append(int(ids.shape[0]) if ids.dim() > 1 else int(ids_np.size))
                _buf["t"].append(time.time())
                _ctx["calls"] += 1
                if _ctx["calls"] % _FLUSH_EVERY == 0:
                    _flush()
        except Exception as exc:  # never break serving because of logging
            logger.warning("PLE log failed: %s", exc)
        return out

    embedding_cls.forward = forward
    embedding_cls._ple_log_patched = True
    logger.info("PLE log: patched %s.forward -> %s", embedding_cls.__name__, _DIR)


def apply_layer(layer_cls) -> None:
    """Also record the token-side inputs of the last call (for hash validation)."""
    if getattr(layer_cls, "_ple_log_layer_patched", False):
        return
    orig = layer_cls._ple_mmap_orig_forward_impl

    def wrapped(self, hidden_states, input_ids, query_start_loc, ngram_context, *a, **k):
        try:
            _ctx["input_ids"] = input_ids.detach().to("cpu").numpy().reshape(-1)
            _ctx["qsl"] = query_start_loc.detach().to("cpu").numpy().reshape(-1)
            _ctx["ngram_context"] = ngram_context.detach().to("cpu").numpy()
        except Exception:
            pass
        return orig(self, hidden_states, input_ids, query_start_loc, ngram_context, *a, **k)

    layer_cls._ple_mmap_orig_forward_impl = wrapped
    layer_cls._ple_log_layer_patched = True
