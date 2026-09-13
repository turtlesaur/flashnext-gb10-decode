"""H1.4: log MoE routing decisions (top-k expert ids per token per layer) to measure expert-set
overlap between adjacent tokens, i.e. how much expert weight traffic a (k+1)-token verify pass
really costs.

Meant for an --enforce-eager instrumentation run (the .cpu() sync breaks graphs/compile).
Activate with VLLM_ROUTE_LOG_DIR=/research/route_log. Wiring: append to the image's
vllm/model_executor/layers/fused_moe/layer.py
    import os as _os
    if _os.environ.get("VLLM_ROUTE_LOG_DIR"):
        from expert_route_log import apply as _route_log_apply
        _route_log_apply(FusedMoE)
Analysis: route_overlap.py.
"""
import logging
import os
import time

import numpy as np
import torch

logger = logging.getLogger("vllm.route_log")
_DIR = os.environ.get("VLLM_ROUTE_LOG_DIR")
_buf = []  # (layer_name, topk_ids [T, k] int16)
_calls = [0]


def _flush():
    if not _buf:
        return
    os.makedirs(_DIR, exist_ok=True)
    path = os.path.join(_DIR, f"route-{int(time.time())}-{os.getpid()}.npz")
    names = np.array([n for n, _ in _buf])
    lens = np.array([x.shape[0] for _, x in _buf], dtype=np.int64)
    ids = np.concatenate([x for _, x in _buf]) if _buf else np.zeros((0, 1), np.int16)
    np.savez_compressed(path, names=names, lens=lens, ids=ids)
    _buf.clear()
    logger.info("route log: flushed to %s", path)


def apply(fused_moe_cls) -> None:
    if getattr(fused_moe_cls, "_route_log_patched", False):
        return
    orig = fused_moe_cls.select_experts
    is_static = isinstance(fused_moe_cls.__dict__.get("select_experts"), staticmethod)

    def wrapped(*args, **kwargs):
        out = orig(*args, **kwargs)
        try:
            topk_ids = None
            if isinstance(out, tuple):
                for t in out:
                    if isinstance(t, torch.Tensor) and t.dtype in (torch.int32, torch.int64) and t.dim() == 2:
                        topk_ids = t
            if topk_ids is not None:
                layer_name = kwargs.get("layer_name") or getattr(args[0], "layer_name", "?") if args else "?"
                _buf.append((str(layer_name), topk_ids.detach().to("cpu").numpy().astype(np.int16)))
                _calls[0] += 1
                if _calls[0] % 480 == 0:  # ~10 steps x 48 layers
                    _flush()
        except Exception as exc:
            logger.warning("route log failed: %s", exc)
        return out

    fused_moe_cls.select_experts = staticmethod(wrapped) if is_static else wrapped
    fused_moe_cls._route_log_patched = True
    logger.info("route log: patched FusedMoE.select_experts (static=%s) -> %s", is_static, _DIR)
