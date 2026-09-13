"""FR-Spec style frequency-ranked draft vocabulary for the Qwen3.8-Flash-Next MTP drafter.

The MTP drafter shares the target's `lm_head` (248,320 x 2560 bf16 = 1.27 GB). At batch 1 every
draft pass reads all of it: profiling shows 3 GEMV calls of ~8.2 ms per decode step, 501 ms of
the 553 ms spent in cuBLAS GEMV. Restricting the *draft* argmax to the K most frequent tokens
replaces that with a [K, 2560] slice. The target model still verifies over the full vocabulary,
so the output distribution is unchanged; only which tokens the drafter can propose changes.

Activate with VLLM_MTP_DRAFT_VOCAB=/path/to/ids.npy (int64 token ids).
Optional VLLM_MTP_DRAFT_HEAD_DTYPE=fp8 stores the slice as fp8 e4m3 with per-row scales.

Design note: the id table and the sliced weight are held in MODULE-LEVEL state, not as
attributes on the model. The proposer reaches the draft model through `self.model`, which is
not always the same object the patched `compute_logits` runs on (wrappers, compiled modules),
and an attribute lookup that silently misses would return unmapped indices - i.e. the drafter
proposing token id 5 when it means the 5th most frequent token. That failure is silent: output
stays correct because the target rejects the garbage, but acceptance collapses.
"""
import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger("vllm.frspec_mtp")

_IDS_PATH = os.environ.get("VLLM_MTP_DRAFT_VOCAB")
_MODE = os.environ.get("VLLM_MTP_DRAFT_HEAD_DTYPE", "bf16").lower()
_LOG_EVERY = int(os.environ.get("VLLM_MTP_DRAFT_LOG_EVERY", "300"))

# module-level state, shared by the model patch and the proposer patch
_IDS: torch.Tensor | None = None        # [K] int64, full-vocab ids, ascending
_W: torch.Tensor | None = None          # [K, H] bf16
_W8: torch.Tensor | None = None         # [K, H] fp8_e4m3
_SCALE_T: torch.Tensor | None = None    # [1, K] fp32
_FULL_V: int = 0                        # full vocabulary width of lm_head
_STATS = {"logits_calls": 0, "logits_ms": 0.0, "mapped": 0, "unmapped": 0, "built": False}


def _build(lm_head_weight: torch.Tensor) -> None:
    global _IDS, _W, _W8, _SCALE_T
    ids = torch.from_numpy(np.load(_IDS_PATH).astype(np.int64))
    ids = ids[ids < lm_head_weight.shape[0]].unique().to(lm_head_weight.device)
    sub = lm_head_weight.index_select(0, ids).contiguous()
    _IDS = ids
    if _MODE == "fp8":
        scale = sub.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-8) / 448.0
        _W8 = (sub.float() / scale).to(torch.float8_e4m3fn).contiguous()
        _SCALE_T = scale.t().contiguous()
        _W = None
        mb = _W8.numel() / 1e6
    else:
        _W = sub
        _W8 = None
        mb = sub.numel() * 2 / 1e6
    _STATS["built"] = True
    logger.warning(
        "FR-Spec: draft head built - %d of %d vocab rows, %.0f MB (%s), from %s",
        ids.numel(), lm_head_weight.shape[0], mb, _MODE, _IDS_PATH,
    )


def _draft_logits(model, hidden_states: torch.Tensor) -> torch.Tensor:
    """Reduced-vocabulary draft logits, returned at FULL vocabulary width.

    Returning [T, K] and mapping indices afterwards requires patching whichever sampling path
    the drafter happens to use, and a miss is silent (the drafter then proposes index i as if
    it were token id i). Scattering into a [T, V] buffer of -inf instead makes the reduction
    invisible to every downstream consumer: argmax, top-k and softmax all behave exactly as if
    the out-of-set tokens had been assigned probability zero. The buffer costs ~0.5 MB per row
    to write and read, against the 1.27 GB of weights this avoids touching.
    """
    global _FULL_V
    if not _STATS["built"]:
        _build(model.lm_head.weight)
        _FULL_V = model.lm_head.weight.shape[0]
    t0 = time.perf_counter()
    if _W is not None:
        small = F.linear(hidden_states, _W)
    else:
        xs = hidden_states.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-8) / 448.0
        x8 = (hidden_states.float() / xs).to(torch.float8_e4m3fn)
        small = torch._scaled_mm(x8, _W8.t(), scale_a=xs, scale_b=_SCALE_T, out_dtype=torch.bfloat16)
    out = hidden_states.new_full((small.shape[0], _FULL_V), float("-inf"), dtype=small.dtype)
    out.index_copy_(1, _IDS, small)
    _STATS["mapped"] += 1
    _STATS["logits_calls"] += 1
    _STATS["logits_ms"] += (time.perf_counter() - t0) * 1e3
    if _LOG_EVERY and _STATS["logits_calls"] % _LOG_EVERY == 0:
        logger.warning(
            "FR-Spec: %d draft-logit calls, %.3f ms/call (host-side), mapped=%d unmapped=%d",
            _STATS["logits_calls"], _STATS["logits_ms"] / _STATS["logits_calls"],
            _STATS["mapped"], _STATS["unmapped"],
        )
    return out


def _map(draft_ids: torch.Tensor) -> torch.Tensor:
    """Reduced-vocab indices -> full-vocab token ids."""
    if _IDS is None:
        _STATS["unmapped"] += 1
        if _STATS["unmapped"] in (1, 10, 100):
            logger.error("FR-Spec: draft ids could NOT be mapped (id table missing) - "
                         "acceptance will collapse. This is a bug, not a tuning issue.")
        return draft_ids
    _STATS["mapped"] += 1
    return _IDS[draft_ids]


def apply_model(cls) -> None:
    """compute_logits returns reduced-vocab logits [T, K]."""
    if getattr(cls, "_frspec_model_patched", False):
        return

    def compute_logits(self, hidden_states, spec_step_idx: int = 0):
        return _draft_logits(self, hidden_states)

    cls._frspec_orig_compute_logits = cls.compute_logits
    cls.compute_logits = compute_logits
    cls._frspec_model_patched = True
    logger.warning("FR-Spec: patched %s.compute_logits", cls.__name__)


def apply_proposer(cls) -> None:
    """No longer required.

    `_draft_logits` returns full-width logits, so every sampling path already produces correct
    full-vocabulary token ids. Kept so the entrypoint injection stays valid, and because the
    counters it used exposed the original silent-mapping bug.
    """
    logger.warning("FR-Spec: proposer patch not needed (logits are returned at full width)")
