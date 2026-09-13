"""H2.3: pad the MTP draft batch from M=1 to M=2 rows on GB10.

cuBLAS bf16 GEMMs at the model's shapes run 15-25 % slower at M=1 than at M=2 on sm_121
(skinny_bench2). Every single-stream draft pass is M=1. Padding the draft batch to 2 rows
(a CUDA-graph capture size that already exists) makes the kernels take the faster path;
the padded row is garbage-in/garbage-out and never sampled.

Activate with VLLM_MTP_DRAFT_PAD=2. Wiring: append to the image's
v1/spec_decode/qwen3_8_flash_next.py
    import os as _os
    if _os.environ.get("VLLM_MTP_DRAFT_PAD"):
        from draftpad import apply as _draftpad_apply
        _draftpad_apply(Qwen3_8FlashNextMTPProposer)
"""
import logging
import os

logger = logging.getLogger("vllm.draftpad")
_PAD = int(os.environ.get("VLLM_MTP_DRAFT_PAD", "0") or 0)


def apply(cls) -> None:
    if _PAD <= 1 or getattr(cls, "_draftpad_patched", False):
        return
    orig = cls._determine_batch_execution_and_padding

    def _determine_batch_execution_and_padding(self, batch_size, *args, **kwargs):
        mode, input_batch_size, across_dp = orig(self, batch_size, *args, **kwargs)
        if input_batch_size < _PAD:
            input_batch_size = _PAD
        return mode, input_batch_size, across_dp

    cls._determine_batch_execution_and_padding = _determine_batch_execution_and_padding
    cls._draftpad_patched = True
    logger.info("draftpad: MTP draft batches padded to >= %d rows", _PAD)
