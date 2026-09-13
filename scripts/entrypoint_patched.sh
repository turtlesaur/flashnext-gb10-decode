#!/usr/bin/env bash
# Entrypoint wrapper: injects research patches into the vLLM install inside the container,
# then execs the normal `vllm serve` entrypoint. The edits live in the container's writable
# layer and disappear with it, so the image stays pristine.
#
# Enabled by environment:
#   VLLM_MTP_DRAFT_VOCAB=/research/vocab/draft_vocab_top32768.npy   FR-Spec draft head
#   VLLM_MTP_DRAFT_HEAD_DTYPE=bf16|fp8
#   VLLM_MTP_DRAFT_PAD=2                                            pad draft batch to M rows
#   VLLM_PLE_LOG_DIR=/research/ple_log                              log engram row ids
#   VLLM_ROUTE_LOG_DIR=/research/route_log                          log MoE routing (eager only)
set -u
SP=/usr/local/lib/python3.12/dist-packages
export PYTHONPATH=/research/patches:${PYTHONPATH:-}

inject() {  # inject <file> <marker> <python-lines>
  local f=$1 marker=$2 body=$3
  grep -q "$marker" "$f" 2>/dev/null && { echo "[patch] $marker already present"; return; }
  printf '\n\n# --- %s ---\n%s\n' "$marker" "$body" >> "$f"
  python3 -c "import ast,sys; ast.parse(open('$f').read())" \
    && echo "[patch] injected $marker into $f" \
    || { echo "[patch] SYNTAX ERROR after injecting $marker into $f"; exit 1; }
}

if [ -n "${VLLM_MTP_DRAFT_VOCAB:-}" ]; then
  inject "$SP/vllm/models/qwen3_8_flash_next/nvidia/mtp.py" "research-frspec-model" \
"import os as _os
if _os.environ.get('VLLM_MTP_DRAFT_VOCAB'):
    from frspec_mtp import apply_model as _frspec_apply_model
    _frspec_apply_model(Qwen3_8FlashNextMTP)"
  # Patch the BASE proposer, which every drafter inherits from. Patching only the
  # Qwen3_8FlashNextMTPProposer subclass silently missed in the engine process
  # (mapped=0 unmapped=0 in the counters) and left the drafter emitting reduced-vocab
  # indices as if they were token ids: output stayed correct because the target rejects
  # them, but acceptance fell from 3.23 to 1.16.
  inject "$SP/vllm/v1/spec_decode/llm_base_proposer.py" "research-frspec-proposer-base" \
"import os as _os
if _os.environ.get('VLLM_MTP_DRAFT_VOCAB'):
    from frspec_mtp import apply_proposer as _frspec_apply_proposer
    _frspec_apply_proposer(SpecDecodeBaseProposer)"
fi

if [ "${VLLM_PLE_PREAD:-0}" = 1 ]; then
  inject "$SP/vllm_ple_mmap.py" "research-ple-pread" \
"import os as _os
if _os.environ.get('VLLM_PLE_PREAD') == '1':
    from ple_pread import apply as _ple_pread_apply
    _ple_pread_apply(MmapPleTable)"
fi

if [ -n "${VLLM_MTP_DRAFT_PAD:-}" ]; then
  inject "$SP/vllm/v1/spec_decode/qwen3_8_flash_next.py" "research-draftpad" \
"import os as _os
if _os.environ.get('VLLM_MTP_DRAFT_PAD'):
    from draftpad import apply as _draftpad_apply
    _draftpad_apply(Qwen3_8FlashNextMTPProposer)"
fi

if [ -n "${VLLM_PLE_LOG_DIR:-}" ]; then
  inject "$SP/vllm_ple_mmap.py" "research-ple-log" \
"import os as _os
if _os.environ.get('VLLM_PLE_LOG_DIR'):
    from ple_log_patch import apply as _ple_log_apply
    _ple_log_apply(_MmapNgramEmbedding)"
fi

if [ -n "${VLLM_ROUTE_LOG_DIR:-}" ]; then
  inject "$SP/vllm/model_executor/layers/fused_moe/layer.py" "research-route-log" \
"import os as _os
if _os.environ.get('VLLM_ROUTE_LOG_DIR'):
    from expert_route_log import apply as _route_log_apply
    _route_log_apply(FusedMoE)"
fi

exec vllm serve "$@"
