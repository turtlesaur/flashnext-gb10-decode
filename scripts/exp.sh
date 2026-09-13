#!/usr/bin/env bash
# Experiment launcher for the Flash-Next ablit lane (research copy on :9115).
# Usage: exp.sh <label> [extra vllm args...]
# Env knobs: MTP=3 CG=PIECEWISE MOE=auto CPUSET= AUTOTUNE=0 LOAD=safetensors PREWARM=0 KV=20g SEQS=12
#            MEMLIMIT=100g (host-side cgroup cap; GPU pool allocations are not charged to it)
#            ENV_EXTRA="-e X=1 -v /a:/b"  extra docker args   WAIT=1
# NOTE: never use LOAD=fastsafetensors on this checkpoint: it pulled the 51 GB PLE shards into RAM
#       and thrashed the whole box on 2026-09-12 (MemAvailable 0, load >200, SSH dead for 40+ min).
set -uo pipefail
LABEL=$1; shift
NAME=${NAME:-exp-flashnext}; PORT=${PORT:-9115}
IMAGE=${IMAGE:-qwen38-flash-dgx:upstream-nvfp4}
SNAP=/hf/hub/models--<org>--Qwen3.8-Flash-Next-NVFP4/snapshots/<revision>
# HYBRID=1 serves the -fp8hybrid snapshot: NVFP4 experts as published, plus the dense side
# layers (GDN in/out projections, QSA q/k/v/o, shared experts) rewritten as blockwise fp8-e4m3.
# Build it once with scripts/prepare-hybrid.sh. VLLM_USE_DEEP_GEMM=0 matches upstream.
HYB=()
if [ "${HYBRID:-0}" = 1 ]; then
  SNAP="${SNAP}-fp8hybrid"
  HYB=(-e VLLM_FP8_HYBRID=1 -e VLLM_USE_DEEP_GEMM=0)
fi
SPLIT='["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::qwen3_8_flash_next_ple_short_conv","vllm::qwen3_8_flash_next_qsa_with_output","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::qwen_gdn_attention_core_fused_norm_packed","vllm::sparse_attn_indexer","vllm::ple_mmap_lookup"]'
MTP=${MTP:-3}; CG=${CG:-PIECEWISE}; MOE=${MOE:-auto}; CPUSET=${CPUSET:-}; LOAD=${LOAD:-safetensors}
if [ "$LOAD" = fastsafetensors ]; then echo "!! refusing LOAD=fastsafetensors (box thrash, see header)"; exit 1; fi
CGARGS=(-cc.cudagraph_mode=$CG)
case "$CG" in PIECEWISE|FULL_AND_PIECEWISE) CGARGS+=(-cc.splitting_ops=$SPLIT);; esac
SPEC=(); [ "$MTP" != 0 ] && SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP}")
MOEARGS=(); [ "$MOE" != auto ] && MOEARGS=(--moe-backend "$MOE")
AT=--no-enable-flashinfer-autotune; [ "${AUTOTUNE:-0}" = 1 ] && AT=--enable-flashinfer-autotune
# Torch profiler: this build registers /start_profile and /stop_profile only when a
# ProfilerConfig with profiler != None is passed on the command line. The env var
# VLLM_TORCH_PROFILER_DIR alone is NOT enough (routes stay 404).
PROF=()
[ "${PROFILER:-0}" = 1 ] && PROF=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"/research/profiles/$LABEL\",\"torch_profiler_with_stack\":false}")
CPU=(); [ -n "$CPUSET" ] && CPU=(--cpuset-cpus "$CPUSET")
MEM=${MEMLIMIT:-100g}
mkdir -p ~/research/profiles/$LABEL ~/research/logs ~/research/patches ~/llm-stack/vllm-cache/ablit
docker rm -f $NAME >/dev/null 2>&1 || true
echo ">> launching $NAME label=$LABEL MTP=$MTP CG=$CG MOE=$MOE CPUSET='$CPUSET' LOAD=$LOAD AUTOTUNE=${AUTOTUNE:-0} MEM=$MEM extra: $*"
# PATCHED=1 routes through /research/entrypoint_patched.sh, which injects the research
# patches (FR-Spec draft head, draft-batch padding, loggers) into the container's writable
# layer before exec'ing `vllm serve`. The image itself is never modified.
EP=()
[ "${PATCHED:-0}" = 1 ] && EP=(--entrypoint /research/entrypoint_patched.sh)
docker run -d --name $NAME --gpus all --ipc=host --shm-size 16g -p $PORT:8000 "${CPU[@]}" \
  --memory $MEM --memory-swap $MEM "${EP[@]}" \
  -v ~/.cache/huggingface:/hf -v ~/research:/research -v ~/llm-stack/vllm-cache/ablit:/root/.cache/vllm \
  -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e VLLM_PLE_MMAP=1 -e VLLM_PLE_MMAP_WORKERS=32 -e VLLM_PLE_MMAP_PREWARM=${PREWARM:-0} \
  -e VLLM_QSA_EXACT_TOPK=0 -e VLLM_QSA_DET_TOPK=1 -e VLLM_QSA_DET_LIB=/opt/llm/kernel-det/_C_det.so -e VLLM_FP8_PAD_M4=0 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=0 -e VLLM_TORCH_PROFILER_DIR=/research/profiles/$LABEL \
  -e PYTHONPATH=/research/patches \
  ${ENV_EXTRA:-} "${HYB[@]}" \
  $IMAGE $SNAP --served-model-name qwen3.8-flash-next --host 0.0.0.0 --port 8000 --load-format $LOAD \
  --max-model-len ${CTX:-262144} --max-num-seqs ${SEQS:-12} --gpu-memory-utilization 0.01 --kv-cache-memory-bytes ${KV:-20g} \
  --enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 8192 "${CGARGS[@]}" $AT --kv-cache-dtype auto \
  --chat-template /hf/chat-templates/flashnext.jinja --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  "${SPEC[@]}" "${MOEARGS[@]}" "${PROF[@]}" "$@" || { echo "docker run failed"; exit 1; }
[ "${WAIT:-1}" = 1 ] || exit 0
T0=$(date +%s)
for i in $(seq 1 240); do
  sleep 10
  if curl -sf -m 3 localhost:$PORT/health >/dev/null 2>&1; then echo ">> healthy after $(( $(date +%s) - T0 ))s"; break; fi
  ST=$(docker inspect -f '{{.State.Status}}' $NAME 2>/dev/null || echo missing)
  if [ "$ST" != running ]; then echo "!! container $ST"; docker logs --tail 60 $NAME 2>&1 | grep -vE "min_frames|max_frames" | tail -40; exit 1; fi
done
docker logs $NAME > ~/research/logs/boot-$LABEL.log 2>&1
grep -E "MoE backend|GDN|kernel|Capturing|capturing finished|Graph capturing|took|WARNING|ERROR|prefill kernel|autotune|Using cache|skinny|async|FR-Spec" ~/research/logs/boot-$LABEL.log | grep -vE "min_frames|max_frames|Unknown vLLM env|Loading safetensors" | cut -c1-200 | head -40
