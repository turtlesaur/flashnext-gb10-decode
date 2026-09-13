#!/usr/bin/env bash
# H2.2: which GPU architectures do the kernels actually loaded by the running engine target?
# Usage (host): bash ~/research/cubin_audit.sh <container-name>
CONT=${1:-exp-flashnext}
docker exec "$CONT" bash -c '
SP=/usr/local/lib/python3.12/dist-packages
PID=$(pgrep -f "VLLM::EngineCore" | head -1)
echo "=== engine pid $PID: mapped .so files from flashinfer / vllm / triton caches ==="
grep -oE "/[^ ]+\.so[^ ]*" /proc/$PID/maps 2>/dev/null | sort -u | grep -iE "flashinfer|jit_cache|\.cache/flashinfer|vllm/_C|_C_det|triton|cutlass|cute" > /tmp/mapped.txt
wc -l /tmp/mapped.txt
echo "=== arch summary per mapped module (cuobjdump --list-elf) ==="
while read -r so; do
  a=$(cuobjdump --list-elf "$so" 2>/dev/null | grep -oE "sm_[0-9]+a?" | sort | uniq -c | tr "\n" " ")
  [ -n "$a" ] && printf "%-110s %s\n" "$(basename "$so" | cut -c1-108)" "$a"
done < /tmp/mapped.txt
echo "=== runtime JIT cache contents (~/.cache/flashinfer) ==="
find ~/.cache/flashinfer -name "*.so" 2>/dev/null | while read -r so; do
  a=$(cuobjdump --list-elf "$so" 2>/dev/null | grep -oE "sm_[0-9]+a?" | sort -u | tr "\n" " ")
  printf "%-90s %s\n" "$(basename "$so" | cut -c1-88)" "$a"
done | head -40
echo "=== triton cache (per-arch dirs) ==="
ls ~/.triton/cache 2>/dev/null | head -3; find ~/.triton/cache -name "*.cubin" 2>/dev/null | head -3 | xargs -I{} sh -c "cuobjdump --list-elf {} 2>/dev/null | grep -oE \"sm_[0-9]+a?\" | sort -u | tr \"\n\" \" \"; echo"
'
