#!/bin/sh
# Time every exported model under the GPU lock: a batch of models per hold
# (BATCH, default 12, about two minutes), jaicv's batch and then the peers'
# batch, so neither side gets all of the hot device and no hold runs long.
#
#   examples/onnx_zoo/run_zoo.sh                 # every model export.py wrote
#   examples/onnx_zoo/run_zoo.sh resnet18 vit_tiny_dyn1
#
# Writes $ONNX_ZOO_DIR/zoo.tsv (jaicv) and $ONNX_ZOO_DIR/peer.json (peers), then
# prints the README table. PEERS=0 skips the peers; PEERS=missing times them
# only for models that have no peer row yet.
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
Z=${ONNX_ZOO_DIR:-$HOME/.cache/jaithon/onnx_zoo}
cd "$ROOT"
if [ $# -eq 0 ]; then set -- $(cat "$Z/models.txt"); fi
BATCH=${BATCH:-12}

run_batch() {
  echo "=== $* $(date +%H:%M:%S) load $(uptime | sed 's/.*averages: //')"
  ./scripts/bench/gpu_lock.sh ./jaithon run examples/onnx_zoo/zoo.jai -- --tsv "$Z/zoo.tsv" "$@" 2>&1 \
    | grep -v "Incompatible element type"
  peers=""
  for m in "$@"; do
    if [ "${PEERS:-1}" = "0" ]; then continue; fi
    if [ "${PEERS:-1}" = "missing" ] && [ -f "$Z/peer.json" ] && grep -q "\"$m\":" "$Z/peer.json"; then
      continue
    fi
    peers="$peers $m"
  done
  if [ -n "$peers" ]; then
    ./scripts/bench/gpu_lock.sh env VIRTUAL_ENV="$HOME/.venvs/scratch" uv run --active --offline --no-project \
      --with onnx --with onnxruntime --with onnxscript --with torchvision \
      python examples/onnx_zoo/peer_time.py $peers 2>&1 | grep -v -i warn || true
  fi
}

batch=""
count=0
for m in "$@"; do
  batch="$batch $m"
  count=$((count + 1))
  if [ $count -ge "$BATCH" ]; then
    run_batch $batch
    batch=""
    count=0
  fi
done
if [ -n "$batch" ]; then run_batch $batch; fi
uv run --offline --no-project python examples/onnx_zoo/peer_time.py --table "$Z/zoo.tsv"
