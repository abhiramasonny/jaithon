#!/usr/bin/env bash
# Export a YOLO model to ONNX, next to this script.
#
#   fetch_model.sh              yolov8n, the 12 MB default
#   fetch_model.sh yolov8s      a bigger one
#   fetch_model.sh yolov8x      biggest of the 640px family, ~260 MB
#   fetch_model.sh yolov8x6     exported at 1280 instead of 640
#
# Any name Ultralytics recognises works, including the yolo11 family and a
# local .pt of your own. The detector reads the input size and the class count
# out of the ONNX file rather than assuming them, so it does not need to be
# told which of these you picked -- see `open_model` in detector.jai.
#
# The weights are not committed: they are Ultralytics' to distribute, not this
# repository's. The export runs entirely locally in a throwaway environment, so
# nothing is added to your Python install.
#
# YOLOv8 is AGPL-3.0. Using it here to try the ONNX importer is fine; shipping
# a product on top of it is a licence question worth reading up on first.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model="${1:-yolov8n}"
model="${model%.pt}"
out="$here/${model##*/}.onnx"

# A name ending in 6 is a -p6 model, which is trained at 1280 and detects small
# things much better; exporting it at 640 would throw that away.
size=640
case "$model" in
    *6) size=1280 ;;
esac

if [[ -f "$out" ]]; then
    echo "already have $out"
    exit 0
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "this needs uv (https://docs.astral.sh/uv/) to build a throwaway environment" >&2
    exit 1
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cd "$work"
echo "exporting $model at ${size}x${size}"
MODEL="$model" IMGSZ="$size" uv run --with ultralytics --with onnx --python 3.11 python - <<'PY'
import os
from ultralytics import YOLO
YOLO(os.environ["MODEL"] + ".pt").export(
    format="onnx", imgsz=int(os.environ["IMGSZ"]), opset=13, dynamic=False
)
PY
mv "$work/${model##*/}.onnx" "$out"
echo "wrote $out"
echo "run it with: YOLO_MODEL=$out jaithon run examples/yolo_detect/detect.jai photo.jpg out.png"
