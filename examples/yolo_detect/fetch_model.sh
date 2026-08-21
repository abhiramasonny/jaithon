#!/usr/bin/env bash
# Export YOLOv8n to ONNX, next to this script.
#
# The weights are not committed: 12 MB of them, and they are Ultralytics'
# to distribute, not this repository's. The export runs entirely locally in a
# throwaway environment, so nothing is added to your Python install.
#
# YOLOv8 is AGPL-3.0. Using it here to try the ONNX importer is fine; shipping
# a product on top of it is a licence question worth reading up on first.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="$here/yolov8n.onnx"

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
uv run --with ultralytics --with onnx --python 3.11 python - <<'PY'
from ultralytics import YOLO
YOLO("yolov8n.pt").export(format="onnx", imgsz=640, opset=13, dynamic=False)
PY
mv "$work/yolov8n.onnx" "$out"
echo "wrote $out"
