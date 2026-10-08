#!/bin/sh
# One batch of models, each in a process of its own, run back to back --
# run_zoo.sh wraps this in a single GPU-lock hold. A process apiece because a
# model timed after fifteen others in the same process reads up to twice as
# slow on the interpreter: what it measures then is the process, not the model.
Z=${ONNX_ZOO_DIR:-$HOME/.cache/jaithon/onnx_zoo}
for m in "$@"; do
  ./jaithon run examples/onnx_zoo/zoo.jai -- --tsv "$Z/zoo.tsv" "$m" 2>&1 | grep -v "Incompatible element type"
done
