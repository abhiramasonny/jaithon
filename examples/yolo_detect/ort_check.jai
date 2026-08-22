#: jaicv's graph against onnxruntime, on the same fixed input.
#:
#: The recorded OpenCV cases check operators one at a time; this checks that a
#: real network, imported from a real file and scheduled by jaicv, lands where
#: the reference runtime lands. Run the peer beside it:
#:
#:     uv run --with onnxruntime --with numpy python examples/yolo_detect/ort_check.py
#:     jaithon run examples/yolo_detect/ort_check.jai
#:
#: The two should agree to float32: as of 2026-08-22 the sums are 6883170.492
#: and 6883170.500, a relative difference of 1.2e-9, and every sampled value
#: agrees to six digits. Needs the model, which `fetch_model.sh` gets and which
#: is deliberately not committed -- so this is a tool, not a gate.
import jaitensor as jt
from jaitensor import Tensor
from jaicv.dnn.plan import run_best
from std.os import env

from .detector import open_model

let SIDE = 640

fn main() -> int {
    let graph = open_model(env("YOLO_MODEL"))
    if graph is null { return 1 }
    var values: list[float] = []
    for index in 0..3 * SIDE * SIDE { values.push(float(index * 37 % 251) / 251.0) }
    let input = Tensor([1, 3, SIDE, SIDE], values)
    let outputs = run_best(graph, {graph.inputs[0]: input})
    let out = outputs[graph.outputs[0]]
    let flat = out.download()
    var total = 0.0
    var peak = 0.0
    for value in flat {
        total += value
        if abs(value) > peak { peak = abs(value) }
    }
    print(f"shape {out.shape}")
    print(f"sum {total:.6f}")
    print(f"absmax {peak:.6f}")
    for index in [0, 1, 1000, 8400, 100000, flat.len() - 1] {
        print(f"at {index}: {flat[index]:.6f}")
    }
    return 0
}
