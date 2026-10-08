#: The zoo's acceptance check: four exports that take seconds to write, each
#: of which once failed for a different reason, held against onnxruntime on
#: both of jaicv's paths.
#:
#: - squeezenet1_1: the smallest real CNN, and the control.
#: - mobilenet_v3_small: HardSwish, and depthwise convolutions, which ran a
#:   group at a time and took seconds.
#: - resnet18_dyn: torch's default dynamo export, weights in `.onnx.data`
#:   beside the model, which the reader used to refuse.
#: - vit_tiny_dyn1: the class token is a `Gather` by a rank-zero index, which
#:   used to keep a width-one axis and stop the classifier's `Gemm`.
#:
#:     VIRTUAL_ENV=~/.venvs/scratch uv run --active --offline --no-project \
#:         --with onnx --with onnxruntime --with onnxscript --with torchvision \
#:         python examples/onnx_zoo/export.py squeezenet1_1 mobilenet_v3_small resnet18 vit_tiny
#:     jaithon run examples/onnx_zoo/check.jai
#:
#: Exits non-zero on any miss, so it can sit in a gate.
from .zoo_model import Exported, TOLERANCE, measure, zoo_dir

let ACCEPTANCE = ["squeezenet1_1", "mobilenet_v3_small", "resnet18_dyn", "vit_tiny_dyn1"]

fn main() -> int {
    var failed = 0
    for tag in ACCEPTANCE {
        let found = Exported.find(tag)
        if found is null {
            print(f"MISSING {tag}: not exported in {zoo_dir()}; see this file's header")
            failed += 1
            continue
        }
        let row = measure(found, 0.0, 0)
        if row.both_ok() {
            print(f"ok      {tag}: interp {row.interp_rel:.1e}, plan {row.plan_rel:.1e}")
        } else {
            failed += 1
            let reason = if row.refusal.len() > 0 {
                row.refusal
            } else {
                "differs from onnxruntime"
            }
            print(f"FAIL    {tag}: interp {row.interp_rel:.1e}, plan {row.plan_rel:.1e}: {reason}")
        }
    }
    if failed > 0 {
        print(f"{failed} of {ACCEPTANCE.len()} missed (tolerance {TOLERANCE})")
        return 1
    }
    print(f"all {ACCEPTANCE.len()} within {TOLERANCE} of onnxruntime on both paths")
    return 0
}
