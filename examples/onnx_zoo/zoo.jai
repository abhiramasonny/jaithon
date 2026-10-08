#: Run exported real architectures through jaicv.dnn and hold each against
#: onnxruntime: the interpreter, then the compiled plan.
#:
#: Export the models first (see README.md); then
#:
#:     jaithon run examples/onnx_zoo/zoo.jai                 # every exported model
#:     jaithon run examples/onnx_zoo/zoo.jai resnet18 vit_tiny_dyn1
#:     jaithon run examples/onnx_zoo/zoo.jai --check-only    # correctness, no timing
#:
#: `--warm S` and `--samples N` set the timing protocol (default 1.5 s of
#: warm-up by the wall clock, then 7 samples). Each row prints the first
#: refusal, if any, the relative difference from onnxruntime on each path, and
#: the import, interpreter, plan-compile and plan times. With `--tsv PATH` the
#: rows are also written as tab-separated values, which `peer_time.py --table`
#: joins with the peers' numbers.
#:
#: Time it under the GPU lock, one model per hold:
#:
#:     for m in $(cat ~/.cache/jaithon/onnx_zoo/models.txt); do
#:         ./scripts/bench/gpu_lock.sh ./jaithon run examples/onnx_zoo/zoo.jai --tsv zoo.tsv $m
#:     done
import std.io as io
from std.os import argv, env, set_env

from .zoo_model import Exported, Row, TOLERANCE, every_tag, measure, zoo_dir

fn number(value: float) -> str {
    if value < 0.0 { return "-" }
    return f"{value:.3f}"
}

fn rel(value: float) -> str {
    if value < 0.0 { return "-" }
    if value >= 1.0e29 { return "mismatch" }
    return f"{value:.1e}"
}

fn status(row: Row) -> str {
    if row.both_ok() { return "ok" }
    if row.interp_ok() { return "interp only" }
    return "refused"
}

fn tsv_line(row: Row) -> str {
    let interp = row.interp
    let plan = row.plan
    let fields = [row.tag, status(row), number(row.import_ms), number(row.interp_cold_ms), if interp is null {
        "-"
    } else {
        number(interp.median)
    }, if interp is null {
        "-"
    } else {
        number(interp.low)
    }, if interp is null {
        "-"
    } else {
        number(interp.high)
    }, number(row.compile_ms), if plan is null {
        "-"
    } else {
        number(plan.median)
    }, if plan is null {
        "-"
    } else {
        number(plan.low)
    }, if plan is null {
        "-"
    } else {
        number(plan.high)
    }, rel(row.interp_rel), rel(row.plan_rel), row.refusal.replace("\t", " ").replace("\n", " ")]
    return "\t".join(fields)
}

fn main() -> int {
    let args = argv()
    var warm_s = 1.5
    var samples = 7
    var tsv = ""
    var tags: list[str] = []
    var index = 1
    while index < args.len() {
        let arg = args[index]
        if arg == "--warm" {
            warm_s = float(args[index + 1])
            index += 2
        } elif arg == "--samples" {
            samples = int(args[index + 1])
            index += 2
        } elif arg == "--check-only" {
            samples = 0
            index += 1
        } elif arg == "--tsv" {
            tsv = args[index + 1]
            index += 2
        } else {
            tags.push(arg)
            index += 1
        }
    }
    if tags.len() == 0 or tags == ["all"] { tags = every_tag() }
    if tags.len() == 0 {
        print(f"no models in {zoo_dir()}; run examples/onnx_zoo/export.py first (see README.md)")
        return 1
    }
    #: The compiler names the operator that stopped it, which is the one fact
    #: needed to fix a model that falls back.
    if env("JAICV_GRAPH_PLAN_REPORT") is null { set_env("JAICV_GRAPH_PLAN_REPORT", "1") }

    var rows: list[Row] = []
    for tag in tags {
        let found = Exported.find(tag)
        if found is null {
            print(f"{tag}: not exported in {zoo_dir()}")
            continue
        }
        let row = measure(found, warm_s, samples)
        rows.push(row)
        let interp = row.interp
        let plan = row.plan
        print(f"== {tag}  [{status(row)}]  nodes={row.nodes}")
        if row.refusal.len() > 0 { print(f"   first refusal: {row.refusal}") }
        print(f"   rel vs onnxruntime: interp {rel(row.interp_rel)}  plan {rel(row.plan_rel)}")
        print(
            f"   import {number(row.import_ms)} ms  interp cold {number(row.interp_cold_ms)} ms  " +
                f"plan compile {number(row.compile_ms)} ms"
        )
        if interp is not null { print(f"   interp {interp}") }
        if plan is not null { print(f"   plan   {plan}") }
        if tsv.len() > 0 {
            var lines: list[str] = []
            if io.Path(tsv).exists() {
                for line in io.read_lines(tsv) {
                    if line.len() > 0 and not line.starts_with(f"{tag}\t") { lines.push(line) }
                }
            }
            lines.push(tsv_line(row))
            io.write_file(tsv, "\n".join(lines) + "\n")
        }
    }
    var both = 0
    for row in rows {
        if row.both_ok() { both += 1 }
    }
    print(f"{both} of {rows.len()} exports land within {TOLERANCE} of onnxruntime on both paths")
    return 0
}
