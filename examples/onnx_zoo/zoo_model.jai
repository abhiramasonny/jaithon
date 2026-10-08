#: What `zoo.jai` and `check.jai` share: finding a model `export.py` wrote,
#: feeding it the input it recorded, holding the answer against the one
#: onnxruntime gave for that input, and timing both of jaicv's paths.
#:
#: The two paths are the interpreter (`run_graph`, one operator at a time) and
#: the compiled plan (`compile_plan`, the whole model as one MPSGraph). A model
#: counts as running on a path when it imports, runs, and lands within a
#: relative 1e-5 of onnxruntime CPU -- the largest absolute difference over the
#: largest absolute reference value, over the first output -- or within twice
#: what torch's own CPU answer is from onnxruntime's, when that is larger.
import std.gpu as gpu
import std.io as io
from std.os import env, home_dir
from std.time import Instant
from jaitensor import Tensor
from jaicv.dnn.graph import Graph, run_graph
from jaicv.dnn.onnx import free_onnx_weights, read_onnx_model
from jaicv.dnn.plan import Plan, compile_plan

#: How far a path may land from onnxruntime and still count as correct.
pub let TOLERANCE = 1e-5

#: A model whose two CPU references already disagree by more than `TOLERANCE`
#: gets twice their disagreement instead. With measured BatchNorm statistics,
#: torch and onnxruntime are 2e-5 apart on ResNet-50 and 1.5e-4 on
#: FCN-ResNet-50, which no third float32 runtime can be held inside.
pub let REFERENCE_SLACK = 2.0

#: Where `export.py` writes: `$ONNX_ZOO_DIR`, or `~/.cache/jaithon/onnx_zoo`.
#: Outside the repository on purpose -- the exports are 2.8 GB.
pub fn zoo_dir() -> str {
    let named = env("ONNX_ZOO_DIR")
    if named is not null and named.len() > 0 { return named }
    return f"{home_dir()}/.cache/jaithon/onnx_zoo"
}

#: Every tag export.py managed to export and run through onnxruntime.
pub fn every_tag() -> list[str] {
    let listing = f"{zoo_dir()}/models.txt"
    if not io.Path(listing).exists() { return [] }
    var tags: list[str] = []
    for line in io.read_lines(listing) {
        if line.len() > 0 { tags.push(line) }
    }
    return tags
}

#: One tensor a `.meta` file names: `<kind> <name> <rank> <dims...> <path>`.
pub class Spec {
    pub let name: str
    pub let shape: list[int]
    pub let path: str

    pub fn init(self, name: str, shape: list[int], path: str) {
        self.name = name
        self.shape = shape
        self.path = path
    }

    pub fn count(self) -> int {
        var n = 1
        for d in self.shape { n *= d }
        return n
    }

    pub fn values(self) -> list[float] {
        return __prim__.bytes_f32_le(io.read_bytes(self.path), 0, self.count())
    }
}

fn parse_spec(line: str) -> Spec {
    let fields = line.split(" ")
    let rank = int(fields[2])
    var shape: list[int] = []
    for axis in 0..rank { shape.push(int(fields[3 + axis])) }
    return Spec(fields[1], shape, fields[3 + rank])
}

#: One exported model, as its `.meta` file describes it.
pub class Exported {
    pub let tag: str
    pub var model: str
    pub var inputs: list[Spec]
    pub var outputs: list[Spec]
    #: How far torch's CPU answer is from onnxruntime's, measured the way
    #: `relative_difference` measures; negative when export.py did not say.
    pub var agreement: float

    pub fn init(self, tag: str) {
        self.tag = tag
        self.model = ""
        self.inputs = []
        self.outputs = []
        self.agreement = -1.0
    }

    #: The export for `tag`, or null when export.py has not written it.
    pub static fn find(tag: str) -> Exported? {
        let meta = f"{zoo_dir()}/{tag}.meta"
        if not io.Path(meta).exists() { return null }
        let found = Exported(tag)
        for line in io.read_lines(meta) {
            if line.starts_with("model ") { found.model = line.split(" ")[1] }
            if line.starts_with("input ") { found.inputs.push(parse_spec(line)) }
            if line.starts_with("output ") { found.outputs.push(parse_spec(line)) }
            if line.starts_with("agreement ") { found.agreement = float(line.split(" ")[1]) }
        }
        return found
    }

    pub fn feeds(self) -> dict[str, Tensor] {
        var fed: dict[str, Tensor] = {}
        for spec in self.inputs { fed[spec.name] = Tensor(spec.shape, spec.values()) }
        return fed
    }
}

#: The largest absolute difference over the largest absolute reference value.
#: A shape mismatch or a NaN is infinitely wrong. The shape, not just the
#: count: a spurious width-one axis has the right number of values.
pub fn relative_difference(got: Tensor, want: list[float], shape: list[int]) -> float {
    #: A rank-zero answer is carried as `[1]`, as every scalar is here.
    let expected = if shape.len() == 0 { [1] } else { shape }
    if got.shape != expected { return 1.0e30 }
    let have = got.download()
    if have.len() != want.len() { return 1.0e30 }
    var worst = 0.0
    var scale = 0.0
    for index in 0..have.len() {
        let gap = abs(have[index] - want[index])
        if gap != gap { return 1.0e30 }
        if gap > worst { worst = gap }
        let size = abs(want[index])
        if size > scale { scale = size }
    }
    return worst / (if scale > 0.0 { scale } else { 1.0 })
}

fn ms_since(clock: Instant) -> float { return clock.elapsed().to_seconds() * 1000.0 }

fn free_all(outs: dict[str, Tensor]) -> void {
    for (_name, tensor) in outs.items() { tensor.free() }
}

#: The median of `samples`, with the spread beside it.
pub class Timing {
    pub let median: float
    pub let low: float
    pub let high: float
    pub let count: int

    pub fn init(self, samples: list[float]) {
        var sorted = [value for value in samples]
        sorted.sort()
        self.median = sorted[sorted.len() // 2]
        self.low = sorted[0]
        self.high = sorted[sorted.len() - 1]
        self.count = sorted.len()
    }

    fn __str__(self) -> str {
        return f"{self.median:.3f} ms (min {self.low:.3f}, max {self.high:.3f}, n={self.count})"
    }
}

#: Warm by the wall clock, not by a count -- the GPU's clock needs time busy
#: to come up, and a fixed number of calls of a fast model never gets it there
#: -- then `samples` samples of enough calls to fill about 150 ms each, with
#: one live result at a time. A call slower than 300 ms gets three samples of
#: one call, since a warm-up of those is already a measurement.
pub fn time_calls(call: fn() -> dict[str, Tensor], warm_s: float, samples: int) -> Timing {
    let start = Instant.now()
    var calls = 0
    var slow = false
    while start.elapsed().to_seconds() < warm_s or calls < 3 {
        let outs = call()
        gpu.synchronize()
        free_all(outs)
        calls += 1
        if ms_since(start) / float(calls) > 300.0 {
            slow = true
            break
        }
    }
    let per = ms_since(start) / float(calls)
    let reps = if per > 150.0 { 1 } else { max(1, int(150.0 / per)) }
    var times: list[float] = []
    for _sample in 0..(if slow { 3 } else { samples }) {
        let clock = Instant.now()
        for _rep in 0..reps {
            let outs = call()
            gpu.synchronize()
            free_all(outs)
        }
        times.push(ms_since(clock) / float(reps))
    }
    return Timing(times)
}

#: Everything one model's row says. A field that never got measured is
#: negative, and `refusal` is the first thing that stopped it, if anything did.
pub class Row {
    pub let tag: str
    pub var import_ms: float
    pub var nodes: int
    pub var interp_cold_ms: float
    pub var interp_rel: float
    pub var interp: Timing?
    pub var compile_ms: float
    pub var plan_rel: float
    pub var plan: Timing?
    pub var refusal: str
    #: `TOLERANCE`, or twice the references' own disagreement when larger.
    pub var tolerance: float

    pub fn init(self, tag: str) {
        self.tag = tag
        self.import_ms = -1.0
        self.nodes = 0
        self.interp_cold_ms = -1.0
        self.interp_rel = -1.0
        self.interp = null
        self.compile_ms = -1.0
        self.plan_rel = -1.0
        self.plan = null
        self.refusal = ""
        self.tolerance = TOLERANCE
    }

    pub fn interp_ok(self) -> bool {
        return self.interp_rel >= 0.0 and self.interp_rel <= self.tolerance
    }

    pub fn plan_ok(self) -> bool {
        return self.plan_rel >= 0.0 and self.plan_rel <= self.tolerance
    }

    pub fn both_ok(self) -> bool { return self.interp_ok() and self.plan_ok() }
}

#: Import `found`, run it through the interpreter and then the plan, and
#: compare each against onnxruntime. With `samples` above zero each path is
#: timed too, after a `warm_s`-second warm-up.
pub fn measure(found: Exported, warm_s: float, samples: int) -> Row {
    let row = Row(found.tag)
    row.tolerance = max(TOLERANCE, REFERENCE_SLACK * found.agreement)
    var clock = Instant.now()
    var graph: Graph? = null
    try {
        graph = read_onnx_model(found.model).graph
    } catch failure: Error {
        row.refusal = f"import: {failure.message}"
        return row
    }
    row.import_ms = ms_since(clock)
    let held = graph
    if held is null { return row }
    row.nodes = held.nodes.len()
    let feeds = found.feeds()
    let want = found.outputs[0].values()
    let out_name = held.outputs[0]

    clock = Instant.now()
    try {
        let outs = run_graph(held, feeds)
        gpu.synchronize()
        row.interp_cold_ms = ms_since(clock)
        row.interp_rel = relative_difference(outs[out_name], want, found.outputs[0].shape)
        free_all(outs)
    } catch failure: Error {
        row.refusal = f"interpreter: {failure.message}"
    }
    if row.interp_cold_ms >= 0.0 and samples > 0 {
        row.interp = time_calls(|| run_graph(held, feeds), warm_s, samples)
    }

    clock = Instant.now()
    var compiled: Plan? = null
    try {
        compiled = compile_plan(held, feeds)
    } catch failure: Error {
        if row.refusal.len() == 0 { row.refusal = f"plan: {failure.message}" }
    }
    row.compile_ms = ms_since(clock)
    let plan = compiled
    if plan is null {
        if row.refusal.len() == 0 {
            row.refusal = "plan: refused (JAICV_GRAPH_PLAN_REPORT=1 names the operator)"
        }
    } else {
        let outs = plan.run(feeds)
        gpu.synchronize()
        row.plan_rel = relative_difference(outs[out_name], want, found.outputs[0].shape)
        free_all(outs)
        if samples > 0 { row.plan = time_calls(|| plan.run(feeds), warm_s, samples) }
        plan.free()
    }
    free_all(feeds)
    free_onnx_weights(held)
    return row
}
