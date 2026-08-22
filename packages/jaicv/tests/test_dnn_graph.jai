#: The graph IR, its topological order, and every operator the executor runs.
#:
#: The operator cases are onnxruntime's own answers rather than a reading of
#: the ONNX specification: `tools/dnn_graph_oracle.py ops` builds a one-node
#: model for each case, runs it, and writes what came back to
#: `tests/oracle/dnn_graph_cases.txt`. The network at the bottom is torch's
#: answer instead, in `tests/oracle/dnn_graph_network.txt`, because torch is
#: what someone porting a model has in front of them. Both files are checked
#: in, so this runs with neither package installed; the script's docstring has
#: the two commands that rebuild them.
#:
#: Tolerances are relative above one and absolute below it, and are set per
#: group rather than globally -- a convolution accumulates a hundred products
#: on the GPU and will never match a CPU sum to the last bit, while `Relu`
#: should.
import std.gpu as gpu
from std.io import Path
from std.test import assert_contains, assert_eq, assert_near, assert_throws, assert_true
from jaicv.dnn.graph import Attr, Graph, GraphNode, is_implemented, run_graph
from jaicv.dnn.plan import Plan, compile_plan
from jaitensor import Tensor

#: One tensor out of the oracle file: `<tag> <rank> <dims...> <values...>`.
class Recorded {
    pub let shape: list[int]
    pub let values: list[float]

    pub fn init(self, shape: list[int], values: list[float]) {
        self.shape = shape
        self.values = values
    }

    pub fn upload(self) -> Tensor { return Tensor(self.shape, self.values) }
}

#: One operator, its inputs, and what the reference produced. `feeds` holds
#: `null` where ONNX leaves an optional input out, because position is what
#: identifies an argument.
class Case {
    pub let op: str
    pub let name: str
    pub var feeds: list[Recorded?]
    pub var attrs: dict[str, Attr]
    pub var wants: list[Recorded]

    pub fn init(self, op: str, name: str) {
        self.op = op
        self.name = name
        self.feeds = []
        self.attrs = {}
        self.wants = []
    }
}

fn read_tensor(fields: list[str], start: int) -> Recorded {
    let rank = int(fields[start])
    var shape: list[int] = []
    for index in 0..rank { shape.push(int(fields[start + 1 + index])) }
    var values: list[float] = []
    for index in start + 1 + rank..fields.len() {
        if fields[index].len() == 0 { continue }
        values.push(float(fields[index]))
    }
    return Recorded(shape, values)
}

fn read_attr(item: Case, fields: list[str]) -> void {
    let key = fields[1]
    let kind = fields[2]
    if kind == "int" {
        item.attrs[key] = Attr.of_int(int(fields[3]))
        return
    }
    if kind == "float" {
        item.attrs[key] = Attr.of_float(float(fields[3]))
        return
    }
    if kind == "str" {
        item.attrs[key] = Attr.of_str(fields[3])
        return
    }
    if kind == "ints" {
        var values: list[int] = []
        for index in 3..fields.len() { values.push(int(fields[index])) }
        item.attrs[key] = Attr.of_ints(values)
        return
    }
    if kind == "floats" {
        var values: list[float] = []
        for index in 3..fields.len() { values.push(float(fields[index])) }
        item.attrs[key] = Attr.of_floats(values)
        return
    }
    if kind == "tensor" {
        item.attrs[key] = Attr.of_tensor(read_tensor(fields, 3).upload())
        return
    }
    throw ValueError(f"the oracle file has an attribute of kind {kind}")
}

fn load_cases(path: Path) -> dict[str, list[Case]] {
    if not path.exists() { throw ValueError(f"the oracle file {path} is missing") }
    var groups: dict[str, list[Case]] = {}
    var group = "ungrouped"
    var current: Case? = null
    for line in path.read_text().split("\n") {
        if line.len() == 0 or line[0] == "#" { continue }
        let fields = line.split(" ")
        if fields[0] == "group" {
            group = fields[1]
            continue
        }
        if fields[0] == "case" {
            let started = Case(fields[1], fields[2])
            current = started
            if groups.get(group, null) is null { groups[group] = [] }
            groups[group].push(started)
            continue
        }
        let item = current
        if item is null { throw ValueError("the oracle file has a line before its first case") }
        if fields[0] == "attr" {
            read_attr(item, fields)
        } elif fields[0] == "in" {
            item.feeds.push(if fields[1] == "null" { null } else { read_tensor(fields, 1) })
        } elif fields[0] == "out" {
            item.wants.push(read_tensor(fields, 1))
        } else {
            throw ValueError(f"the oracle file has a line starting {fields[0]}")
        }
    }
    return groups
}

var _cases: dict[str, list[Case]] = {}

fn cases_in(group: str) -> list[Case] {
    if _cases.len() == 0 {
        _cases = load_cases(Path("packages/jaicv/tests/oracle/dnn_graph_cases.txt"))
    }
    let found = _cases.get(group, null)
    if found is null { throw ValueError(f"the oracle file has no {group} group") }
    return found
}

#: Run one case as a one-node graph and return the worst difference from the
#: reference, scaled so a value of ten gets ten times the slack of a value of
#: one and anything under one is judged absolutely.
fn worst_difference(item: Case) -> float {
    var names: list[str] = []
    var declared: list[str] = []
    var feeds: dict[str, Tensor] = {}
    for (index, feed) in item.feeds.enumerate() {
        if feed is null {
            #: An omitted optional input keeps its position and gets no name.
            names.push("")
            continue
        }
        let name = f"in{index}"
        names.push(name)
        declared.push(name)
        feeds[name] = feed.upload()
    }
    var outs: list[str] = []
    for slot in 0..item.wants.len() { outs.push(f"out{slot}") }
    let graph = Graph(
        nodes: [
            GraphNode(
                item.op,
                name: item.name,
                inputs: names,
                outputs: outs,
                attrs: item.attrs
            ),
        ],
        inputs: declared,
        outputs: outs
    )
    let produced = run_graph(graph, feeds)
    var worst = 0.0
    for (slot, want) in item.wants.enumerate() {
        let got = produced[outs[slot]]
        assert_eq(got.shape, want.shape, f"{item.name} output {slot} has the wrong shape")
        let values = got.download()
        for index in 0..values.len() {
            let expected = want.values[index]
            let gap = abs(values[index] - expected) / max(1.0, abs(expected))
            if gap > worst { worst = gap }
        }
    }
    for (_name, tensor) in produced.items() { tensor.free() }
    for (_name, tensor) in feeds.items() { tensor.free() }
    return worst
}

#: The same case, run through the compiled graph rather than the interpreter.
#:
#: A negative number means the compiler declined it, which is allowed -- the
#: interpreter answers then, and the whole point of the fallback is that a
#: graph it cannot express still runs. What is not allowed is compiling it and
#: getting a different answer, which is what this catches.
fn compiled_difference(item: Case, weights_are_constant: bool) -> float {
    var names: list[str] = []
    var declared: list[str] = []
    var feeds: dict[str, Tensor] = {}
    var fixed: dict[str, Tensor] = {}
    for (index, feed) in item.feeds.enumerate() {
        if feed is null {
            names.push("")
            continue
        }
        let name = f"in{index}"
        names.push(name)
        #: With `weights_are_constant`, everything after the first input is an
        #: initializer rather than something fed -- which is what a real model
        #: looks like, and what lets the compiler fold and reorder weights.
        if weights_are_constant and index > 0 {
            fixed[name] = feed.upload()
        } else {
            declared.push(name)
            feeds[name] = feed.upload()
        }
    }
    var outs: list[str] = []
    for slot in 0..item.wants.len() { outs.push(f"out{slot}") }
    let graph = Graph(
        nodes: [
            GraphNode(
                item.op,
                name: item.name,
                inputs: names,
                outputs: outs,
                attrs: item.attrs
            ),
        ],
        inputs: declared,
        outputs: outs,
        initializers: fixed
    )
    var plan: Plan? = null
    try {
        plan = compile_plan(graph, feeds)
    } catch _failure: Error {
        plan = null
    }
    let built = plan
    if built is null {
        for (_name, tensor) in feeds.items() { tensor.free() }
        for (_name, tensor) in fixed.items() { tensor.free() }
        return -1.0
    }
    let produced = built.run(feeds)
    var worst = 0.0
    for (slot, want) in item.wants.enumerate() {
        let got = produced[outs[slot]]
        assert_eq(got.shape, want.shape, f"{item.name} compiled to the wrong shape")
        let values = got.download()
        for index in 0..values.len() {
            let expected = want.values[index]
            let gap = abs(values[index] - expected) / max(1.0, abs(expected))
            if gap > worst { worst = gap }
        }
    }
    for (_name, tensor) in produced.items() { tensor.free() }
    for (_name, tensor) in feeds.items() { tensor.free() }
    for (_name, tensor) in fixed.items() { tensor.free() }
    built.free()
    return worst
}

#: Every case the compiler accepts has to give the reference answer.
fn check_compiled(group: str, tolerance: float) -> int {
    var compiled = 0
    #: Both ways round: with the operands fed, and with all but the first
    #: baked in as weights. The second is what a real model looks like and is
    #: the only shape in which the compiler will fold or reorder anything.
    for constant in [false, true] {
        for item in cases_in(group) {
            let gap = compiled_difference(item, constant)
            if gap < 0.0 { continue }
            compiled += 1
            assert_true(
                gap <= tolerance,
                f"{item.op} case {item.name}, compiled, is off the reference by {gap}"
            )
        }
    }
    return compiled
}

#: The compiled graph against the same references the interpreter is held to.
#:
#: Counted as well as checked: if a change quietly stopped the compiler taking
#: anything at all, every case would be skipped and the test would pass while
#: testing nothing.
fn test_the_compiled_graph_matches_onnxruntime() -> void {
    var compiled = 0
    for (group, tolerance) in [
        ("elementwise", 1e-6),
        ("broadcast", 1e-6),
        ("shape", 1e-6),
        ("window", 5e-6),
        ("linear", 1e-6),
        ("norm", 1e-5),
        ("reduce", 1e-6),
        ("resample", 3e-6),
        ("recurrent", 3e-5),
    ] {
        compiled += check_compiled(group, tolerance)
    }
    assert_true(compiled >= 40, f"only {compiled} cases compiled, which is too few to mean much")
}

fn check_group(group: str, tolerance: float) -> void {
    let cases = cases_in(group)
    assert_true(cases.len() > 0, f"the {group} group is empty")
    for item in cases {
        let gap = worst_difference(item)
        assert_true(
            gap <= tolerance,
            f"{item.op} case {item.name} is off the reference by {gap}, over {tolerance}"
        )
    }
}

fn test_elementwise_operators_match_onnxruntime() -> void { check_group("elementwise", 1e-6) }

fn test_broadcasting_matches_onnxruntime() -> void { check_group("broadcast", 1e-6) }

fn test_shape_operators_match_onnxruntime() -> void { check_group("shape", 1e-7) }

fn test_window_operators_match_onnxruntime() -> void { check_group("window", 5e-6) }

fn test_matrix_operators_match_onnxruntime() -> void { check_group("linear", 1e-6) }

fn test_normalization_operators_match_onnxruntime() -> void { check_group("norm", 1e-6) }

#: An LSTM runs its gates a timestep at a time, so the last step carries every
#: rounding difference of the ones before it -- hence a looser bound than the
#: single-shot operators get.
fn test_recurrent_operators_match_onnxruntime() -> void { check_group("recurrent", 2e-5) }

fn test_reductions_match_onnxruntime() -> void { check_group("reduce", 1e-6) }

fn test_resampling_matches_onnxruntime() -> void { check_group("resample", 3e-6) }

#: Every two-sided elementwise operator is exercised with operands of
#: different shapes, not only matching ones. Named here so that adding an
#: operator without a broadcasting case fails rather than passes quietly.
fn test_every_elementwise_operator_is_broadcast() -> void {
    let wanted = [
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Pow",
        "Min",
        "Max",
        "Sum",
        "Mean",
        "Equal",
        "Greater",
        "Less",
        "And",
        "Or",
        "Where",
    ]
    var covered: list[str] = []
    for item in cases_in("broadcast") {
        var stretched = false
        var first: list[int] = []
        for feed in item.feeds {
            if feed is null { continue }
            if first.len() == 0 {
                first = feed.shape
            } elif feed.shape != first {
                stretched = true
            }
        }
        if stretched and not item.op in covered { covered.push(item.op) }
    }
    for op in wanted {
        assert_contains(covered, op, "this operator has no case with operands of unequal shape")
    }
    #: `PRelu` broadcasts too, and sits with the unary cases because only one
    #: of its two inputs is ever a whole tensor.
    var prelu = false
    for item in cases_in("elementwise") {
        if item.op == "PRelu" { prelu = true }
    }
    assert_true(prelu, "PRelu has no case with a per-channel slope")
}

fn chain_graph() -> Graph {
    #: Written back to front, so an executor trusting the file's order would
    #: read `b` before anything had written it.
    return Graph(
        nodes: [
            GraphNode("Relu", name: "third", inputs: ["b"], outputs: ["c"]),
            GraphNode("Neg", name: "second", inputs: ["a"], outputs: ["b"]),
            GraphNode("Abs", name: "first", inputs: ["x"], outputs: ["a"]),
        ],
        inputs: ["x"],
        outputs: ["c"]
    )
}

fn test_topological_reorders_a_chain() -> void {
    let order = chain_graph().topological()
    assert_eq([node.name for node in order], ["first", "second", "third"])
}

fn test_topological_runs_a_reordered_chain() -> void {
    #: relu(-abs(x)) is zero for every x, which no other ordering produces.
    let feed = Tensor([4], [-1.0, 2.0, -3.0, 0.5])
    let produced = run_graph(chain_graph(), {"x": feed})
    assert_eq(produced["c"].download(), [0.0, 0.0, 0.0, 0.0])
    produced["c"].free()
    feed.free()
}

fn diamond_graph() -> Graph {
    return Graph(
        nodes: [
            GraphNode("Add", name: "join", inputs: ["left", "right"], outputs: ["out"]),
            GraphNode("Neg", name: "right_arm", inputs: ["split"], outputs: ["right"]),
            GraphNode("Abs", name: "left_arm", inputs: ["split"], outputs: ["left"]),
            GraphNode("Relu", name: "fork", inputs: ["x"], outputs: ["split"]),
        ],
        inputs: ["x"],
        outputs: ["out"]
    )
}

fn test_topological_orders_a_diamond() -> void {
    let order = diamond_graph().topological()
    var place: dict[str, int] = {}
    for (index, node) in order.enumerate() { place[node.name] = index }
    assert_eq(order.len(), 4)
    assert_true(place["fork"] < place["left_arm"], "the fork must run before its left arm")
    assert_true(place["fork"] < place["right_arm"], "the fork must run before its right arm")
    assert_true(place["left_arm"] < place["join"], "the left arm must run before the join")
    assert_true(place["right_arm"] < place["join"], "the right arm must run before the join")
}

fn test_a_diamond_runs_both_arms() -> void {
    #: abs(relu(x)) + -relu(x) is zero everywhere, and is not zero if either
    #: arm reads a stale `split` or the join runs first.
    let feed = Tensor([5], [-2.0, -0.5, 0.0, 1.5, 4.0])
    let produced = run_graph(diamond_graph(), {"x": feed})
    assert_eq(produced["out"].download(), [0.0, 0.0, 0.0, 0.0, 0.0])
    produced["out"].free()
    feed.free()
}

fn test_topological_names_the_nodes_in_a_cycle() -> void {
    let graph = Graph(
        nodes: [
            GraphNode("Relu", name: "ping", inputs: ["pong_out"], outputs: ["ping_out"]),
            GraphNode("Neg", name: "pong", inputs: ["ping_out"], outputs: ["pong_out"]),
        ],
        inputs: [],
        outputs: ["pong_out"]
    )
    let error = assert_throws(ValueError, || graph.topological())
    assert_contains(error.message, "cycle")
    assert_contains(error.message, "ping")
    assert_contains(error.message, "pong")
}

fn test_topological_names_a_value_nothing_produces() -> void {
    let graph = Graph(
        nodes: [GraphNode("Add", name: "adder", inputs: ["x", "ghost"], outputs: ["y"])],
        inputs: ["x"],
        outputs: ["y"]
    )
    let error = assert_throws(ValueError, || graph.topological())
    assert_contains(error.message, "ghost")
    assert_contains(error.message, "adder")
    assert_contains(error.message, "nothing in the graph produces")
}

fn test_an_unimplemented_operator_names_itself() -> void {
    assert_true(not is_implemented("Fictional"), "Fictional is not a real operator")
    let graph = Graph(
        nodes: [GraphNode("Fictional", name: "bar", inputs: ["x"], outputs: ["y"])],
        inputs: ["x"],
        outputs: ["y"]
    )
    let feed = Tensor([2], [1.0, 2.0])
    let error = assert_throws(ValueError, || run_graph(graph, {"x": feed}))
    assert_eq(error.message, "operator Fictional at node bar is not implemented")
    assert_true(not feed.is_freed(), "a failed run must not free the caller's feed")
    feed.free()
}

fn test_an_unsupported_attribute_names_the_node() -> void {
    let graph = Graph(
        nodes: [
            GraphNode(
                "Conv",
                name: "dilated",
                inputs: ["x", "w"],
                outputs: ["y"],
                attrs: {"dilations": Attr.of_ints([2, 2])}
            ),
        ],
        inputs: ["x", "w"],
        outputs: ["y"]
    )
    let x = Tensor([1, 1, 5, 5], [float(value) for value in 0..25])
    let w = Tensor([1, 1, 3, 3], [1.0 for _value in 0..9])
    let error = assert_throws(ValueError, || run_graph(graph, {"x": x, "w": w}))
    assert_contains(error.message, "operator Conv at node dilated")
    assert_contains(error.message, "dilations")
    x.free()
    w.free()
}

fn test_feeds_and_initializers_are_never_freed() -> void {
    let weight = Tensor([3], [1.0, 2.0, 3.0])
    let graph = Graph(
        nodes: [
            GraphNode("Mul", name: "scale", inputs: ["x", "w"], outputs: ["y"]),
            GraphNode(
                "Identity",
                name: "passthrough",
                inputs: ["w"],
                outputs: ["w_out"]
            ),
        ],
        inputs: ["x"],
        outputs: ["y", "w_out"],
        initializers: {"w": weight}
    )
    let feed = Tensor([2, 3], [1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    let first = run_graph(graph, {"x": feed})
    assert_eq(first["y"].download(), [1.0, 2.0, 3.0, 2.0, 4.0, 6.0])
    assert_eq(first["w_out"].download(), [1.0, 2.0, 3.0])
    for (_name, tensor) in first.items() { tensor.free() }

    #: Freeing the results must not have reached the initializer or the feed,
    #: so the very same graph runs again and gives the very same answer.
    assert_true(not weight.is_freed(), "the initializer was freed by the run")
    assert_true(not feed.is_freed(), "the feed was freed by the run")
    let second = run_graph(graph, {"x": feed})
    assert_eq(second["y"].download(), [1.0, 2.0, 3.0, 2.0, 4.0, 6.0])
    for (_name, tensor) in second.items() { tensor.free() }
    weight.free()
    feed.free()
}

fn test_a_long_chain_of_intermediates_runs() -> void {
    #: Sixty-four copies of a tensor: if nothing were released the device
    #: would hold every link at once, and the point of the last-read
    #: bookkeeping is that it does not.
    var nodes: list[GraphNode] = []
    for step in 0..64 {
        nodes.push(
            GraphNode(
                "Identity",
                name: f"copy{step}",
                inputs: [f"v{step}"],
                outputs: [f"v{step + 1}"]
            )
        )
    }
    let graph = Graph(nodes: nodes, inputs: ["v0"], outputs: ["v64"])
    let feed = Tensor([256], [float(value) for value in 0..256])
    let produced = run_graph(graph, {"v0": feed})
    assert_eq(produced["v64"].download()[255], 255.0)
    produced["v64"].free()
    feed.free()
}

fn network_graph() -> Graph {
    return Graph(
        nodes: [
            GraphNode(
                "Conv",
                name: "conv",
                inputs: ["x", "w", "b"],
                outputs: ["conv_out"],
                attrs: {
                    "kernel_shape": Attr.of_ints([3, 3]),
                    "pads": Attr.of_ints([1, 1, 1, 1]),
                    "strides": Attr.of_ints([1, 1]),
                }
            ),
            GraphNode(
                "BatchNormalization",
                name: "norm",
                inputs: ["conv_out", "gamma", "beta", "mean", "var"],
                outputs: ["norm_out"],
                attrs: {"epsilon": Attr.of_float(1e-5)}
            ),
            GraphNode("Relu", name: "act", inputs: ["norm_out"], outputs: ["act_out"]),
            GraphNode(
                "MaxPool",
                name: "pool",
                inputs: ["act_out"],
                outputs: ["pool_out"],
                attrs: {"kernel_shape": Attr.of_ints([2, 2]), "strides": Attr.of_ints([2, 2])}
            ),
            GraphNode(
                "Flatten",
                name: "flatten",
                inputs: ["pool_out"],
                outputs: ["flat"],
                attrs: {"axis": Attr.of_int(1)}
            ),
            GraphNode(
                "Gemm",
                name: "dense",
                inputs: ["flat", "dense_w", "dense_b"],
                outputs: ["logits"],
                attrs: {"transB": Attr.of_int(1)}
            ),
            GraphNode(
                "Softmax",
                name: "probabilities",
                inputs: ["logits"],
                outputs: ["probs"],
                attrs: {"axis": Attr.of_int(1)}
            ),
        ],
        inputs: ["x"],
        outputs: ["probs"]
    )
}

#: The whole point of the module in one graph -- a convolution, the batch norm
#: that follows it, an activation, a pool, a flatten, a dense layer and a
#: softmax -- built by hand and checked against the same seven lines of torch.
fn test_end_to_end_network_matches_torch() -> void {
    let recorded = load_cases(Path("packages/jaicv/tests/oracle/dnn_graph_network.txt"))["network"]
    let item = recorded[0]
    let order = ["x", "w", "b", "gamma", "beta", "mean", "var", "dense_w", "dense_b"]
    assert_eq(item.feeds.len(), order.len())
    var weights: dict[str, Tensor] = {}
    for (index, name) in order.enumerate() {
        let feed = item.feeds[index]
        if feed is null { throw ValueError("the network oracle is missing a tensor") }
        weights[name] = feed.upload()
    }
    let graph = network_graph()
    for (name, tensor) in weights.items() {
        if name != "x" { graph.initializers[name] = tensor }
    }
    let produced = run_graph(graph, {"x": weights["x"]})
    let probs = produced["probs"]
    let want = item.wants[0]
    assert_eq(probs.shape, want.shape)
    let values = probs.download()
    var worst = 0.0
    for index in 0..values.len() {
        let gap = abs(values[index] - want.values[index])
        if gap > worst { worst = gap }
    }
    #: Measured at 1.8e-7 on an M2 Max against torch's float64 forward pass.
    assert_true(worst <= 2e-6, f"the network is off torch by {worst}")
    var total = 0.0
    for value in values { total += value }
    assert_true(abs(total - 2.0) < 1e-5, f"two softmax rows should sum to two, got {total}")
    probs.free()
    for (_name, tensor) in weights.items() { tensor.free() }
}

#: The platform differentiates the graph it already holds.
#:
#: This is what lets a whole training step -- forward, loss, gradients and the
#: update after them -- compile into one thing rather than running as a few
#: hundred separate dispatches with the processor waiting between each.
fn test_the_graph_differentiates_itself() -> void {
    if not gpu.is_available() { return }
    let n = 6
    let builder = __prim__.graph_new()
    let x = __prim__.graph_input(builder, [n])
    let w = __prim__.graph_input(builder, [n])
    #: loss = sum(x * w * w), whose derivative in w is 2 * x * w.
    let squared = __prim__.graph_binary(builder, w, w, 2)
    let scaled = __prim__.graph_binary(builder, x, squared, 2)
    let loss = __prim__.graph_reduce(builder, scaled, [0], 0)
    let slots = __prim__.graph_gradients(builder, loss, [w])
    assert_eq(slots.len(), 1)
    assert_true(slots[0] >= 0, "no derivative came back")

    let handle = __prim__.graph_compile(builder, [x, w], [slots[0]])
    assert_true(handle >= 0, "the differentiated graph would not compile")
    let plan = Plan(handle, ["x", "w"], [[n], [n]], ["dw"], [[n]])
    let xs = Tensor([n], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    let ws = Tensor([n], [0.5, -1.0, 2.0, 0.25, -3.0, 1.5])
    let produced = plan.run({"x": xs, "w": ws})
    let got = produced["dw"].download()
    for index in 0..n {
        assert_near(got[index], 2.0 * xs.download()[index] * ws.download()[index], 1e-5)
    }
}
