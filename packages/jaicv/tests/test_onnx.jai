#: Reading ONNX models, against the runtimes that wrote them.
#:
#: `tools/onnx_fixtures.py` exports five small networks out of torch, runs each
#: one, and writes what came back into `tests/models/onnx_cases.txt`. Nothing
#: here reads the specification and decides what a `Conv` ought to do: the
#: numbers being compared against are torch's own, so an importer that mislays
#: a transposed weight or a bias fails whatever the specification says.
#:
#: The three fixtures torch cannot write are built with `onnx.helper` instead.
#: One holds an initializer of every data type the importer decodes, in both
#: encodings, which is checked exactly -- decoding bytes is not arithmetic and
#: a decoder that is close is a decoder that is wrong. One is an opset-11
#: `Softmax` over a rank-4 input, where the pre-13 rule gives a different
#: answer from the modern one and onnxruntime says which; that case is the
#: whole reason the importer rewrites those nodes rather than passing them on.
#: The last few hold nothing runnable at all and exist to be refused.
#:
#: The fixtures are checked in, so this needs neither torch, onnx, nor
#: onnxruntime installed.
import std.io as io
import std.math
from jaicv.core.types import CV_32F, CV_MAKETYPE
from jaicv.core.mat import Mat
from jaicv.dnn.graph import run_graph
from jaicv.dnn.onnx import free_onnx_weights, read_net_from_onnx, read_net_from_onnx_as_net, read_net_from_onnx_bytes, read_onnx_engine
from jaitensor import Tensor
from std.test import assert_contains, assert_eq, assert_throws, assert_true

let MODELS = "packages/jaicv/tests/models/"

#: One tensor the reference recorded: `<kind> <name> <rank> <dims...> <values...>`.
class Recorded {
    pub let name: str
    pub let shape: list[int]
    pub let values: list[float]

    pub fn init(self, name: str, shape: list[int], values: list[float]) {
        self.name = name
        self.shape = shape
        self.values = values
    }

    pub fn upload(self) -> Tensor { return Tensor(self.shape, self.values) }
}

#: One model, its input, what the reference produced, and any weight whose
#: decoding is being checked without running anything.
class Case {
    pub let file: str
    pub var feeds: list[Recorded]
    pub var wants: list[Recorded]
    pub var weights: list[Recorded]

    pub fn init(self, file: str) {
        self.file = file
        self.feeds = []
        self.wants = []
        self.weights = []
    }

    pub fn path(self) -> str { return MODELS + self.file }
}

#: A recorded value. The three that have no digits are written as words,
#: because a float that cannot be printed cannot be read back either.
fn number(text: str) -> float {
    if text == "nan" { return math.NAN }
    if text == "inf" { return math.INF }
    if text == "-inf" { return -math.INF }
    return float(text)
}

fn read_recorded(fields: list[str], start: int) -> Recorded {
    let rank = int(fields[start + 1])
    var shape: list[int] = []
    for index in 0..rank { shape.push(int(fields[start + 2 + index])) }
    var values: list[float] = []
    for index in start + 2 + rank..fields.len() {
        if fields[index].len() == 0 { continue }
        values.push(number(fields[index]))
    }
    return Recorded(fields[start], shape, values)
}

fn load_cases() -> dict[str, Case] {
    var cases: dict[str, Case] = {}
    var current: Case? = null
    for line in io.read_file(MODELS + "onnx_cases.txt").split("\n") {
        if line.len() == 0 or line[0] == "#" { continue }
        let fields = line.split(" ")
        if fields[0] == "model" {
            let started = Case(fields[1])
            cases[fields[1]] = started
            current = started
            continue
        }
        let item = current
        if item is null { throw ValueError("the case file has a line before its first model") }
        let recorded = read_recorded(fields, 1)
        if fields[0] == "feed" {
            item.feeds.push(recorded)
        } elif fields[0] == "want" {
            item.wants.push(recorded)
        } elif fields[0] == "init" {
            item.weights.push(recorded)
        } else {
            throw ValueError(f"the case file has a line starting {fields[0]}")
        }
    }
    return cases
}

var _cases: dict[str, Case] = {}

fn case_for(file: str) -> Case {
    if _cases.len() == 0 { _cases = load_cases() }
    let found = _cases.get(file, null)
    if found is null { throw ValueError(f"the case file has nothing for {file}") }
    return found
}

#: How far one value is from the reference, scaled so that anything above one
#: is judged relatively and anything below it absolutely.
#:
#: The infinities and the NaN are compared as themselves rather than
#: subtracted, since a difference of two infinities is a NaN and would pass
#: every threshold there is.
fn difference(got: float, want: float) -> float {
    if want != want { return if got != got { 0.0 } else { 1.0 } }
    if got != got { return 1.0 }
    if want == math.INF or want == -math.INF { return if got == want { 0.0 } else { 1.0 } }
    if got == math.INF or got == -math.INF { return 1.0 }
    return abs(got - want) / max(1.0, abs(want))
}

fn compare(got: list[float], want: Recorded, what: str) -> float {
    assert_eq(got.len(), want.values.len(), f"{what} has the wrong number of values")
    var worst = 0.0
    for (index, value) in got.enumerate() {
        let gap = difference(value, want.values[index])
        if gap > worst { worst = gap }
    }
    return worst
}

#: Import one model, run it on what the reference was given, and return the
#: worst difference from what the reference produced.
fn worst_difference(file: str) -> float {
    let item = case_for(file)
    let graph = read_net_from_onnx(item.path())
    defer { free_onnx_weights(graph) }
    assert_eq(graph.inputs.len(), item.feeds.len(), f"{file} does not take one feed per input")
    var feeds: dict[str, Tensor] = {}
    for (index, name) in graph.inputs.enumerate() { feeds[name] = item.feeds[index].upload() }
    defer {
        for (_name, tensor) in feeds.items() { tensor.free() }
    }
    let produced = run_graph(graph, feeds)
    defer {
        for (_name, tensor) in produced.items() { tensor.free() }
    }
    var worst = 0.0
    for (index, name) in graph.outputs.enumerate() {
        let want = item.wants[index]
        let output = produced[name]
        assert_eq(output.shape, want.shape, f"{file} output {name} has the wrong shape")
        let gap = compare(output.download(), want, f"{file} output {name}")
        if gap > worst { worst = gap }
    }
    return worst
}

fn check(file: str, tolerance: float) -> void {
    let worst = worst_difference(file)
    assert_true(worst <= tolerance, f"{file} is off the reference by {worst}")
}

#: Two layers and an activation, which is the smallest thing that catches a
#: `Gemm` whose `transB` was ignored. Measured at 3.0e-8 on an M2 Max.
fn test_mlp_matches_torch() -> void { check("onnx_mlp.onnx", 1e-6) }

#: A convolution, the batch norm that follows it, an activation, a pool, a
#: flatten, a dense layer and a softmax -- exported without constant folding so
#: the batch norm is still its own node rather than multiplied into the
#: convolution's weights. Measured at 8.9e-8.
fn test_cnn_matches_torch() -> void { check("onnx_cnn.onnx", 1e-6) }

#: A residual `Add`, where the skip connection reads a value two nodes older
#: than the one beside it. Measured at 1.2e-7, the worst of the five.
fn test_residual_matches_torch() -> void { check("onnx_residual.onnx", 1e-6) }

#: Two branches joined by a `Concat`. Measured at 3.0e-8.
fn test_concat_matches_torch() -> void { check("onnx_concat.onnx", 1e-6) }

#: A `Transpose` and a `Reshape`, whose shape torch exports as a `Constant`
#: node rather than an initializer. Exact, since neither operator does
#: arithmetic.
fn test_shuffle_matches_torch() -> void { check("onnx_shuffle.onnx", 1e-6) }

#: Every shape a `Constant` node is written in, and what folding them does.
fn test_constants_become_weights() -> void {
    let item = case_for("onnx_constants.onnx")
    let graph = read_net_from_onnx(item.path())
    defer { free_onnx_weights(graph) }
    for node in graph.nodes {
        assert_true(node.op != "Constant", "a folded graph keeps no Constant nodes")
    }
    assert_eq(graph.nodes.len(), 3, "the three arithmetic nodes are all that should be left")
    for want in item.weights {
        let held = graph.initializers.get(want.name, null)
        assert_true(held is not null, f"{want.name} should have become a weight")
        assert_eq(held.shape, want.shape, f"{want.name} folded to the wrong shape")
        assert_eq(compare(held.download(), want, want.name), 0.0, f"{want.name} folded wrongly")
    }
    check("onnx_constants.onnx", 1e-6)
}

#: The case the rewrite exists for.
#:
#: A pre-13 `Softmax` flattens to two dimensions at its axis and normalises
#: rows of that, so on this rank-4 input each of the six values it divides by
#: covers four elements rather than three. Read the node as the modern
#: operator instead and every number comes out wrong while every shape stays
#: right, which is why the reference here is onnxruntime rather than an
#: argument.
fn test_legacy_softmax_matches_onnxruntime() -> void {
    let graph = read_net_from_onnx(MODELS + "onnx_legacy_softmax.onnx")
    defer { free_onnx_weights(graph) }
    var ops: list[str] = []
    for node in graph.nodes { ops.push(node.op) }
    ops.sort()
    assert_eq(ops, ["Flatten", "Reshape", "Shape", "Softmax"], "the pre-13 rule is four operators")
    #: This file leaves the axis out, so the pre-13 default of 1 is what is
    #: being checked as much as the coercion is -- the modern default is -1.
    check("onnx_legacy_softmax.onnx", 1e-6)
}

#: The same, with the axis written down, so that reading it is checked as well
#: as defaulting it.
fn test_legacy_softmax_honours_a_written_axis() -> void {
    check("onnx_legacy_softmax_axis.onnx", 1e-6)
}

#: What a pre-13 `Softmax` means depends on an opset the file did not declare,
#: and there is no reading of it that is safe to guess at.
fn test_a_softmax_with_no_opset_declared_is_refused() -> void {
    let error = assert_throws(ValueError, || read_net_from_onnx(MODELS + "onnx_no_opset.onnx"))
    assert_contains(error.message, "declares no opset")
    assert_contains(error.message, "legacy")
}

#: A weight whose `raw_data` is a byte short of what its dimensions call for.
#: The length is checked against the dimensions before a byte is read, so this
#: is refused rather than turned into however many values the bytes make.
fn test_a_weight_short_of_its_bytes_is_refused() -> void {
    let error = assert_throws(ValueError, || read_net_from_onnx(MODELS + "onnx_short_weight.onnx"))
    assert_contains(error.message, "the initializer weight")
    assert_contains(error.message, "16 bytes")
}

#: Every data type, in `raw_data` and in the repeated field it also has.
#:
#: Exactly, with no tolerance at all: these are bytes the file holds and the
#: importer either puts them back together or does not. The values were chosen
#: at the edges -- both zeroes, the widest value of each integer type, a
#: subnormal of each float width, both infinities and a NaN -- because that is
#: where a decoder written from the shape of the format rather than from the
#: standard goes wrong.
fn test_every_data_type_decodes() -> void {
    let item = case_for("onnx_dtypes.onnx")
    let graph = read_net_from_onnx(item.path())
    defer { free_onnx_weights(graph) }
    assert_true(item.weights.len() >= 28, "the data type fixture should cover every type twice")
    for want in item.weights {
        let held = graph.initializers.get(want.name, null)
        assert_true(held is not null, f"{want.name} is missing from the imported graph")
        assert_eq(held.shape, want.shape, f"{want.name} decoded to the wrong shape")
        assert_eq(
            compare(held.download(), want, want.name),
            0.0,
            f"{want.name} did not decode to what wrote it"
        )
    }
}

#: A rank-zero tensor is the one shape ONNX has that jaitensor does not, and it
#: arrives as a single element rather than as an error.
fn test_a_scalar_weight_becomes_one_element() -> void {
    let graph = read_net_from_onnx(MODELS + "onnx_dtypes.onnx")
    defer { free_onnx_weights(graph) }
    assert_eq(graph.initializers["scalar"].shape, [1])
    assert_eq(graph.initializers["matrix"].shape, [2, 3])
}

#: A weight the file also lists as an input is a default, not something the
#: caller has to supply.
fn test_weights_are_not_asked_for_as_inputs() -> void {
    let graph = read_net_from_onnx(MODELS + "onnx_cnn.onnx")
    defer { free_onnx_weights(graph) }
    assert_eq(graph.inputs, ["input"], "only the image should be asked for")
    assert_true(graph.initializers.len() > 0, "the convolution has to have brought weights")
    for name in graph.inputs {
        assert_true(
            graph.initializers.get(name, null) is null,
            f"{name} is both asked for and supplied"
        )
    }
}

fn test_a_file_that_is_not_onnx_is_refused() -> void {
    #: A ZIP's first four bytes. Read as protobuf this is field 10 as a
    #: varint, then a tag naming field zero, which protobuf disallows.
    let error = assert_throws(
        ValueError,
        || read_net_from_onnx_bytes(bytes([0x50, 0x4B, 0x03, 0x04]))
    )
    assert_contains(error.message, "not a readable ONNX model")
}

fn test_a_protobuf_that_is_not_a_model_is_refused() -> void {
    #: Field 1 holding the varint 3, which parses cleanly and is not a model.
    let error = assert_throws(ValueError, || read_net_from_onnx_bytes(bytes([0x08, 0x03])))
    assert_contains(error.message, "holds no graph")
}

fn test_a_truncated_model_is_refused() -> void {
    let whole = io.read_bytes(MODELS + "onnx_cnn.onnx")
    let cut = bytes([whole[index] for index in 0..whole.len() * 2 // 3])
    let error = assert_throws(ValueError, || read_net_from_onnx_bytes(cut))
    assert_contains(error.message, "not a readable ONNX model")
}

fn test_an_unimplemented_operator_is_refused_before_the_forward_pass() -> void {
    let error = assert_throws(ValueError, || read_net_from_onnx(MODELS + "onnx_unsupported.onnx"))
    assert_eq(
        error.message,
        "operator Celu at node curved is not an operator this executor implements"
    )
}

fn test_somebody_elses_operator_set_is_refused() -> void {
    let error = assert_throws(ValueError, || read_net_from_onnx(MODELS + "onnx_custom_domain.onnx"))
    assert_contains(error.message, "com.microsoft")
    assert_contains(error.message, "fused")
}

fn test_weights_in_another_file_are_refused() -> void {
    let error = assert_throws(ValueError, || read_net_from_onnx(MODELS + "onnx_external.onnx"))
    assert_contains(error.message, "weight.bin")
}

#: The `Net` an OpenCV detector would be holding, fed the blob layout
#: `blob_from_image` produces: `N * C` rows of `H * W`.
fn test_the_net_wrapper_runs_the_cnn() -> void {
    let item = case_for("onnx_cnn.onnx")
    let feed = item.feeds[0]
    let net = read_net_from_onnx_as_net(item.path())
    assert_true(not net.empty(), "a Net around a model is not empty")
    let blob = Mat.from_list(
        feed.shape[0] * feed.shape[1],
        feed.shape[2] * feed.shape[3],
        CV_MAKETYPE(CV_32F, 1),
        feed.values
    )
    net.set_input(blob)
    let output = net.forward()
    let want = item.wants[0]
    assert_eq([output.rows, output.cols], want.shape, "the output blob is the wrong shape")
    let worst = compare(output.values(), want, "the Net's output")
    assert_true(worst <= 1e-6, f"the Net is off torch by {worst}")
}

#: The engine behind that `Net`, which is what a caller holds when the weights
#: have to be freed again.
fn test_the_engine_reports_the_declared_shape() -> void {
    let engine = read_onnx_engine(MODELS + "onnx_cnn.onnx")
    defer { free_onnx_weights(engine.graph) }
    assert_eq(engine.input_name, "input")
    assert_eq(engine.input_shape, [1, 1, 8, 8])
    assert_eq(engine.output_name, "output")
}

#: A blob with more values than the declared shape holds is a mistake worth
#: naming, not something to reshape around.
fn test_a_blob_of_the_wrong_size_is_refused() -> void {
    let engine = read_onnx_engine(MODELS + "onnx_cnn.onnx")
    defer { free_onnx_weights(engine.graph) }
    let blob = Mat.from_list(1, 63, CV_MAKETYPE(CV_32F, 1), [0.0 for _index in 0..63])
    let error = assert_throws(ValueError, || engine.run(blob))
    assert_contains(error.message, "input")
}

#: The exporter option that lists every weight as a graph input beside the
#: image, which is also what every file written before IR version 4 looks
#: like. A weight that arrives with the model is not something to ask the
#: caller for.
fn test_weights_listed_as_inputs_are_not_asked_for() -> void {
    let graph = read_net_from_onnx(MODELS + "onnx_weights_as_inputs.onnx")
    defer { free_onnx_weights(graph) }
    assert_eq(graph.initializers.len(), 4, "the two dense layers bring four weights")
    assert_eq(graph.inputs, ["input"], "only the image should be asked for")
    check("onnx_weights_as_inputs.onnx", 1e-6)
}

#: A model exported with a free batch, run at a batch it was not traced at.
fn test_a_symbolic_batch_is_settled_by_the_blob() -> void {
    let item = case_for("onnx_dynamic_batch.onnx")
    let engine = read_onnx_engine(item.path())
    defer { free_onnx_weights(engine.graph) }
    assert_eq(engine.input_shape, [-1, 6], "the batch should have come back symbolic")
    let feed = item.feeds[0]
    assert_eq(feed.shape[0], 5, "the reference should have run at a batch of five")
    let blob = Mat.from_list(feed.shape[0], feed.shape[1], CV_MAKETYPE(CV_32F, 1), feed.values)
    let output = engine.run(blob)
    let want = item.wants[0]
    assert_eq([output.rows, output.cols], want.shape, "the batch did not survive the forward pass")
    let worst = compare(output.values(), want, "the dynamic-batch output")
    assert_true(worst <= 1e-6, f"the dynamic-batch model is off torch by {worst}")
}

#: Attributes with `AttributeProto.type` stripped, as an exporter old enough to
#: predate that field wrote them. Which shape each one is in then has to be
#: worked out from the payload field that is present, and getting it wrong
#: writes a field the executor never reads -- so this covers an `ints` perm, an
#: `int` axis, a `float` alpha, a `str` pad mode and a `floats` constant at
#: once.
fn test_attributes_without_a_declared_type() -> void {
    let item = case_for("onnx_untyped_attrs.onnx")
    let graph = read_net_from_onnx(item.path())
    defer { free_onnx_weights(graph) }
    for node in graph.nodes {
        if node.op == "Transpose" { assert_eq(node.attrs["perm"].kind, "ints") }
        if node.op == "Flatten" { assert_eq(node.attrs["axis"].kind, "int") }
        if node.op == "LeakyRelu" { assert_eq(node.attrs["alpha"].kind, "float") }
        if node.op == "Pad" { assert_eq(node.attrs["mode"].kind, "str") }
    }
    let bias = item.weights[0]
    assert_eq(
        compare(graph.initializers[bias.name].download(), bias, bias.name),
        0.0,
        "the untyped value_floats constant folded wrongly"
    )
    check("onnx_untyped_attrs.onnx", 1e-6)
}

#: Reshape carried its shape as an attribute until opset 5 and as an input
#: afterwards. The executor reads the input only, so the old form has to become
#: a weight and a second input on the way in.
fn test_reshape_with_its_shape_as_an_attribute() -> void {
    let graph = read_net_from_onnx(MODELS + "onnx_legacy_reshape.onnx")
    defer { free_onnx_weights(graph) }
    assert_eq(graph.nodes.len(), 1)
    let node = graph.nodes[0]
    assert_eq(node.inputs.len(), 2, "the shape should have become the second input")
    let shape = graph.initializers.get(node.inputs[1], null)
    assert_true(shape is not null, "the shape attribute should have become a weight")
    assert_eq(shape.download(), [4.0, 6.0])
    check("onnx_legacy_reshape.onnx", 1e-6)
}
