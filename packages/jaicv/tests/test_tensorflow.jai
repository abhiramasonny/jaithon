#: Reading frozen TensorFlow graphs, against TensorFlow's own answers.
#:
#: `tools/tensorflow_fixtures.py` builds a graph per test with TensorFlow
#: 2.21, freezes each one with `convert_variables_to_constants_v2`, writes the
#: binary `GraphDef` into `tests/models/`, then runs the frozen graph and
#: records the input it ran on and the output it produced in
#: `tensorflow_cases.txt`. Both are written already permuted into `NCHW`, so
#: this test feeds and compares without reordering anything: a transposition
#: this importer gets wrong has nothing here to cancel against.
#:
#: Every extent in every fixture differs -- an 11x7 image through a 3x2 kernel,
#: a 7x4 image through a depthwise 3x2 -- because a swapped `H` and `W`, or an
#: `HWIO` filter read as `OIHW`, comes out of a square graph looking perfectly
#: plausible.
#:
#: `tools/tensorflow_hostile_fixtures.py` builds a second kind of fixture the
#: same way `tests/test_protobuf.jai`'s malformed cases are built: by hand,
#: straight against TensorFlow's own protobuf message classes rather than by
#: tracing a `tf.function`. TensorFlow's tracer would refuse half of these
#: itself -- there is no way to trace a node with too few inputs, and `Relu`
#: has no complex kernel -- so what a real graph would never produce is built
#: directly instead, the way a hand-edited or truncated model file could
#: arrive at this reader's door.
import std.io as io
import std.math
from jaicv.dnn.graph import Attr, run_graph
from jaicv.dnn.tensorflow import read_net_from_tensorflow, read_net_from_tensorflow_bytes
from jaitensor import Tensor
from std.test import assert_eq, assert_true

let MODELS = "packages/jaicv/tests/models/"

#: One tensor as TensorFlow had it.
class Recorded {
    pub let shape: list[int]
    pub let values: list[float]

    pub fn init(self, shape: list[int], values: list[float]) {
        self.shape = shape
        self.values = values
    }

    pub fn upload(self) -> Tensor {
        #: TensorFlow has rank-zero tensors and the executor does not.
        let dims = if self.shape.len() == 0 { [1] } else { self.shape }
        return Tensor(dims, self.values)
    }
}

#: One frozen graph, its feed, and what TensorFlow produced from it.
class Case {
    pub let name: str
    pub var inputs: list[str]
    pub var outputs: list[str]
    pub var feeds: list[Recorded]
    pub var wants: list[Recorded]

    pub fn init(self, name: str) {
        self.name = name
        self.inputs = []
        self.outputs = []
        self.feeds = []
        self.wants = []
    }
}

fn words(line: str) -> list[str] {
    var out: list[str] = []
    for part in line.split(" ") {
        if part.len() > 0 { out.push(part) }
    }
    return out
}

fn read_recorded(parts: list[str]) -> Recorded {
    let rank = int(parts[1])
    var shape: list[int] = []
    for axis in 0..rank { shape.push(int(parts[2 + axis])) }
    let count = int(parts[2 + rank])
    var values: list[float] = []
    for index in 0..count { values.push(float(parts[3 + rank + index])) }
    return Recorded(shape, values)
}

fn load_cases() -> list[Case] {
    var cases: list[Case] = []
    var current: Case? = null
    for line in io.read_file(MODELS + "tensorflow_cases.txt").split("\n") {
        let parts = words(line)
        if parts.len() == 0 or parts[0].starts_with("#") { continue }
        if parts[0] == "graph" {
            let started = Case(parts[1])
            cases.push(started)
            current = started
            continue
        }
        let item = current
        if item is null { throw ValueError("the oracle file has a line before its first graph") }
        if parts[0] == "input" {
            item.inputs.push(parts[1])
        } elif parts[0] == "output" {
            item.outputs.push(parts[1])
        } elif parts[0] == "feed" {
            item.feeds.push(read_recorded(parts))
        } elif parts[0] == "want" {
            item.wants.push(read_recorded(parts))
        } else {
            throw ValueError(f"the oracle file has a line starting {parts[0]}")
        }
    }
    return cases
}

var _cases: list[Case] = []

fn case_named(name: str) -> Case {
    if _cases.len() == 0 { _cases = load_cases() }
    for item in _cases {
        if item.name == name { return item }
    }
    throw ValueError(f"the oracle file has no graph called {name}")
}

#: How far one graph's output is from TensorFlow's, as the largest absolute
#: difference over every output tensor.
fn disagreement(name: str) -> float {
    let item = case_named(name)
    let graph = read_net_from_tensorflow(MODELS + name + ".pb")
    defer { for (_key, weight) in graph.initializers.items() { weight.free() } }
    assert_eq(graph.inputs.len(), item.inputs.len(), f"{name} takes a different number of inputs")
    for (slot, wanted) in item.inputs.enumerate() {
        assert_eq(graph.inputs[slot], wanted, f"{name} names its input differently")
    }
    assert_eq(
        graph.outputs.len(),
        item.outputs.len(),
        f"{name} produces a different number of outputs"
    )
    var feeds: dict[str, Tensor] = {}
    for (slot, fed) in item.feeds.enumerate() { feeds[item.inputs[slot]] = fed.upload() }
    defer { for (_key, fed) in feeds.items() { fed.free() } }
    let produced = run_graph(graph, feeds)
    defer { for (_key, got) in produced.items() { got.free() } }
    var worst = 0.0
    for (slot, wanted) in item.outputs.enumerate() {
        let found = produced.get(wanted, null)
        assert_true(found is not null, f"{name} produced no output called {wanted}")
        let got: Tensor = found
        let expected = item.wants[slot]
        let dims = if expected.shape.len() == 0 { [1] } else { expected.shape }
        assert_eq(got.shape, dims, f"{name}'s output {wanted} came out the wrong shape")
        let values = got.download()
        for index in 0..values.len() {
            let apart = abs(values[index] - expected.values[index])
            if apart > worst { worst = apart }
        }
    }
    return worst
}

fn test_mlp_matches_tensorflow() -> void {
    let worst = disagreement("tf_mlp")
    assert_true(worst < 1e-6, f"the MLP is {worst} away from TensorFlow")
}

fn test_cnn_matches_tensorflow() -> void {
    let worst = disagreement("tf_cnn")
    assert_true(worst < 5e-6, f"the CNN is {worst} away from TensorFlow")
}

fn test_depthwise_matches_tensorflow() -> void {
    let worst = disagreement("tf_depthwise")
    assert_true(worst < 5e-6, f"the depthwise graph is {worst} away from TensorFlow")
}

fn test_mean_matches_tensorflow() -> void {
    let worst = disagreement("tf_mean")
    assert_true(worst < 5e-6, f"the mean graph is {worst} away from TensorFlow")
}

fn test_misc_matches_tensorflow() -> void {
    let worst = disagreement("tf_misc")
    assert_true(worst < 5e-6, f"the mixed graph is {worst} away from TensorFlow")
}

#: The reductions and the squeeze whose surviving axes come out in a different
#: order here than they do in TensorFlow, which is the one class of layout
#: mistake that produces a correctly shaped answer.
fn test_scrambled_axes_match_tensorflow() -> void {
    let worst = disagreement("tf_axes")
    assert_true(worst < 1e-5, f"the reductions are {worst} away from TensorFlow")
}

#: A rank-four constant used as data, a rank-three constant and a rank-three
#: computed value broadcast against an image, a softmax over the channels of a
#: four-dimensional tensor, a concatenation on a spatial axis, and a reshape in
#: each direction across rank four -- none of which a plain classifier reaches.
fn test_layout_corners_match_tensorflow() -> void {
    let worst = disagreement("tf_layout")
    assert_true(worst < 5e-6, f"the layout graph is {worst} away from TensorFlow")
}

fn test_arithmetic_matches_tensorflow() -> void {
    let worst = disagreement("tf_arith")
    assert_true(worst < 5e-6, f"the arithmetic graph is {worst} away from TensorFlow")
}

fn test_ramp_convolutions_match_tensorflow() -> void {
    let worst = disagreement("tf_ramp")
    assert_true(worst < 1e-5, f"the ramp convolutions are {worst} away from TensorFlow")
}

fn only_ending(weights: dict[str, Tensor], suffix: str) -> Tensor {
    var found: Tensor? = null
    for (key, tensor) in weights.items() {
        if key.ends_with(suffix) {
            assert_true(found is null, f"two initializers end with {suffix}")
            found = tensor
        }
    }
    if found is null { throw ValueError(f"no initializer ends with {suffix}") }
    let only: Tensor = found
    return only
}

#: The filters in `tf_ramp` are 0, 1, 2, ... so the permuted initializer has
#: one right answer, written out here from the layout rule rather than
#: measured. A forward pass that agrees has already ruled a transposition out,
#: but only for the shapes it happened to run on; this pins the permutation
#: itself.
fn test_filters_are_transposed_element_by_element() -> void {
    let graph = read_net_from_tensorflow(MODELS + "tf_ramp.pb")
    defer { for (_key, weight) in graph.initializers.items() { weight.free() } }

    #: TensorFlow's filter is `[kH, kW, I, O]` holding `((h * 3 + w) * 4 + i) * 5 + o`,
    #: and ONNX wants that same element at `[o][i][h][w]`.
    let plain = only_ending(graph.initializers, "$oihw")
    assert_eq(plain.shape, [5, 4, 2, 3], "the convolution weights came out the wrong shape")
    let got = plain.download()
    for o in 0..5 {
        for i in 0..4 {
            for h in 0..2 {
                for w in 0..3 {
                    let wanted = float(((h * 3 + w) * 4 + i) * 5 + o)
                    let at = ((o * 4 + i) * 2 + h) * 3 + w
                    assert_eq(
                        got[at],
                        wanted,
                        f"OIHW[{o}][{i}][{h}][{w}] is not HWIO[{h}][{w}][{i}][{o}]"
                    )
                }
            }
        }
    }

    #: A depthwise filter is `[kH, kW, C, M]` and becomes `[C * M, 1, kH, kW]`,
    #: because a grouped convolution numbers output channel `c * M + m` out of
    #: input channel `c` exactly as TensorFlow does.
    let deep = only_ending(graph.initializers, "$depthwise")
    assert_eq(deep.shape, [8, 1, 2, 3], "the depthwise weights came out the wrong shape")
    let held = deep.download()
    for c in 0..4 {
        for m in 0..2 {
            for h in 0..2 {
                for w in 0..3 {
                    let wanted = float(((h * 3 + w) * 4 + c) * 2 + m)
                    let at = ((c * 2 + m) * 2 + h) * 3 + w
                    assert_eq(held[at], wanted, f"group {c} multiplier {m} at {h},{w} is wrong")
                }
            }
        }
    }
}

#: A `Placeholder` keeps its own name and the graph's outputs are the names
#: nothing else reads, which is how a caller addresses a frozen graph at all.
fn test_names_come_from_the_file() -> void {
    let graph = read_net_from_tensorflow(MODELS + "tf_cnn.pb")
    defer { for (_key, weight) in graph.initializers.items() { weight.free() } }
    assert_eq(graph.inputs, ["image"], "the placeholder did not keep its name")
    assert_eq(graph.outputs, ["Identity"], "the sink was not found")
    var ops: list[str] = []
    for node in graph.nodes { ops.push(node.op) }
    assert_true(ops.contains("Conv"), "Conv2D did not become Conv")
    assert_true(ops.contains("BatchNormalization"), "FusedBatchNormV3 did not become a norm")
    assert_true(ops.contains("Clip"), "Relu6 did not become Clip")
    assert_true(ops.contains("Gemm"), "MatMul did not become Gemm")
    #: Nothing but the flatten needs the values moved, and it needs exactly one.
    var transposes = 0
    for node in graph.nodes {
        if node.op == "Transpose" { transposes += 1 }
    }
    assert_eq(transposes, 1, "the graph moves data around more than the reshape needs")
}

#: TensorFlow's SAME divides the padding with the remainder at the end, and the
#: importer has to say `SAME_UPPER` rather than `SAME_LOWER` -- the shapes agree
#: either way and the pixels do not.
fn test_same_padding_is_upper() -> void {
    let graph = read_net_from_tensorflow(MODELS + "tf_cnn.pb")
    defer { for (_key, weight) in graph.initializers.items() { weight.free() } }
    var seen = false
    for node in graph.nodes {
        if node.name != "Conv2D" { continue }
        let pad = node.attrs.get("auto_pad", null)
        if pad is null { throw ValueError("the convolution lost its padding") }
        let held: Attr = pad
        assert_eq(held.s, "SAME_UPPER", "SAME did not become SAME_UPPER")
        seen = true
    }
    assert_true(seen, "the convolution is not in the graph")
}

#: The message a call threw, or `""` when it returned.
fn refusal_of(path: str) -> str {
    try {
        let graph = read_net_from_tensorflow(path)
        for (_key, weight) in graph.initializers.items() { weight.free() }
    } catch e: Error {
        return e.message
    }
    return ""
}

fn test_an_untranslatable_operator_is_named() -> void {
    let message = refusal_of(MODELS + "tf_unsupported.pb")
    assert_true(message != "", "a graph holding a Tile was accepted")
    assert_true(message.contains("Tile"), f"the refusal does not name the operator: {message}")
}

fn test_a_training_batch_norm_is_refused() -> void {
    #: With `is_training` set there is no running mean or variance in the file
    #: to normalise with, so the five inputs the executor wants are not all
    #: there. Reading it as inference would silently normalise by the wrong
    #: statistics.
    let message = refusal_of(MODELS + "tf_training.pb")
    assert_true(message != "", "a batch norm still in training mode was accepted")
    assert_true(message.contains("training"), f"the refusal does not say what is wrong: {message}")
}

fn test_a_text_graph_configuration_is_refused_rather_than_ignored() -> void {
    var message = ""
    try {
        let _graph = read_net_from_tensorflow(MODELS + "tf_mlp.pb", "graph.pbtxt")
    } catch e: Error {
        message = e.message
    }
    assert_true(message != "", "the configuration was accepted and then ignored")
    assert_true(
        message.contains("GraphDef"),
        f"the refusal does not say what is read instead: {message}"
    )
}

fn test_a_file_that_is_not_a_graph_is_refused() -> void {
    var message = ""
    try {
        let _graph = read_net_from_tensorflow_bytes(bytes([]))
    } catch e: Error {
        message = e.message
    }
    assert_true(message.contains("no nodes"), f"an empty file was not named as one: {message}")
}

#: Reading from a path and reading from bytes must agree, because the second
#: is what a caller who already has the file uses.
fn test_bytes_and_path_agree() -> void {
    let from_path = read_net_from_tensorflow(MODELS + "tf_mean.pb")
    defer { for (_key, weight) in from_path.initializers.items() { weight.free() } }
    let from_bytes = read_net_from_tensorflow_bytes(io.read_bytes(MODELS + "tf_mean.pb"))
    defer { for (_key, weight) in from_bytes.initializers.items() { weight.free() } }
    assert_eq(from_path.nodes.len(), from_bytes.nodes.len(), "the two readings differ in size")
    for (slot, node) in from_path.nodes.enumerate() {
        assert_eq(node.op, from_bytes.nodes[slot].op, "the two readings differ in operators")
        assert_eq(node.name, from_bytes.nodes[slot].name, "the two readings differ in names")
    }
}

#: `translate_squeeze`'s `rank != 4` path emitted the squeeze with
#: TensorFlow's own axes and then tagged a result that landed at rank four as
#: `NCHW`, without ever transposing it there. `tf_squeeze_lands` squeezes a
#: rank-five tensor -- `tf.squeeze(tf.expand_dims(x, 1), axis=[1])` -- straight
#: back down to a rank-four one and feeds it to a `Conv2D`, so a still-`NHWC`
#: result would either come out the wrong shape for the filter or the right
#: shape with the wrong values in it. Before the fix it threw a shape
#: mismatch on this exact fixture.
fn test_squeeze_realigns_when_it_lands_at_rank_four() -> void {
    let worst = disagreement("tf_squeeze_lands")
    assert_true(worst < 5e-6, f"the squeeze-then-conv graph is {worst} away from TensorFlow")
}

#: The same invariant break in `translate_reduce`: `tf.reduce_sum(tf.expand_dims(x, 2), axis=[2])`
#: on a rank-four `[None, 5, 3, 2]` input reduces the newly inserted axis of a
#: rank-five tensor and lands back at rank four still in TensorFlow's order.
#: Before the fix this came back `NHWC`-shaped -- `[2, 5, 3, 2]` -- instead of
#: the `[2, 2, 5, 3]` NCHW the rest of the graph expects.
fn test_reduce_realigns_when_it_lands_at_rank_four() -> void {
    let worst = disagreement("tf_reduce_lands")
    assert_true(worst < 1e-5, f"the reduce-lands graph is {worst} away from TensorFlow")
}

#: `translate_arg_extreme` drops exactly one axis the same way a
#: non-keepdims reduction does and had the identical gap: an `ArgMax` on a
#: rank-five tensor that lands at rank four came out still `NHWC`. Not one of
#: the seven defects reported, but the same invariant break in the same file,
#: fixed with the same `lands_in_nhwc` rule as `Squeeze` and the reductions.
fn test_argmax_realigns_when_it_lands_at_rank_four() -> void {
    let worst = disagreement("tf_argmax_lands")
    assert_true(worst < 1e-5, f"the argmax-lands graph is {worst} away from TensorFlow")
}

#: `broadcast_input` returned `value_input` unchanged whenever the node's own
#: result rank was not four, which is right for every rank except rank four
#: itself: a rank-four constant broadcast against a rank-five operand still
#: got permuted into `NCHW` by `value_input`, while the rank-five operand
#: stayed in TensorFlow's order -- `tf.add(tf.expand_dims(x, 1), tf.constant(C4))`
#: on a `[None, 4, 4, 3]` input. Before the fix this threw a broadcast
#: mismatch on this exact fixture, because the permuted constant's shape no
#: longer lined up on the trailing axes the way TensorFlow's own does.
fn test_broadcast_realigns_a_rank_four_operand() -> void {
    let worst = disagreement("tf_broadcast_lands")
    assert_true(worst < 5e-6, f"the broadcast-lands graph is {worst} away from TensorFlow")
}

#: `typed_values` sent every `WHOLE_DTYPES` entry through `int_val` (field 7),
#: but `TensorProto` stores `uint32` in `uint32_val` (field 16) and `uint64`
#: in `uint64_val` (field 17). `tf_uint_consts.pb` is hand-built -- TensorFlow
#: itself never takes this path, always spending `tensor_content` on a
#: numeric array -- with a `uint32` and a `uint64` constant that hold only
#: their own typed field, the way a smaller or older encoder might. Before
#: the fix, `read_tensor_proto` found nothing in `int_val`, and "no values at
#: all means every element is zero" silently turned both into zero.
fn test_uint_constants_are_not_read_as_zero() -> void {
    let graph = read_net_from_tensorflow(MODELS + "tf_uint_consts.pb")
    defer { for (_key, weight) in graph.initializers.items() { weight.free() } }
    let feed = Tensor([3], [1.0, 2.0, 3.0])
    defer { feed.free() }
    var feeds: dict[str, Tensor] = {"x": feed}
    let produced = run_graph(graph, feeds)
    defer { for (_key, tensor) in produced.items() { tensor.free() } }
    let found32 = produced.get("out32", null)
    assert_true(found32 is not null, "the uint32 add produced no output")
    let got32: Tensor = found32
    assert_eq(got32.download(), [11.0, 22.0, 33.0], "a uint32 constant was read as zero")
    let found64 = produced.get("out64", null)
    assert_true(found64 is not null, "the uint64 add produced no output")
    let got64: Tensor = found64
    assert_eq(got64.download(), [101.0, 202.0, 303.0], "a uint64 constant was read as zero")
}

#: `read_tensor_proto` materialised `count` -- the declared shape's product --
#: elements whenever the message held zero or one actual values, with no
#: check against how many bytes the file could have backed that with. A
#: model file is untrusted input, so `tf_huge_shape.pb` is a fifty-three-byte
#: `GraphDef` holding one `Const` that declares a shape of four billion
#: elements and carries no data at all. Before the fix this asked for an
#: allocation of four billion floats from fifty-three bytes; now it is a
#: refusal that names the mismatch.
fn test_a_huge_declared_shape_is_refused() -> void {
    let message = refusal_of(MODELS + "tf_huge_shape.pb")
    assert_true(
        message != "",
        "a constant declaring four billion elements from a 53-byte file was accepted"
    )
    assert_true(
        message.contains("4000000000"),
        f"the refusal does not name the declared count: {message}"
    )
}

#: Most translators indexed `node.inputs` by position without checking its
#: length first, so a malformed file produced a bare index error instead of
#: the named refusal the module documents elsewhere. `tf_missing_unary.pb`
#: holds one `Relu` with no inputs at all, which used to fail as
#: "list index 0 out of range for length 0".
fn test_a_unary_op_with_no_inputs_is_a_named_refusal() -> void {
    let message = refusal_of(MODELS + "tf_missing_unary.pb")
    assert_true(message != "", "a Relu with no inputs was accepted")
    assert_true(message.contains("Relu"), f"the refusal does not name the operator: {message}")
    assert_true(
        not message.contains("index") and not message.contains("range"),
        f"the refusal looks like a raw index error rather than a named one: {message}"
    )
}

#: The same gap in a binary translator: `tf_missing_binary.pb` holds a
#: `Placeholder` and a `MatMul` reading only that one input, which used to
#: fail as "list index 1 out of range for length 1".
fn test_a_binary_op_with_one_input_is_a_named_refusal() -> void {
    let message = refusal_of(MODELS + "tf_missing_binary.pb")
    assert_true(message != "", "a MatMul with one input was accepted")
    assert_true(message.contains("MatMul"), f"the refusal does not name the operator: {message}")
    assert_true(
        not message.contains("index") and not message.contains("range"),
        f"the refusal looks like a raw index error rather than a named one: {message}"
    )
}

#: The `Const` pass wrapped decoding in a catch that dropped every failure
#: into "carries no value", when only a string constant is meant to be
#: dropped silently -- the doc comment right above it says why: a frozen
#: graph carries string constants for an assertion's message or a file name
#: that this importer never runs. `tf_complex_const.pb` holds a `DT_COMPLEX64`
#: constant read by a `Relu`, so decoding fails for a real reason -- this
#: importer has no tensor for a complex dtype -- and that reason has to
#: survive, not get folded into the string case's message.
fn test_a_const_the_reader_cannot_decode_keeps_its_reason() -> void {
    let message = refusal_of(MODELS + "tf_complex_const.pb")
    assert_true(message != "", "a Relu reading a complex64 constant was accepted")
    assert_true(message.contains("complex64"), f"the refusal does not say why: {message}")
    assert_true(
        not message.contains("carries no value"),
        f"the real reason was swallowed into the string constant's refusal: {message}"
    )
}
