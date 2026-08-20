#: Reading Caffe models, against torch's answers and protobuf's own bytes.
#:
#: `tools/caffe_fixtures.py` writes every fixture in `tests/models/` through
#: the real protobuf package from upstream's caffe.proto, so a `.caffemodel`
#: here is a genuine `NetParameter` and a `.prototxt` is what protobuf's text
#: printer produces. The same script records, into `caffe_cases.txt`, the feeds
#: it made up, the raw contents of every weight blob it wrote, and a forward
#: pass computed by torch in float64.
#:
#: torch is the reference because it is the only thing to hand that reproduces
#: the two details of Caffe most easily got wrong: `max_pool2d` and
#: `avg_pool2d` round up, drop the last window when it would begin inside the
#: padding, and divide an average by the window clipped to the padded image --
#: which is Caffe exactly, and is none of ONNX's settings on their own. Both
#: shapes are in `opsnet` on purpose: a pool whose last window overhangs, and
#: one that ceil alone would give an extra column.
#:
#: The weights are checked separately from the arithmetic, because the classic
#: way to import a Caffe model wrongly is to run it beautifully on the wrong
#: numbers: `bnnet` carries a BatchNorm scale factor of 7.5, another of 3, and
#: a third of 0, and the test compares the imported statistics against the raw
#: blobs divided by each.
import std.io as io
from jaicv.dnn.caffe import free_caffe_weights, parse_net_text, read_net_from_caffe, read_net_from_caffe_bytes, translated_layers
from jaicv.dnn.graph import Graph, is_implemented, run_graph
from jaitensor import Tensor
from std.test import assert_contains, assert_eq, assert_near, assert_throws, assert_true

let MODELS = "packages/jaicv/tests/models/"

#: One tensor as the fixture recorded it.
class Recorded {
    pub let shape: list[int]
    pub let values: list[float]

    pub fn init(self, shape: list[int], values: list[float]) {
        self.shape = shape
        self.values = values
    }

    pub fn upload(self) -> Tensor { return Tensor(self.shape, self.values) }
}

#: One net: what to feed it, what its weights are, and what should come out.
class Case {
    pub var feeds: dict[str, Recorded]
    pub var blobs: dict[str, list[float]]
    pub var wants: dict[str, Recorded]
    pub var outputs: list[str]
    pub var order: list[str]

    pub fn init(self) {
        self.feeds = {}
        self.blobs = {}
        self.wants = {}
        self.outputs = []
        self.order = []
    }
}

fn words(line: str) -> list[str] {
    var out: list[str] = []
    for part in line.split(" ") {
        if part != "" { out.push(part) }
    }
    return out
}

fn whole(text: str) -> int { return text.to_int() ?? 0 }

fn number(text: str) -> float { return text.to_float() ?? 0.0 }

#: `<tag> <name> <rank> <dims...> <count> <values...>`.
fn read_tensor(parts: list[str]) -> Recorded {
    let rank = whole(parts[2])
    var shape: list[int] = []
    for axis in 0..rank { shape.push(whole(parts[3 + axis])) }
    let count = whole(parts[3 + rank])
    var values: list[float] = []
    for index in 0..count { values.push(number(parts[4 + rank + index])) }
    return Recorded(shape, values)
}

var _cases: dict[str, Case] = {}

fn cases() -> dict[str, Case] {
    if _cases.len() > 0 { return _cases }
    var current = Case()
    for line in io.read_file(MODELS + "caffe_cases.txt").split("\n") {
        let parts = words(line)
        if parts.len() == 0 { continue }
        if parts[0] == "net" {
            current = Case()
            _cases[parts[1]] = current
        } elif parts[0] == "feed" {
            current.feeds[parts[1]] = read_tensor(parts)
            current.order.push(parts[1])
        } elif parts[0] == "blob" {
            let count = whole(parts[3])
            var values: list[float] = []
            for index in 0..count { values.push(number(parts[4 + index])) }
            current.blobs[f"{parts[1]}#{parts[2]}"] = values
        } elif parts[0] == "want" {
            current.wants[parts[1]] = read_tensor(parts)
            current.outputs.push(parts[1])
        } else {
            throw ValueError(f"caffe_cases.txt has a line starting {parts[0]}")
        }
    }
    return _cases
}

fn case_of(name: str) -> Case {
    let found = cases().get(name, null)
    if found is null { throw ValueError(f"caffe_cases.txt has no net {name}") }
    let item: Case = found
    return item
}

fn open_net(name: str) -> Graph {
    return read_net_from_caffe(f"{MODELS}caffe_{name}.prototxt", f"{MODELS}caffe_{name}.caffemodel")
}

#: Compare against the reference, relative above one and absolute below it,
#: which is the same rule `test_dnn_graph.jai` uses.
fn agrees(actual: list[float], expected: list[float], tolerance: float, what: str) -> void {
    assert_eq(actual.len(), expected.len(), f"{what} has the wrong number of values")
    for index in 0..actual.len() {
        let want = expected[index]
        let magnitude = if want < 0.0 { -want } else { want }
        let allowed = if magnitude > 1.0 { tolerance * magnitude } else { tolerance }
        assert_near(actual[index], want, allowed, f"{what} at {index}")
    }
}

#: Run a whole net against the recorded forward pass.
fn check_forward(name: str, tolerance: float) -> void {
    let graph = open_net(name)
    let item = case_of(name)
    assert_eq(graph.inputs, item.order, f"{name} declares the wrong inputs")
    assert_eq(graph.outputs, item.outputs, f"{name} produces the wrong outputs")

    var feeds: dict[str, Tensor] = {}
    for (blob, recorded) in item.feeds.items() { feeds[blob] = recorded.upload() }
    let produced = run_graph(graph, feeds)
    for (blob, recorded) in item.wants.items() {
        let found = produced.get(blob, null)
        assert_true(found is not null, f"{name} produced nothing called {blob}")
        let tensor: Tensor = found
        assert_eq(tensor.shape, recorded.shape, f"{name} shaped {blob} wrongly")
        agrees(tensor.download(), recorded.values, tolerance, f"{name}'s {blob}")
    }
    for (_blob, tensor) in produced.items() { tensor.free() }
    for (_blob, tensor) in feeds.items() { tensor.free() }
    free_caffe_weights(graph)
}

#: A conv, an in-place ReLU, a max pool, an inner product and a softmax.
pub fn test_smallnet_forward() -> void { check_forward("smallnet", 2e-6) }

#: Every eltwise operation, including SUM with coefficients.
pub fn test_eltwise_forward() -> void { check_forward("eltnet", 2e-6) }

#: The rest of the layer table, with the two awkward pooling shapes.
pub fn test_opsnet_forward() -> void { check_forward("opsnet", 2e-6) }

#: BatchNorm with a scale factor of 7.5, of 3, and of 0.
pub fn test_batch_norm_forward() -> void { check_forward("bnnet", 2e-6) }

#: Global pooling, floor rounding, a Scale whose multiplier is another blob, a
#: transposed InnerProduct and one that lumps from an inner axis, the
#: deprecated `concat_dim`, a Flatten that stops short of the last axis, a
#: Reshape of part of a shape, and a PReLU with one shared slope.
pub fn test_extras_forward() -> void { check_forward("extranet", 2e-6) }

#: An in-place layer may not produce a name another node already produces, so
#: each rewrite is renamed -- but the last one keeps the blob's own name, which
#: is what makes `graph.outputs` readable.
pub fn test_in_place_layers_are_renamed() -> void {
    let graph = open_net("smallnet")
    var produced: dict[str, str] = {}
    for node in graph.nodes {
        for name in node.outputs {
            assert_true(not name in produced, f"{name} is produced twice")
            produced[name] = node.op
        }
    }
    assert_eq(produced.get("conv1:1", ""), "Conv", "the convolution should write the renamed blob")
    assert_eq(produced.get("conv1", ""), "Relu", "the in-place relu should keep the blob's name")
    assert_eq(graph.outputs, ["prob"], "smallnet ends at prob")
    let _order = graph.topological()
    free_caffe_weights(graph)
}

#: Caffe's BatchNorm blobs are the mean and the variance times the factor in
#: the third blob, and the layer divides by it on the way in. Getting this
#: wrong runs perfectly and answers wrongly, so it is checked against the raw
#: blobs rather than only through the forward pass.
pub fn test_batch_norm_scale_factor() -> void {
    let graph = open_net("bnnet")
    let item = case_of("bnnet")
    for (layer, factor) in [("bn1", 7.5), ("bn2", 3.0), ("bn3", 0.0)] {
        let reciprocal = if factor == 0.0 { 0.0 } else { 1.0 / factor }
        assert_eq(
            item.blobs.get(f"{layer}#2", [])[0],
            factor,
            f"{layer} should carry the factor {factor}"
        )
        for (index, part) in [(0, "mean"), (1, "variance")] {
            let raw = item.blobs.get(f"{layer}#{index}", [])
            let held = graph.initializers.get(f"{layer}/{part}", null)
            assert_true(held is not null, f"{layer} should hold a {part}")
            let tensor: Tensor = held
            agrees(
                tensor.download(),
                [value * reciprocal for value in raw],
                1e-6,
                f"{layer}'s {part}"
            )
        }
    }

    #: The Scale immediately after bn1 is folded in, so its blobs are the
    #: fused operator's scale and shift and there is no separate node.
    var kinds: list[str] = []
    for node in graph.nodes { kinds.push(node.op) }
    assert_eq(kinds, ["BatchNormalization", "Relu", "BatchNormalization", "BatchNormalization"])
    agrees(
        graph.initializers.get("scale1/scale", Tensor([1])).download(),
        item.blobs.get("scale1#0", []),
        1e-6,
        "the folded scale"
    )
    agrees(
        graph.initializers.get("scale1/bias", Tensor([1])).download(),
        item.blobs.get("scale1#1", []),
        1e-6,
        "the folded shift"
    )

    #: A BatchNorm with nothing folded into it still needs the affine part
    #: spelled out, because ONNX's operator has no default for it.
    agrees(
        graph.initializers.get("bn2/scale", Tensor([1])).download(),
        [1.0, 1.0, 1.0],
        1e-6,
        "an unfused BatchNorm's scale"
    )
    agrees(
        graph.initializers.get("bn2/bias", Tensor([1])).download(),
        [0.0, 0.0, 0.0],
        1e-6,
        "an unfused BatchNorm's shift"
    )
    free_caffe_weights(graph)
}

#: Every weight the importer holds is the blob the caffemodel carried, in the
#: shape the operator needs rather than the one the file happened to declare --
#: `ip1`'s blobs are in the deprecated four-number form.
pub fn test_weights_match_the_blobs() -> void {
    let graph = open_net("smallnet")
    let item = case_of("smallnet")
    for (name, layer) in [
        ("conv1/weight", "conv1#0"),
        ("conv1/bias", "conv1#1"),
        ("ip1/weight", "ip1#0"),
        ("ip1/bias", "ip1#1"),
    ] {
        let held = graph.initializers.get(name, null)
        assert_true(held is not null, f"{name} is missing")
        let tensor: Tensor = held
        agrees(tensor.download(), item.blobs.get(layer, []), 0.0, name)
    }
    assert_eq(graph.initializers.get("conv1/weight", Tensor([1])).shape, [4, 3, 3, 3])
    assert_eq(graph.initializers.get("ip1/weight", Tensor([1])).shape, [5, 64])
    assert_eq(graph.initializers.get("ip1/bias", Tensor([1])).shape, [5])
    free_caffe_weights(graph)
}

#: A prototxt on its own is a complete architecture with no numbers in it. The
#: graph keeps every node and the weights it could not find become inputs, so
#: the caller can see what the file needs and supply it.
pub fn test_prototxt_without_a_caffemodel() -> void {
    let bare = read_net_from_caffe(f"{MODELS}caffe_smallnet.prototxt")
    let full = open_net("smallnet")
    assert_eq(bare.initializers.len(), 0, "a prototxt carries no weights")
    assert_eq(
        bare.inputs,
        ["data", "conv1/weight", "conv1/bias", "ip1/weight", "ip1/bias"],
        "the missing weights should be named as inputs"
    )
    assert_eq(bare.outputs, full.outputs, "the two readings should end in the same place")
    assert_eq(bare.nodes.len(), full.nodes.len(), "the two readings should have the same nodes")
    for (index, node) in bare.nodes.enumerate() {
        assert_eq(node.op, full.nodes[index].op)
        assert_eq(node.inputs, full.nodes[index].inputs)
        assert_eq(node.outputs, full.nodes[index].outputs)
    }
    free_caffe_weights(bare)
    free_caffe_weights(full)
}

#: A `Scale` is folded into the `BatchNorm` in front of it only when nothing
#: else can see the value in between. Folding one that a later layer reads
#: past would silently apply the scale to a reader that Caffe does not.
pub fn test_when_a_scale_is_folded() -> void {
    let head = "input: \"data\"\ninput_shape { dim: 1 dim: 3 dim: 2 dim: 2 }\n"
    let norm = "layer { name: \"bn\" type: \"BatchNorm\" bottom: \"data\" top: \"bn\" }\n"

    #: In place, which is how a deploy prototxt writes the pair: everything
    #: below reads the same blob name and so reads the scaled value anyway.
    let inline = parse_and_build(
        head +
            norm +
            "layer { name: \"sc\" type: \"Scale\" bottom: \"bn\" top: \"bn\" }\n" +
            "layer { name: \"r\" type: \"ReLU\" bottom: \"bn\" top: \"r\" }\n"
    )
    assert_eq(ops_of(inline), ["BatchNormalization", "Relu"], "an in-place Scale folds")
    assert_eq(inline.nodes[0].inputs[1], "sc/scale", "and brings its blob with it")

    #: Renaming, with nothing below reading what it consumed.
    let renamed = parse_and_build(
        head +
            norm +
            "layer { name: \"sc\" type: \"Scale\" bottom: \"bn\" top: \"sc\" }\n"
    )
    assert_eq(ops_of(renamed), ["BatchNormalization"], "a dead intermediate folds too")
    assert_eq(renamed.outputs, ["sc"])

    #: Renaming, with a later layer reading what the fold would have swallowed.
    let watched = parse_and_build(
        head +
            norm +
            "layer { name: \"sc\" type: \"Scale\" bottom: \"bn\" top: \"sc\" }\n" +
            "layer { name: \"r\" type: \"ReLU\" bottom: \"bn\" top: \"r\" }\n"
    )
    assert_eq(
        ops_of(watched),
        ["BatchNormalization", "Mul", "Relu"],
        "a Scale whose input is read past it stays its own operator"
    )

    #: A Scale over some other axis is not the affine part of a BatchNorm.
    let sideways = parse_and_build(
        head +
            norm +
            "layer { name: \"sc\" type: \"Scale\" bottom: \"bn\" top: \"bn\" " +
            "scale_param { axis: 0 num_axes: 2 } }\n"
    )
    assert_eq(ops_of(sideways), ["BatchNormalization", "Mul"], "only the channel axis folds")
    for graph in [inline, renamed, watched, sideways] { free_caffe_weights(graph) }
}

fn ops_of(graph: Graph) -> list[str] {
    var out: list[str] = []
    for node in graph.nodes { out.push(node.op) }
    return out
}

#: Every net reads the same either way round, so nothing in the translation
#: depends on the weights having turned up.
pub fn test_prototxt_alone_matches_the_pair() -> void {
    for name in ["smallnet", "bnnet", "eltnet", "opsnet", "extranet"] {
        let bare = read_net_from_caffe(f"{MODELS}caffe_{name}.prototxt")
        let full = open_net(name)
        assert_eq(bare.outputs, full.outputs, f"{name} should end in the same place")
        assert_eq(bare.nodes.len(), full.nodes.len(), f"{name} should have the same nodes")
        for (index, node) in bare.nodes.enumerate() {
            assert_eq(node.op, full.nodes[index].op, f"{name} node {index}")
            assert_eq(node.inputs, full.nodes[index].inputs, f"{name} node {index}")
            assert_eq(node.outputs, full.nodes[index].outputs, f"{name} node {index}")
        }

        #: A weight the pair holds is either a blob the caffemodel carried --
        #: which the prototxt alone asks to be fed instead -- or a constant the
        #: importer invented, which is there either way.
        var wanted: list[str] = []
        for asked in bare.inputs {
            if not asked in full.inputs { wanted.push(asked) }
        }
        for (invented, _tensor) in bare.initializers.items() { wanted.push(invented) }
        var held: list[str] = []
        for (weight, _tensor) in full.initializers.items() { held.push(weight) }
        wanted.sort()
        held.sort()
        assert_eq(wanted, held, f"{name}'s weights should be named either way")
        free_caffe_weights(bare)
        free_caffe_weights(full)
    }
}

#: Reading from memory is the same reading.
pub fn test_reading_from_bytes() -> void {
    let graph = read_net_from_caffe_bytes(
        io.read_bytes(f"{MODELS}caffe_smallnet.prototxt"),
        io.read_bytes(f"{MODELS}caffe_smallnet.caffemodel")
    )
    let item = case_of("smallnet")
    assert_eq(graph.outputs, ["prob"])
    agrees(
        graph.initializers.get("conv1/weight", Tensor([1])).download(),
        item.blobs.get("conv1#0", []),
        0.0,
        "conv1's weights"
    )
    free_caffe_weights(graph)
}

#: The three ways a net can say what it is fed: `input_shape`, the older
#: `input_dim`, and an `Input` layer.
pub fn test_every_input_declaration() -> void {
    for (net, blob, shape) in [
        ("smallnet", "data", [2, 3, 8, 8]),
        ("bnnet", "data", [2, 3, 4, 4]),
        ("eltnet", "a", [2, 2, 3, 3]),
    ] {
        let graph = read_net_from_caffe(f"{MODELS}caffe_{net}.prototxt")
        assert_contains(graph.inputs, blob, f"{net} should declare {blob}")

        #: The shape is not on the `Graph`, so it is checked by what depends
        #: on it: a Scale or a pooled extent is only computable with it.
        let item = case_of(net)
        assert_eq(item.feeds.get(blob, Recorded([], [])).shape, shape)
        free_caffe_weights(graph)
    }
}

#: Caffe rounds a pooled extent up and then, if there is padding, drops a
#: window that would begin inside it. `opsnet` has one of each: `pavg` keeps
#: its four columns, and `pmax` would have three under ceil alone and has two.
pub fn test_caffe_pooling_extents() -> void {
    let graph = open_net("opsnet")
    let item = case_of("opsnet")
    assert_eq(item.wants.get("cut_b", Recorded([], [])).shape, [2, 4, 4, 4], "pavg pools 6 to 4")
    assert_eq(item.wants.get("pre", Recorded([], [])).shape, [2, 6, 4, 4])
    var found = false
    for node in graph.nodes {
        if node.name != "pmax" { continue }
        found = true
        assert_eq(node.op, "MaxPool")

        #: The extent is handed over as explicit padding rather than as
        #: `ceil_mode`, which cannot express the dropped window.
        assert_eq(node.attrs.get("pads", node.attrs["kernel_shape"]).ints, [1, 1, 0, 0])
        assert_true(not "ceil_mode" in node.attrs, "an inferred extent needs no ceil_mode")
    }
    assert_true(found, "opsnet should have a layer called pmax")
    free_caffe_weights(graph)
}

#: An average pool over padding divides by the window clipped to the padded
#: image, so the padding is materialised and the pool counts only what is in
#: it. Neither of ONNX's `count_include_pad` settings says that on its own.
pub fn test_average_pool_pads_explicitly() -> void {
    let graph = open_net("opsnet")
    var padded = false
    for node in graph.nodes {
        if node.op == "Pad" and node.inputs[0] == "lrn1" { padded = true }
        if node.name != "pavg" { continue }
        assert_eq(node.op, "AveragePool")
        assert_eq(node.attrs["count_include_pad"].i, 0)
        assert_eq(node.attrs["pads"].ints, [0, 0, 1, 1], "one row past the padding, both axes")
    }
    assert_true(padded, "the padding should be materialised before the pool")
    free_caffe_weights(graph)
}

#: A prototxt is a hand-written file and arrives looking like one. Everything
#: in `caffe_awkward.prototxt` is accepted by protobuf's own text parser --
#: the fixture script writes the file only after parsing it.
pub fn test_awkward_prototxt() -> void {
    let net = parse_net_text(io.read_file(f"{MODELS}caffe_awkward.prototxt"))
    assert_eq(net.str_of("name", ""), "awkward", "adjacent literals join")
    assert_eq(net.strs_of("input"), ["data"])

    #: One shape in angle brackets, one in braces holding a bracketed list.
    let shapes = net.children_of("input_shape")
    assert_eq(shapes.len(), 2)
    assert_eq(shapes[0].ints_of("dim"), [1, 3, 4, 4])
    assert_eq(shapes[1].ints_of("dim"), [1, 3, 4, 4])

    let layers = net.children_of("layer")
    assert_eq(layers.len(), 3)
    assert_eq(
        layers[0].str_of("name", ""),
        "od\td\"d\\AA",
        "a tab, a quote, a backslash, a hex escape and an octal one"
    )
    assert_eq(layers[0].str_of("type", ""), "Convolution", "single quotes are quotes")
    assert_eq(layers[0].strs_of("bottom"), ["data"])
    assert_eq(layers[0].strs_of("top"), ["conv"])
    let convolution = layers[0].param("convolution_param", "ConvolutionParameter")
    assert_eq(convolution.int_of("num_output", 0), 16, "0x10 is sixteen")
    assert_eq(convolution.ints_of("kernel_size"), [3])
    assert_eq(convolution.ints_of("pad"), [1])
    assert_true(not convolution.has("weight_filler"), "an unmodelled field is skipped whole")
    assert_true(not layers[0].has("param"), "so is a repeated one")

    let pooling = layers[1].param("pooling_param", "PoolingParameter")
    assert_eq(pooling.int_of("pool", -1), 0, "MAX is the bare enum for zero")
    assert_eq(pooling.int_of("round_mode", -1), 1, "FLOOR is one")
    assert_eq(pooling.int_of("kernel_size", 0), 2)
    assert_eq(pooling.int_of("stride", 0), 2)

    assert_eq(layers[2].strs_of("bottom"), ["pool", "pool"], "a repeated field, twice")
    let eltwise = layers[2].param("eltwise_param", "EltwiseParameter")
    assert_eq(eltwise.int_of("operation", -1), 0)
    assert_eq(eltwise.floats_of("coeff"), [1.0, -1.0])
}

#: The scalar forms the text format allows, each read into a field whose type
#: is what decides how the token is taken.
pub fn test_text_scalar_forms() -> void {
    let net = parse_net_text(
        "layer { name: \"x\" type: \"Convolution\"\n" +
            "  convolution_param { num_output: 010 group: 0x1f bias_term: False }\n" +
            "  relu_param { negative_slope: -1.5e-2 }\n" +
            "  lrn_param { alpha: .5 beta: 2. k: 3f }\n" +
            "  power_param { power: inf scale: -inf }\n" +
            "}"
    )
    let layer = net.children_of("layer")[0]
    let convolution = layer.param("convolution_param", "ConvolutionParameter")
    assert_eq(convolution.int_of("num_output", 0), 8, "a leading zero is octal")
    assert_eq(convolution.int_of("group", 0), 31, "0x1f is thirty-one")
    assert_eq(convolution.bool_of("bias_term", true), false, "False is false")
    assert_near(
        layer.param("relu_param", "ReLUParameter").float_of("negative_slope", 0.0),
        -0.015,
        1e-12
    )
    let lrn = layer.param("lrn_param", "LRNParameter")
    assert_near(lrn.float_of("alpha", 0.0), 0.5, 1e-12)
    assert_near(lrn.float_of("beta", 0.0), 2.0, 1e-12)
    assert_near(lrn.float_of("k", 0.0), 3.0, 1e-12, "a trailing f is a float suffix")
    let power = layer.param("power_param", "PowerParameter")
    assert_true(power.float_of("power", 0.0) > 1e300, "inf is infinity")
    assert_true(power.float_of("scale", 0.0) < -1e300, "and it takes a sign")
}

#: A prototxt that is not one has to say where it stopped making sense.
pub fn test_text_parser_refusals() -> void {
    for (text, fragment) in [
        ("layer { name: \"x\"", "no closing"),
        ("name \"x\"", "needs a ':'"),
        ("name: unquoted", "not quoted"),
        ("name: \"x\n\"", "past the end of its line"),
        ("layer { name: \"x\" type: \"ReLU\" > ", "where } was expected"),
        ("input_shape { dim: 1 dim: two }", "not a whole number"),
        ("layer { name: \"x\" pooling_param { pool: SIDEWAYS } }", "not one of PoolMethod"),
        ("layer { name: \"x\" phase: MAYBE }", "not one of Phase"),
        ("name: \"\\q\"", "is not an escape"),
        ("[caffe.thing]: 1", "extension fields"),
    ] {
        let failure = assert_throws(ValueError, || parse_net_text(text))
        assert_contains(failure.message, fragment, f"reading {text}")
    }
}

#: A layer type this importer does not translate has to name itself rather
#: than produce a graph that is quietly missing a step.
pub fn test_unsupported_layer_type() -> void {
    let text = "input: \"data\"\nlayer { name: \"acc\" type: \"Accuracy\" bottom: \"data\" " +
        "top: \"acc\" }\n"
    let failure = assert_throws(ValueError, || parse_and_build(text))
    assert_contains(failure.message, "Accuracy")
    assert_contains(failure.message, "does not translate")
}

fn parse_and_build(text: str) -> Graph {
    return read_net_from_caffe_bytes(bytes(text.bytes()))
}

#: The pre-2015 `layers` field is a different message with different field
#: numbers, and reading it as the modern one would produce nonsense.
pub fn test_v1_layers_are_refused() -> void {
    let failure = assert_throws(ValueError, || parse_and_build("layers { name: \"x\" type: RELU }"))
    assert_contains(failure.message, "V1")
}

#: A blob nothing produces is an error naming the layer that wanted it.
pub fn test_missing_bottom() -> void {
    let failure = assert_throws(
        ValueError,
        || parse_and_build("layer { name: \"r\" type: \"ReLU\" bottom: \"ghost\" top: \"r\" }")
    )
    assert_contains(failure.message, "ghost")
}

#: The layer types this importer translates are the ones it says it does, and
#: every one of them lands on an operator the shared executor implements.
pub fn test_translated_layers() -> void {
    let named = translated_layers()
    assert_eq(named.len(), 24, "the table should hold every type this module maps")
    for kind in ["Convolution", "Deconvolution", "Pooling", "InnerProduct", "BatchNorm", "Scale"] {
        assert_contains(named, kind)
    }
    assert_true(not "Accuracy" in named, "a loss layer is not translated")
    var seen: dict[str, bool] = {}
    for name in ["smallnet", "bnnet", "eltnet", "opsnet", "extranet"] {
        let graph = open_net(name)
        for node in graph.nodes {
            assert_true(
                is_implemented(node.op),
                f"{node.op} should be an operator the executor has"
            )
            seen[node.op] = true
        }
        free_caffe_weights(graph)
    }
    assert_true(seen.len() >= 20, "the fixtures should reach most of the operator set")
}

#: A layer this importer reads but cannot honour has to say which one it is
#: and what about it it cannot do. Each of these is a real Caffe file that
#: means something this translation does not.
pub fn test_layers_it_cannot_honour() -> void {
    let head = "input: \"data\"\ninput_shape { dim: 1 dim: 4 dim: 8 dim: 8 }\n"
    for (body, fragment) in [
        (
            "layer { name: \"n\" type: \"BatchNorm\" bottom: \"data\" top: \"n\" " +
                "batch_norm_param { use_global_stats: false } }",
            "training and not inference"
        ),
        (
            "layer { name: \"n\" type: \"LRN\" bottom: \"data\" top: \"n\" " +
                "lrn_param { norm_region: WITHIN_CHANNEL } }",
            "across-channel"
        ),
        (
            "layer { name: \"n\" type: \"Pooling\" bottom: \"data\" top: \"n\" " +
                "pooling_param { pool: STOCHASTIC kernel_size: 2 } }",
            "stochastically"
        ),
        (
            "layer { name: \"n\" type: \"Convolution\" bottom: \"data\" top: \"n\" " +
                "convolution_param { num_output: 2 kernel_size: 3 axis: 0 } }",
            "usual channel axis"
        ),
        (
            "layer { name: \"n\" type: \"Convolution\" bottom: \"data\" top: \"n\" " +
                "convolution_param { num_output: 0 kernel_size: 3 } }",
            "num_output 0"
        ),
    ] {
        let failure = assert_throws(ValueError, || parse_and_build(head + body))
        assert_contains(failure.message, fragment, f"reading {body}")
        assert_contains(failure.message, "layer n", "the layer should name itself")
    }
}

#: A prototxt may carry its own blobs, and the counts have to add up whichever
#: file they came out of.
pub fn test_blob_counts_must_add_up() -> void {
    let head = "input: \"data\"\ninput_shape { dim: 1 dim: 4 dim: 8 dim: 8 }\n"
    let missing = "layer { name: \"c\" type: \"Convolution\" bottom: \"data\" top: \"c\" " +
        "blobs { shape { dim: 2 dim: 4 dim: 1 dim: 1 } data: 1 data: 2 data: 3 data: 4 " +
        "data: 5 data: 6 data: 7 data: 8 } " +
        "convolution_param { num_output: 2 kernel_size: 1 } }"
    let short = assert_throws(ValueError, || parse_and_build(head + missing))
    assert_contains(short.message, "carries 1 blobs", "a biased convolution has two")

    let ragged = "layer { name: \"c\" type: \"Convolution\" bottom: \"data\" top: \"c\" " +
        "blobs { shape { dim: 5 } data: 1 data: 2 data: 3 data: 4 data: 5 } " +
        "convolution_param { num_output: 2 kernel_size: 1 bias_term: false } }"
    let odd = assert_throws(ValueError, || parse_and_build(head + ragged))
    assert_contains(odd.message, "whole number of 1x1 filters")

    let lying = "layer { name: \"c\" type: \"Convolution\" bottom: \"data\" top: \"c\" " +
        "blobs { shape { dim: 3 } data: 1 data: 2 } " +
        "convolution_param { num_output: 2 kernel_size: 1 bias_term: false } }"
    let wrong = assert_throws(ValueError, || parse_and_build(head + lying))
    assert_contains(wrong.message, "carries 2 numbers", "a blob has to mean its own shape")
}

#: A Scale broadcasts over axes this importer has to be able to count, so a
#: net that never says how big its input is has to be told why it stopped.
pub fn test_a_shape_it_cannot_infer() -> void {
    let text = "input: \"data\"\nlayer { name: \"s\" type: \"Scale\" bottom: \"data\" " +
        "top: \"s\" }"
    let failure = assert_throws(ValueError, || parse_and_build(text))
    assert_contains(failure.message, "needs the shape of data")
}

#: A caffemodel whose layer names do not match the prototxt leaves the
#: prototxt's layers as they were, which is Caffe's own behaviour and not an
#: error -- but a caffemodel that is not protobuf at all is.
pub fn test_unreadable_caffemodel() -> void {
    let text = io.read_bytes(f"{MODELS}caffe_smallnet.prototxt")
    let rubbish = bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF])
    let _failure = assert_throws(ValueError, || read_net_from_caffe_bytes(text, rubbish))
}
