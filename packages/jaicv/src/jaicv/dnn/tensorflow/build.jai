#: Which operator a node becomes, and the walk that turns a `GraphDef` into a `Graph`.
import std.io as io
from ..graph import Attr, Graph
from ..protobuf import field_by_number, fields_by_number, parse_message, parse_nested
from .graphdef import ATTR_TENSOR, GRAPH_NODE, TfNode, attr_float, attr_of
from .graphdef import attr_shape, attr_tensor, read_node
from .importer import Importer
from .ops import translate_arg_extreme, translate_batch_norm, translate_cast
from .ops import translate_concat, translate_conv, translate_elementwise
from .ops import translate_expand_dims, translate_lrn, translate_matmul
from .ops import translate_pad, translate_pool, translate_reduce, translate_reshape
from .ops import translate_softmax, translate_squeeze, translate_transpose
from .ops import translate_unary
from .tensors import DT_FLOAT, DT_STRING, TENSOR_DTYPE, TfTensor

#: Nodes that carry no value and are dropped rather than translated. `NoOp` and
#: `Assert` exist for their control edges, which this importer discards along
#: with every other control dependency.
let DISCARDED = ["NoOp", "Assert", "Print", "PrintV2", "SaveV2", "RestoreV2"]

#: TensorFlow operators that are the identity at inference time.
let PASSTHROUGH = ["Identity", "StopGradient", "CheckNumerics", "Snapshot", "PreventGradient"]

#: Unary TensorFlow operators whose ONNX name is all that changes.
let UNARY_OPS: dict[str, str] = {
    "Relu": "Relu",
    "Elu": "Elu",
    "Sigmoid": "Sigmoid",
    "Tanh": "Tanh",
    "Softplus": "Softplus",
    "Exp": "Exp",
    "Log": "Log",
    "Sqrt": "Sqrt",
    "Abs": "Abs",
    "Neg": "Neg",
    "Erf": "Erf",
    "LogicalNot": "Not",
}

#: Binary TensorFlow operators that are elementwise with numpy broadcasting.
let BINARY_OPS: dict[str, str] = {
    "Add": "Add",
    "AddV2": "Add",
    "BiasAdd": "Add",
    "BiasAddV1": "Add",
    "Sub": "Sub",
    "Mul": "Mul",
    "Div": "Div",
    "RealDiv": "Div",
    "Maximum": "Max",
    "Minimum": "Min",
    "Pow": "Pow",
    "Equal": "Equal",
    "Greater": "Greater",
    "Less": "Less",
    "LogicalAnd": "And",
    "LogicalOr": "Or",
}

#: The four reductions, with the ONNX name and the `row_reduce` they become.
let REDUCE_OPS: dict[str, str] = {
    "Mean": "ReduceMean",
    "Sum": "ReduceSum",
    "Max": "ReduceMax",
    "Min": "ReduceMin",
}

fn translate(im: Importer, node: TfNode) -> void {
    let op = node.op
    if op in PASSTHROUGH {
        if node.inputs.len() == 0 { throw ValueError(f"{node.label()} passes nothing through") }
        translate_unary(im, node, "Identity")
        return
    }
    if op == "Conv2D" {
        translate_conv(im, node, false)
        return
    }
    if op == "DepthwiseConv2dNative" {
        translate_conv(im, node, true)
        return
    }
    if op == "MaxPool" {
        translate_pool(im, node, true)
        return
    }
    if op == "AvgPool" {
        translate_pool(im, node, false)
        return
    }
    if op == "FusedBatchNorm" or op == "FusedBatchNormV2" or op == "FusedBatchNormV3" {
        translate_batch_norm(im, node)
        return
    }
    if op == "MatMul" {
        translate_matmul(im, node)
        return
    }
    if op == "Softmax" {
        translate_softmax(im, node, "Softmax")
        return
    }
    if op == "LogSoftmax" {
        translate_softmax(im, node, "LogSoftmax")
        return
    }
    if op == "LRN" {
        translate_lrn(im, node)
        return
    }
    if op == "Squeeze" {
        translate_squeeze(im, node)
        return
    }
    if op == "ExpandDims" {
        translate_expand_dims(im, node)
        return
    }
    if op == "Transpose" {
        translate_transpose(im, node)
        return
    }
    if op == "Reshape" {
        translate_reshape(im, node)
        return
    }
    if op == "ConcatV2" or op == "Concat" {
        translate_concat(im, node)
        return
    }
    if op == "Pad" or op == "PadV2" or op == "MirrorPad" {
        translate_pad(im, node)
        return
    }
    if op == "Cast" {
        translate_cast(im, node)
        return
    }
    if op == "ArgMax" {
        translate_arg_extreme(im, node, "ArgMax")
        return
    }
    if op == "ArgMin" {
        translate_arg_extreme(im, node, "ArgMin")
        return
    }
    let reduce = REDUCE_OPS.get(op, "")
    if reduce != "" {
        translate_reduce(im, node, reduce)
        return
    }
    if op == "Relu6" {
        #: `Relu6` is the clamp MobileNet is built out of, and ONNX has no
        #: operator for it beyond the general one.
        translate_unary(im, node, "Clip", {"min": Attr.of_float(0.0), "max": Attr.of_float(6.0)})
        return
    }
    if op == "LeakyRelu" {
        translate_unary(
            im,
            node,
            "LeakyRelu",
            {"alpha": Attr.of_float(attr_float(node, "alpha", 0.2))}
        )
        return
    }
    if op == "Square" or op == "Rsqrt" {
        if node.inputs.len() < 1 { throw ValueError(f"{node.label()} needs an input") }
        #: Squaring by multiplying would give the node one name in two input
        #: slots, and the executor frees an intermediate by the slot it was
        #: read in. An exponent costs one initializer and no such question.
        let exponent = if op == "Square" { 2.0 } else { -0.5 }
        let power = im.intern(f"{node.name}$exponent", [1], [exponent])
        let x = im.value_input(node.inputs[0], node)
        let _raised = im.emit("Pow", node.name, [x, power])
        im.ranks[node.name] = im.loose_rank(node.inputs[0])
        return
    }
    if op == "AddN" {
        translate_elementwise(im, node, "Sum")
        return
    }
    let unary = UNARY_OPS.get(op, "")
    if unary != "" {
        translate_unary(im, node, unary)
        return
    }
    let binary = BINARY_OPS.get(op, "")
    if binary != "" {
        translate_elementwise(im, node, binary)
        return
    }
    throw ValueError(f"{node.label()} is an operator this importer does not translate")
}

#: The nodes in an order where every name is written before it is read.
#:
#: A `GraphDef` usually already is in that order, but nothing in the format
#: says it has to be, and this importer needs the rank of a name before the
#: node that reads it. Kahn's, with the readers of each name indexed once so
#: the walk is linear in edges.
fn sorted_nodes(nodes: list[TfNode]) -> list[TfNode] {
    var index: dict[str, int] = {}
    for (slot, node) in nodes.enumerate() {
        if node.name in index { throw ValueError(f"the file has two nodes called {node.name}") }
        index[node.name] = slot
    }
    var waiting = [0 for _slot in 0..nodes.len()]
    var readers: dict[str, list[int]] = {}
    for (slot, node) in nodes.enumerate() {
        for name in node.inputs {
            let producer = index.get(name, -1)
            if producer < 0 {
                throw ValueError(f"{node.label()} reads {name}, which nothing in the file produces")
            }
            waiting[slot] += 1
            if readers.get(name, null) is null { readers[name] = [] }
            readers[name].push(slot)
        }
    }
    var ready: list[int] = []
    for (slot, count) in waiting.enumerate() {
        if count == 0 { ready.push(slot) }
    }
    var order: list[TfNode] = []
    var next = 0
    while next < ready.len() {
        let slot = ready[next]
        next += 1
        order.push(nodes[slot])
        for reader in readers.get(nodes[slot].name, []) {
            waiting[reader] -= 1
            if waiting[reader] == 0 { ready.push(reader) }
        }
    }
    if order.len() != nodes.len() {
        var stuck: list[str] = []
        for (slot, count) in waiting.enumerate() {
            if count > 0 { stuck.push(nodes[slot].name) }
        }
        throw ValueError(f"these nodes are in a cycle and cannot be ordered: {stuck}")
    }
    return order
}

#: Whether a `Const` declares `DT_STRING`, checked without decoding the rest
#: of the tensor so that string is the only reason one gets dropped silently.
#: Anything else that keeps a constant from decoding -- a data type this
#: importer has no tensor for, a declared shape its bytes cannot back -- has
#: to say so, not be folded into the same "carries no value" a dropped string
#: gets.
fn const_is_string(node: TfNode) -> bool {
    let found = attr_of(node, "value")
    if found is null { return false }
    let value = field_by_number(found, ATTR_TENSOR)
    if value is null { return false }
    let dtype_field = field_by_number(parse_nested(value), TENSOR_DTYPE)
    let dtype = if dtype_field is null { DT_FLOAT } else { dtype_field.varint }
    return dtype == DT_STRING
}

fn build_graph(nodes: list[TfNode]) -> Graph {
    let order = sorted_nodes(nodes)
    let im = Importer()
    for node in order {
        if node.op in DISCARDED {
            im.dropped[node.name] = true
            continue
        }
        if node.op != "Const" { continue }
        #: A frozen graph carries string constants for things this importer
        #: never runs -- an assertion's message, a file name. Refusing the
        #: whole file over one is wrong; refusing it only if something reads
        #: it is right.
        if const_is_string(node) {
            im.dropped[node.name] = true
            continue
        }
        let held = attr_tensor(node, "value")
        if held is null { throw ValueError(f"{node.label()} carries no value") }
        let tensor: TfTensor = held
        im.consts[node.name] = tensor
        im.ranks[node.name] = tensor.rank()
    }
    var outputs: list[str] = []
    try {
        for node in order {
            if node.op == "Const" or node.op in DISCARDED { continue }
            if node.op == "Placeholder" or node.op == "PlaceholderV2" {
                let dims = attr_shape(node, "shape")
                im.ranks[node.name] = if dims is null { -1 } else { dims.len() }
                im.graph_inputs.push(node.name)
                continue
            }
            translate(im, node)
        }
        var consumed: dict[str, bool] = {}
        for node in order {
            if node.op == "Const" or node.op in DISCARDED { continue }
            for name in node.inputs { consumed[name] = true }
        }
        #: What nothing else reads is what the graph is for. A frozen file
        #: names no outputs of its own, so this is the only thing to go on,
        #: and it is what every tool that reads one of these does.
        for node in order {
            if node.op in DISCARDED { continue }
            if node.op == "Const" or node.op == "Placeholder" or node.op == "PlaceholderV2" {
                continue
            }
            if not consumed.get(node.name, false) { outputs.push(node.name) }
        }
        if outputs.len() == 0 {
            throw ValueError(
                "every node in this file feeds another one, so it names no output to run it for"
            )
        }
    } catch failure: Error {
        im.discard()
        throw failure
    }
    let graph = Graph(
        im.produced,
        inputs: im.graph_inputs,
        outputs: outputs,
        initializers: im.weights
    )
    try {
        let _order = graph.topological()
    } catch failure: Error {
        im.discard()
        throw failure
    }
    return graph
}

#: What to say when a caller passes the text graph OpenCV takes beside the
#: binary one. Refusing is the honest answer: the configuration exists to
#: replace a subgraph with layers -- `PriorBox`, `DetectionOutput` -- that this
#: executor's operator set does not have, so a parser for it would buy nothing
#: but a later and more confusing failure.
let NO_CONFIG = "this importer reads the binary GraphDef alone; the text graph OpenCV takes beside it exists to substitute detection layers that this executor has no operators for"

#: Read a frozen TensorFlow `GraphDef` from `path`.
#:
#: The graph's inputs and outputs are `NCHW`, whatever the file's own layout
#: was, so a blob from `blob_from_image` feeds one directly. The names are the
#: file's own: a `Placeholder` keeps its name as a graph input, and every node
#: nothing else reads becomes a graph output under its own name.
pub fn read_net_from_tensorflow(path: str, config: str? = null) -> Graph {
    if config is not null { throw ValueError(NO_CONFIG) }
    return read_net_from_tensorflow_bytes(io.read_bytes(path))
}

#: The same, for a file already in memory.
pub fn read_net_from_tensorflow_bytes(data: bytes, config: bytes? = null) -> Graph {
    if config is not null { throw ValueError(NO_CONFIG) }
    var nodes: list[TfNode] = []
    for (slot, entry) in fields_by_number(parse_message(data), GRAPH_NODE).enumerate() {
        nodes.push(read_node(parse_nested(entry), slot))
    }
    if nodes.len() == 0 {
        throw ValueError("this file holds no nodes, so it is not a frozen GraphDef")
    }
    return build_graph(nodes)
}
