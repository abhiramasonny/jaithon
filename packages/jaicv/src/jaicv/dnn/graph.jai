#: The graph a network becomes once it has been imported, and the executor
#: that runs it.
#:
#: Every importer -- ONNX, Caffe, TensorFlow, Torch -- lands here, and they all
#: land in the same vocabulary: the `op` string on a `GraphNode` is an ONNX
#: operator name. There is one executor because there is one vocabulary, so a
#: Caffe `Convolution` and a TensorFlow `Conv2D` both arrive as `Conv` and
#: neither importer carries its own arithmetic.
#:
#: Tensors are jaitensor's, so they live on the GPU and hold float32 and
#: nothing else. Shapes, indices and booleans therefore ride in float32 too --
#: exact up to 2^24, which is every index a vision network ever produces. The
#: layout is ONNX's throughout: `NCHW` for images, `MCkHkW` for convolution
#: weights. jaitensor's convolution and pooling are channel-last, so this
#: module transposes around them rather than the other way round; the transpose
#: is a kernel and the convolution is not, which is the right way for that cost
#: to fall.
#:
#: Ownership is simple and worth stating because it is what stops a graph from
#: freeing a tensor it still needs. Every value a node produces is a fresh
#: tensor this module owns. Initializers and the caller's feeds are never
#: freed, and never handed back as an output without being copied first. An
#: intermediate is freed as soon as the last node that reads it has run.
#:
#: Two shapes ONNX has and jaitensor does not, which an importer has to know
#: about because it is the one holding the file:
#:
#: - There is no rank-zero tensor, so a scalar arrives and leaves as `[1]`.
#:   Broadcasting treats the two the same; what differs is that an operator
#:   which would have dropped an axis (a `Gather` by a scalar index) leaves a
#:   width of one behind instead.
#: - There is no empty tensor, so an optional input that a file skips must be
#:   passed as an empty *name* in `GraphNode.inputs` -- position is what
#:   identifies an argument -- and never as a zero-length tensor.
#:
#: ```
#: let graph = load_onnx("mobilenet.onnx")
#: let outputs = run_graph(graph, {"input": blob})
#: defer { for (_name, tensor) in outputs.items() { tensor.free() } }
#: ```
from jaitensor import Tensor
import std.gpu as gpu
from std.os import env
from std.time import Instant

from .graph.nodes import Attr, Graph, GraphNode, attr_float, attr_int, attr_ints, attr_str
from .graph.ops_layers import op_average_pool, op_batch_norm, op_conv, op_conv_transpose
from .graph.ops_layers import op_global_average_pool, op_global_max_pool, op_instance_norm
from .graph.ops_layers import op_layer_norm, op_lrn, op_max_pool, op_resize, op_upsample
from .graph.ops_math import op_abs, op_add, op_and, op_argmax, op_argmin, op_clip, op_div
from .graph.ops_math import op_dropout, op_elu, op_equal, op_erf, op_exp, op_gemm, op_greater
from .graph.ops_math import op_hard_sigmoid, op_identity, op_leaky_relu, op_less, op_log
from .graph.ops_math import op_log_softmax, op_matmul, op_max, op_mean, op_min, op_mod, op_mul
from .graph.ops_math import op_neg, op_not, op_or, op_pow, op_prelu, op_reduce_max
from .graph.ops_math import op_reduce_mean, op_reduce_min, op_reduce_sum, op_relu, op_sigmoid
from .graph.ops_math import op_silu, op_softmax, op_softplus, op_sqrt, op_sub, op_sum, op_tanh
from .graph.ops_math import op_where
from .graph.ops_rnn import op_gru, op_lstm
from .graph.ops_shape import op_cast, op_concat, op_constant, op_constant_of_shape, op_flatten
from .graph.ops_shape import op_gather, op_pad, op_reshape, op_shape, op_slice, op_split
from .graph.ops_shape import op_squeeze, op_transpose, op_unsqueeze
from .graph.tensors import copy_of, normalize_axis

var _handlers: dict[str, fn(GraphNode, list[Tensor?]) -> list[Tensor]] = {}

#: Every operator the executor knows, keyed by its ONNX name.
#:
#: Built once on first use rather than at module load, because a top-level
#: dict of function values would have to name each one before it is defined.
fn handlers() -> dict[str, fn(GraphNode, list[Tensor?]) -> list[Tensor]] {
    if _handlers.len() > 0 { return _handlers }
    _handlers["Conv"] = op_conv
    _handlers["ConvTranspose"] = op_conv_transpose
    _handlers["Gemm"] = op_gemm
    _handlers["MatMul"] = op_matmul
    _handlers["Relu"] = op_relu
    _handlers["LeakyRelu"] = op_leaky_relu
    _handlers["PRelu"] = op_prelu
    _handlers["Elu"] = op_elu
    _handlers["Sigmoid"] = op_sigmoid
    _handlers["Silu"] = op_silu
    _handlers["HardSigmoid"] = op_hard_sigmoid
    _handlers["Tanh"] = op_tanh
    _handlers["Softplus"] = op_softplus
    _handlers["Erf"] = op_erf
    _handlers["Softmax"] = op_softmax
    _handlers["LogSoftmax"] = op_log_softmax
    _handlers["Clip"] = op_clip
    _handlers["Add"] = op_add
    _handlers["Sub"] = op_sub
    _handlers["Mul"] = op_mul
    _handlers["Div"] = op_div
    _handlers["Mod"] = op_mod
    _handlers["Pow"] = op_pow
    _handlers["Sqrt"] = op_sqrt
    _handlers["Exp"] = op_exp
    _handlers["Log"] = op_log
    _handlers["Abs"] = op_abs
    _handlers["Neg"] = op_neg
    _handlers["Min"] = op_min
    _handlers["Max"] = op_max
    _handlers["Sum"] = op_sum
    _handlers["Mean"] = op_mean
    _handlers["Concat"] = op_concat
    _handlers["Split"] = op_split
    _handlers["Reshape"] = op_reshape
    _handlers["Flatten"] = op_flatten
    _handlers["Squeeze"] = op_squeeze
    _handlers["Unsqueeze"] = op_unsqueeze
    _handlers["Transpose"] = op_transpose
    _handlers["Slice"] = op_slice
    _handlers["Gather"] = op_gather
    _handlers["Pad"] = op_pad
    _handlers["MaxPool"] = op_max_pool
    _handlers["AveragePool"] = op_average_pool
    _handlers["GlobalAveragePool"] = op_global_average_pool
    _handlers["GlobalMaxPool"] = op_global_max_pool
    _handlers["BatchNormalization"] = op_batch_norm
    _handlers["InstanceNormalization"] = op_instance_norm
    _handlers["LayerNormalization"] = op_layer_norm
    _handlers["GRU"] = op_gru
    _handlers["LSTM"] = op_lstm
    _handlers["LRN"] = op_lrn
    _handlers["Dropout"] = op_dropout
    _handlers["Identity"] = op_identity
    _handlers["Constant"] = op_constant
    _handlers["ConstantOfShape"] = op_constant_of_shape
    _handlers["Shape"] = op_shape
    _handlers["Cast"] = op_cast
    _handlers["ReduceMean"] = op_reduce_mean
    _handlers["ReduceSum"] = op_reduce_sum
    _handlers["ReduceMax"] = op_reduce_max
    _handlers["ReduceMin"] = op_reduce_min
    _handlers["ArgMax"] = op_argmax
    _handlers["ArgMin"] = op_argmin
    _handlers["Resize"] = op_resize
    _handlers["Upsample"] = op_upsample
    _handlers["Where"] = op_where
    _handlers["Equal"] = op_equal
    _handlers["Greater"] = op_greater
    _handlers["Less"] = op_less
    _handlers["Not"] = op_not
    _handlers["And"] = op_and
    _handlers["Or"] = op_or
    return _handlers
}

#: Whether `run_graph` can execute this ONNX operator. An importer can ask
#: before it builds a node, and say what it could not translate up front
#: rather than letting the executor discover it halfway through a forward
#: pass.
pub fn is_implemented(op: str) -> bool { return op in handlers() }

#: Every operator name `run_graph` understands, in alphabetical order.
pub fn implemented_operators() -> list[str] {
    var names: list[str] = []
    for (name, _handler) in handlers().items() { names.push(name) }
    names.sort()
    return names
}

#: Run `graph` on `feeds` and return one tensor for each name in
#: `graph.outputs`.
#:
#: The caller owns every tensor that comes back and should free them. Nothing
#: passed in is freed or modified: an output that is only an alias of a feed or
#: an initializer is copied so that freeing the results cannot reach back into
#: the caller's weights.
#:
#: Raises `ValueError` naming the node for an operator that is not
#: implemented, for a missing feed, and for every shape the operators
#: themselves refuse.
pub fn run_graph(
    graph: Graph,
    feeds: dict[str, Tensor],
    shapes: dict[str, list[int]]? = null,
    capture: dict[str, list[float]]? = null
) -> dict[str, Tensor] {
    #: `shapes` and `capture`, when given, are filled with every value's shape
    #: and with the values of the names already present in `capture`. That is
    #: what the plan compiler needs to know before it can build a graph: what
    #: shape everything is, and what the shape arithmetic worked out to. It
    #: costs two dictionary writes a node and is null on every other path.
    if graph.outputs.len() == 0 {
        throw ValueError("this graph names no outputs, so there is nothing to run it for")
    }
    let order = graph.schedule()
    var values: dict[str, Tensor] = {}
    var mine: dict[str, bool] = {}
    for (name, weight) in graph.initializers.items() { values[name] = weight }
    for (name, fed) in feeds.items() { values[name] = fed }
    if shapes is not null {
        for (name, held) in values.items() { shapes[name] = held.shape }
    }
    for name in graph.inputs {
        if values.get(name, null) is null {
            throw ValueError(f"the graph input {name} was not fed")
        }
    }

    #: The index of the last node that reads each name, which is when the
    #: value behind it stops being needed.
    var last_read: dict[str, int] = {}
    for (index, node) in order.enumerate() {
        for name in node.inputs {
            if name.len() > 0 { last_read[name] = index }
        }
    }

    let table = handlers()
    let profiling = env("JAICV_GRAPH_PROFILE") is not null
    var op_ms: dict[str, float] = {}
    var op_count: dict[str, int] = {}
    var produced_slot: list[Tensor] = []
    try {
        for (index, node) in order.enumerate() {
            var ins: list[Tensor?] = []
            for name in node.inputs {
                if name.len() == 0 {
                    ins.push(null)
                    continue
                }
                let found = values.get(name, null)
                if found is null {
                    throw ValueError(f"{node.label()} reads {name}, which has no value")
                }
                ins.push(found)
            }
            #: Whose the input was *before* this node ran: a node may write a
            #: name it also reads, and freeing on the state afterwards would
            #: release the caller's initializer or the fresh output instead of
            #: the value that has just been consumed.
            var owned_input: list[bool] = []
            for name in node.inputs {
                owned_input.push(name.len() > 0 and mine.get(name, false))
            }
            let handler = table.get(node.op, null)
            if handler is null { throw ValueError(f"{node.label()} is not implemented") }
            #: `JAICV_GRAPH_PROFILE=1` totals the wall clock per operator and
            #: prints the table when the graph finishes. Every op queues GPU
            #: work rather than waiting for it, so the figures only mean
            #: anything with a synchronise between them -- which is why the
            #: profile is off by default rather than merely unprinted.
            if profiling {
                gpu.synchronize()
                let started = Instant.now()
                let timed = handler(node, ins)
                gpu.synchronize()
                let spent_ms = started.elapsed().to_seconds() * 1000.0
                op_ms[node.op] = op_ms.get(node.op, 0.0) + spent_ms
                op_count[node.op] = op_count.get(node.op, 0) + 1
                produced_slot = timed
            } else {
                produced_slot = handler(node, ins)
            }
            let produced = produced_slot
            if produced.len() < node.outputs.len() {
                throw ValueError(
                    f"{node.label()} names {node.outputs.len()} outputs and produced " +
                        f"{produced.len()}"
                )
            }
            for (slot, tensor) in produced.enumerate() {
                let name = if slot < node.outputs.len() { node.outputs[slot] } else { "" }
                if name.len() == 0 {
                    tensor.free()
                    continue
                }
                values[name] = tensor
                mine[name] = true
                if shapes is not null { shapes[name] = tensor.shape }
                if capture is not null and capture.get(name, null) is not null {
                    capture[name] = tensor.download()
                }
            }
            for (slot, name) in node.inputs.enumerate() {
                if not owned_input[slot] or last_read.get(name, -1) != index { continue }
                if name in graph.outputs { continue }
                let spent = ins[slot]
                if spent is not null { spent.free() }
            }
        }
    } catch failure: Error {
        for (name, owned) in mine.items() {
            if not owned { continue }
            let stranded = values.get(name, null)
            if stranded is not null { stranded.free() }
        }
        throw failure
    }

    var results: dict[str, Tensor] = {}
    for name in graph.outputs {
        let value = values.get(name, null)
        if value is null { throw ValueError(f"the graph output {name} was never produced") }
        #: A feed or an initializer routed straight to an output is copied, so
        #: that freeing the results never frees the caller's own tensor.
        results[name] = if mine.get(name, false) { value } else { copy_of(value, value.shape) }
    }
    for (name, owned) in mine.items() {
        if owned and not name in graph.outputs {
            let leftover = values.get(name, null)
            if leftover is not null { leftover.free() }
        }
    }
    if profiling { report_profile(op_ms, op_count) }
    return results
}

#: The per-operator totals `JAICV_GRAPH_PROFILE` collected, slowest first.
fn report_profile(op_ms: dict[str, float], op_count: dict[str, int]) -> void {
    var names: list[str] = []
    for name in op_ms.keys() { names.push(name) }
    #: A plain insertion sort: this runs once at the end of a graph and there
    #: are a few dozen distinct operators at most.
    for outer in 1..names.len() {
        let held = names[outer]
        var inner = outer - 1
        while inner >= 0 and op_ms[names[inner]] < op_ms[held] {
            names[inner + 1] = names[inner]
            inner -= 1
        }
        names[inner + 1] = held
    }
    var total = 0.0
    for name in names { total += op_ms[name] }
    print(f"[graph] {total:.1f}ms over {names.len()} operator kinds")
    for name in names {
        let spent = op_ms[name]
        print(
            f"[graph]   {name:<22} {spent:8.2f}ms  {op_count[name]:4} calls  " +
                f"{spent / total * 100.0:5.1f}%"
        )
    }
}

export { Attr, Graph, GraphNode, attr_float, attr_int, attr_ints, attr_str, normalize_axis }
