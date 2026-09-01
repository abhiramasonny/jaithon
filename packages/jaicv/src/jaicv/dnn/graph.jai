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
import jaitensor as jt
from jaitensor import Tensor
import std.gpu as gpu
from std.os import env
from std.time import Instant

from .graph.kernels import HIGHEST_FLOAT, LOWEST_FLOAT, f32_bits, kernel_for
from .graph.nodes import Attr, Graph, GraphNode, attr_float, attr_floats, attr_int, attr_ints
from .graph.nodes import attr_str, ints_from, maybe, need
from .graph.tensors import binary_ex_of, broadcast_shape, concat_along, contiguous_strides
from .graph.tensors import copy_of, elementwise, expand_to, gather_strided, host_ints
from .graph.tensors import normalize_axis, pad_nd_of, row_reduce_of, shaped, slice_axis
from .graph.tensors import softmax_rows, transpose_to, unary_ex_of, volume

fn op_relu(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [jt.relu(need(ins, 0, node))]
}

fn op_sigmoid(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [jt.sigmoid(need(ins, 0, node))]
}

#: `x * sigmoid(x)`, which YOLOv8 and most modern detectors use everywhere.
#:
#: No exporter emits this: ONNX has no SiLU, so a file spells it as a Sigmoid
#: and a Mul. `fuse_activations` spots that pair and rewrites it to this, which
#: turns two passes over the tensor into one and drops the intermediate.
fn op_silu(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [jt.silu(need(ins, 0, node))]
}

fn op_tanh(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [jt.tanh(need(ins, 0, node))]
}

fn op_sqrt(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [jt.sqrt(need(ins, 0, node))]
}

fn op_exp(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [jt.exp(need(ins, 0, node))]
}

fn op_log(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [jt.log(need(ins, 0, node))]
}

fn op_abs(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [jt.abs(need(ins, 0, node))]
}

fn op_neg(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [jt.neg(need(ins, 0, node))]
}

fn op_elu(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [unary_ex_of(need(ins, 0, node), 0, first: attr_float(node, "alpha", 1.0))]
}

fn op_leaky_relu(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [unary_ex_of(need(ins, 0, node), 5, first: attr_float(node, "alpha", 0.01))]
}

fn op_hard_sigmoid(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let alpha = attr_float(node, "alpha", 0.2)
    let beta = attr_float(node, "beta", 0.5)
    return [unary_ex_of(need(ins, 0, node), 1, first: alpha, second: beta)]
}

fn op_softplus(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [unary_ex_of(need(ins, 0, node), 2)]
}

fn op_erf(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [unary_ex_of(need(ins, 0, node), 3)]
}

fn op_not(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [unary_ex_of(need(ins, 0, node), 4)]
}

fn op_identity(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let input = need(ins, 0, node)
    return [copy_of(input, input.shape)]
}

#: Inference-time dropout is the identity, and the mask it can be asked for is
#: all ones because nothing was dropped.
fn op_dropout(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let input = need(ins, 0, node)
    var produced = [copy_of(input, input.shape)]
    if node.outputs.len() > 1 { produced.push(Tensor.ones(input.shape)) }
    return produced
}

fn op_prelu(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let input = need(ins, 0, node)
    let slope = need(ins, 1, node)
    let stretched = if slope.shape == input.shape { slope } else { expand_to(slope, input.shape) }
    let output = binary_ex_of(input, stretched, 6)
    if stretched is not slope { stretched.free() }
    return [output]
}

fn op_clip(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let input = need(ins, 0, node)
    var low = attr_float(node, "min", LOWEST_FLOAT)
    var high = attr_float(node, "max", HIGHEST_FLOAT)
    let low_input = maybe(ins, 1)
    if low_input is not null {
        if low_input.size != 1 {
            throw ValueError(f"{node.label()} needs a single minimum, got {low_input.shape}")
        }
        low = low_input.download()[0]
    }
    let high_input = maybe(ins, 2)
    if high_input is not null {
        if high_input.size != 1 {
            throw ValueError(f"{node.label()} needs a single maximum, got {high_input.shape}")
        }
        high = high_input.download()[0]
    }
    return [jt.clamp(input, low, high)]
}

fn op_add(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [elementwise(need(ins, 0, node), need(ins, 1, node), node.label(), jt.add)]
}

fn op_sub(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [elementwise(need(ins, 0, node), need(ins, 1, node), node.label(), jt.subtract)]
}

fn op_mul(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [elementwise(need(ins, 0, node), need(ins, 1, node), node.label(), jt.multiply)]
}

fn op_div(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [elementwise(need(ins, 0, node), need(ins, 1, node), node.label(), jt.divide)]
}

fn op_pow(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let combine = |a, b| binary_ex_of(a, b, 5)
    return [elementwise(need(ins, 0, node), need(ins, 1, node), node.label(), combine)]
}

#: ONNX `Mod`: the remainder, with the sign following whichever operand the
#: `fmod` attribute says.
#:
#: Transformer exports reach for this in the arithmetic that works out head
#: offsets from a shape, so a graph runner without it stops at the first
#: attention block.
fn op_mod(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let code = if attr_int(node, "fmod", 0) != 0 { 8 } else { 7 }
    let combine = |a, b| binary_ex_of(a, b, code)
    return [elementwise(need(ins, 0, node), need(ins, 1, node), node.label(), combine)]
}

fn op_equal(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let combine = |a, b| binary_ex_of(a, b, 0)
    return [elementwise(need(ins, 0, node), need(ins, 1, node), node.label(), combine)]
}

fn op_greater(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let combine = |a, b| binary_ex_of(a, b, 1)
    return [elementwise(need(ins, 0, node), need(ins, 1, node), node.label(), combine)]
}

fn op_less(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let combine = |a, b| binary_ex_of(a, b, 2)
    return [elementwise(need(ins, 0, node), need(ins, 1, node), node.label(), combine)]
}

fn op_and(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let combine = |a, b| binary_ex_of(a, b, 3)
    return [elementwise(need(ins, 0, node), need(ins, 1, node), node.label(), combine)]
}

fn op_or(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let combine = |a, b| binary_ex_of(a, b, 4)
    return [elementwise(need(ins, 0, node), need(ins, 1, node), node.label(), combine)]
}

#: Fold a variadic operator left to right, broadcasting at every step.
fn fold_inputs(
    node: GraphNode,
    ins: list[Tensor?],
    combine: fn(Tensor, Tensor) -> Tensor
) -> Tensor {
    if ins.len() == 0 { throw ValueError(f"{node.label()} needs at least one input") }
    var total = need(ins, 0, node)
    var mine = false
    for index in 1..ins.len() {
        let merged = elementwise(total, need(ins, index, node), node.label(), combine)
        if mine { total.free() }
        total = merged
        mine = true
    }
    if mine { return total }
    return copy_of(total, total.shape)
}

fn op_min(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [fold_inputs(node, ins, jt.minimum)]
}

fn op_max(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [fold_inputs(node, ins, jt.maximum)]
}

fn op_sum(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return [fold_inputs(node, ins, jt.add)]
}

fn op_mean(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let total = fold_inputs(node, ins, jt.add)
    let averaged = jt.scale(total, 1.0 / float(ins.len()))
    total.free()
    return [averaged]
}

fn op_where(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let condition = need(ins, 0, node)
    let yes = need(ins, 1, node)
    let no = need(ins, 2, node)
    let shape = broadcast_shape(
        broadcast_shape(condition.shape, yes.shape, node.label()),
        no.shape,
        node.label()
    )
    let picked = if condition.shape == shape { condition } else { expand_to(condition, shape) }
    let left = if yes.shape == shape { yes } else { expand_to(yes, shape) }
    let right = if no.shape == shape { no } else { expand_to(no, shape) }
    let output = jt.where(picked, left, right)
    if picked is not condition { picked.free() }
    if left is not yes { left.free() }
    if right is not no { right.free() }
    return [output]
}

fn op_reshape(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let data = need(ins, 0, node)
    var dims = host_ints(need(ins, 1, node))
    let allowzero = attr_int(node, "allowzero", 0)
    var infer = -1
    var known = 1
    for (axis, dimension) in dims.enumerate() {
        if dimension == 0 and allowzero == 0 {
            if axis >= data.rank() {
                throw ValueError(f"{node.label()} copies axis {axis} of a rank-{data.rank()} input")
            }
            dims[axis] = data.shape[axis]
            known *= dims[axis]
        } elif dimension == -1 {
            if infer >= 0 { throw ValueError(f"{node.label()} leaves two dimensions to infer") }
            infer = axis
        } elif dimension < 0 {
            throw ValueError(f"{node.label()} asks for the negative dimension {dimension}")
        } else {
            known *= dimension
        }
    }
    if infer >= 0 {
        if known == 0 or data.size % known != 0 {
            throw ValueError(f"{node.label()} cannot fit {data.size} values into {dims}")
        }
        dims[infer] = data.size // known
    }
    return [copy_of(data, shaped(dims))]
}

fn op_flatten(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let data = need(ins, 0, node)
    let rank = data.rank()
    let axis = attr_int(node, "axis", 1)
    let split = if axis < 0 { rank + axis } else { axis }
    if split < 0 or split > rank {
        throw ValueError(f"{node.label()} flattens at axis {axis}, outside rank {rank}")
    }
    var outer = 1
    for other in 0..split { outer *= data.shape[other] }
    return [copy_of(data, [outer, data.size // outer])]
}

fn op_squeeze(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let data = need(ins, 0, node)
    let asked = ints_from(node, ins, 1, "axes")
    var dims: list[int] = []
    if asked is null {
        for dimension in data.shape {
            if dimension != 1 { dims.push(dimension) }
        }
    } else {
        var dropped: list[int] = []
        for axis in asked { dropped.push(normalize_axis(axis, data.rank(), node.label())) }
        for (axis, dimension) in data.shape.enumerate() {
            if axis in dropped {
                if dimension != 1 {
                    throw ValueError(
                        f"{node.label()} squeezes axis {axis}, which is {dimension} wide"
                    )
                }
                continue
            }
            dims.push(dimension)
        }
    }
    return [copy_of(data, shaped(dims))]
}

fn op_unsqueeze(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let data = need(ins, 0, node)
    let asked = ints_from(node, ins, 1, "axes")
    if asked is null { throw ValueError(f"{node.label()} needs the axes to insert") }
    let rank = data.rank() + asked.len()
    var inserted: list[int] = []
    for axis in asked { inserted.push(normalize_axis(axis, rank, node.label())) }
    var dims: list[int] = []
    var next = 0
    for axis in 0..rank {
        if axis in inserted {
            dims.push(1)
        } else {
            dims.push(data.shape[next])
            next += 1
        }
    }
    return [copy_of(data, dims)]
}

fn op_transpose(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let data = need(ins, 0, node)
    let rank = data.rank()
    var perm = attr_ints(node, "perm", [])
    if perm.len() == 0 {
        var reversed: list[int] = []
        var axis = rank - 1
        while axis >= 0 {
            reversed.push(axis)
            axis -= 1
        }
        perm = reversed
    }
    if perm.len() != rank {
        throw ValueError(f"{node.label()} permutes {perm.len()} axes of a rank-{rank} input")
    }
    var seen: list[int] = []
    for axis in perm {
        let resolved = normalize_axis(axis, rank, node.label())
        if resolved in seen { throw ValueError(f"{node.label()} names axis {resolved} twice") }
        seen.push(resolved)
    }
    return [transpose_to(data, seen)]
}

fn op_concat(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    if ins.len() == 0 { throw ValueError(f"{node.label()} needs at least one input") }
    var parts: list[Tensor] = []
    for index in 0..ins.len() { parts.push(need(ins, index, node)) }
    let axis = normalize_axis(attr_int(node, "axis", 0), parts[0].rank(), node.label())
    return [concat_along(parts, axis, node.label())]
}

fn op_split(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let data = need(ins, 0, node)
    let axis = normalize_axis(attr_int(node, "axis", 0), data.rank(), node.label())
    let width = data.shape[axis]
    let wanted = max(node.outputs.len(), 1)
    var widths = ints_from(node, ins, 1, "split") ?? []
    if widths.len() == 0 {
        let parts = attr_int(node, "num_outputs", wanted)
        if parts <= 0 or width % parts != 0 {
            throw ValueError(f"{node.label()} cannot cut {width} into {parts} even pieces")
        }
        widths = [width // parts for _piece in 0..parts]
    }
    var total = 0
    for piece in widths { total += piece }
    if total != width {
        throw ValueError(f"{node.label()} splits into {widths}, which is not axis {axis}'s {width}")
    }
    var produced: list[Tensor] = []
    var start = 0
    for piece in widths {
        produced.push(slice_axis(data, axis, start, piece))
        start += piece
    }
    return produced
}

fn op_slice(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let data = need(ins, 0, node)
    let rank = data.rank()
    let starts = ints_from(node, ins, 1, "starts")
    let ends = ints_from(node, ins, 2, "ends")
    if starts is null or ends is null {
        throw ValueError(f"{node.label()} needs both starts and ends")
    }
    if starts.len() != ends.len() {
        throw ValueError(f"{node.label()} has {starts.len()} starts and {ends.len()} ends")
    }
    let axes = ints_from(node, ins, 3, "axes") ?? [axis for axis in 0..starts.len()]
    let steps = ints_from(node, ins, 4, "steps") ?? [1 for _axis in starts]
    let source = contiguous_strides(data.shape)
    var shape = [dimension for dimension in data.shape]
    var strides = [stride for stride in source]
    var base = 0
    for (slot, axis) in axes.enumerate() {
        let resolved = normalize_axis(axis, rank, node.label())
        let width = data.shape[resolved]
        let step = steps[slot]
        if step == 0 {
            throw ValueError(f"{node.label()} slices axis {resolved} with a step of zero")
        }
        var first = starts[slot]
        var last = ends[slot]
        if first < 0 { first += width }
        if last < 0 { last += width }
        var span = 0
        if step > 0 {
            first = min(max(first, 0), width)
            last = min(max(last, 0), width)
            span = (last - first + step - 1) // step
        } else {
            first = min(max(first, 0), width - 1)
            last = min(max(last, -1), width - 1)
            #: `ceil((last - first) / step)` with both sides negative, written
            #: for a floor division. Getting the rounding backwards here only
            #: shows up once the step is steeper than one.
            span = (last - first + step + 1) // step
        }
        #: A range that comes up empty is a legitimate answer, not a mistake:
        #: exported graphs slice the tail off a shape and get nothing whenever
        #: the tail is not there, and then concatenate the nothing back in.
        shape[resolved] = max(span, 0)
        strides[resolved] = source[resolved] * step
        if shape[resolved] > 0 { base += first * source[resolved] }
    }
    for dimension in shape {
        if dimension == 0 { return [Tensor(shape)] }
    }
    return [gather_strided(data, shape, base, strides)]
}

fn op_gather(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let data = need(ins, 0, node)
    let indices = need(ins, 1, node)
    let axis = normalize_axis(attr_int(node, "axis", 0), data.rank(), node.label())
    var inner = 1
    for other in axis + 1..data.rank() { inner *= data.shape[other] }
    var dims: list[int] = []
    for other in 0..axis { dims.push(data.shape[other]) }
    #: A rank-zero index would drop this axis in ONNX, but jaitensor has no
    #: rank-zero tensor, so such an index arrives as `[1]` and the axis stays
    #: as a width of one. A graph that needs it gone follows with a Squeeze.
    for dimension in indices.shape { dims.push(dimension) }
    for other in axis + 1..data.rank() { dims.push(data.shape[other]) }
    let output = Tensor(dims)
    kernel_for("gather_axis").dispatch_async(
        [data.buffer(), indices.buffer(), output.buffer()],
        output.size,
        [output.size, data.shape[axis], inner, indices.size]
    )
    return [output]
}

fn op_pad(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let data = need(ins, 0, node)
    let rank = data.rank()
    let widths = ints_from(node, ins, 1, "pads")
    if widths is null { throw ValueError(f"{node.label()} needs the pad widths") }
    let name = attr_str(node, "mode", "constant")
    let mode = match name {
        "constant" => 0,
        "reflect" => 1,
        "edge" => 2,
        _ => throw ValueError(f"{node.label()} asks for pad mode {name}, which is not implemented"),
    }
    var value = attr_float(node, "value", 0.0)
    let supplied = maybe(ins, 2)
    if supplied is not null {
        if supplied.size != 1 {
            throw ValueError(f"{node.label()} needs a single pad value, got {supplied.shape}")
        }
        value = supplied.download()[0]
    }
    let axes = ints_from(node, ins, 3, "axes") ?? [axis for axis in 0..rank]
    if widths.len() != 2 * axes.len() {
        throw ValueError(f"{node.label()} has {widths.len()} pad widths for {axes.len()} axes")
    }
    var before = [0 for _axis in 0..rank]
    var after = [0 for _axis in 0..rank]
    for (slot, axis) in axes.enumerate() {
        let resolved = normalize_axis(axis, rank, node.label())
        before[resolved] = widths[slot]
        after[resolved] = widths[axes.len() + slot]
    }
    return [pad_nd_of(data, before, after, mode, value)]
}

fn op_shape(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let data = need(ins, 0, node)
    let rank = data.rank()
    var first = attr_int(node, "start", 0)
    var last = attr_int(node, "end", rank)
    if first < 0 { first += rank }
    if last < 0 { last += rank }
    first = min(max(first, 0), rank)
    last = min(max(last, 0), rank)
    if last <= first {
        throw ValueError(f"{node.label()} asks for axes [{first}, {last}) of a rank-{rank} input")
    }
    return [Tensor([last - first], [float(data.shape[axis]) for axis in first..last])]
}

#: ONNX's numbering for `Cast`'s `to`, which is TensorProto.DataType.
let INTEGER_TYPES = [2, 3, 4, 5, 6, 7, 12, 13]
let FLOAT_TYPES = [1, 10, 11, 16]

fn op_cast(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let data = need(ins, 0, node)
    let target = attr_int(node, "to", 1)
    if target in FLOAT_TYPES { return [copy_of(data, data.shape)] }
    if target in INTEGER_TYPES { return [unary_ex_of(data, 6)] }
    if target == 9 { return [unary_ex_of(data, 7)] }
    throw ValueError(f"{node.label()} casts to ONNX type {target}, which is not implemented")
}

fn op_constant(node: GraphNode, _ins: list[Tensor?]) -> list[Tensor] {
    let held = node.attrs.get("value", null)
    if held is not null {
        if held.kind != "tensor" {
            throw ValueError(f"{node.label()} carries value as {held.kind}, expected tensor")
        }
        let stored = held.tensor
        if stored is null { throw ValueError(f"{node.label()} carries an empty value") }
        return [copy_of(stored, stored.shape)]
    }
    let single = node.attrs.get("value_float", null)
    if single is not null { return [Tensor([1], [single.f])] }
    let whole = node.attrs.get("value_int", null)
    if whole is not null { return [Tensor([1], [float(whole.i)])] }
    let many = node.attrs.get("value_floats", null)
    if many is not null { return [Tensor([many.floats.len()], many.floats)] }
    let counts = node.attrs.get("value_ints", null)
    if counts is not null {
        return [Tensor([counts.ints.len()], [float(value) for value in counts.ints])]
    }
    throw ValueError(f"{node.label()} has no value attribute this executor understands")
}

fn op_constant_of_shape(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let dims = host_ints(need(ins, 0, node))
    var value = 0.0
    let held = node.attrs.get("value", null)
    if held is not null {
        let stored = held.tensor
        if stored is null or stored.size != 1 {
            throw ValueError(f"{node.label()} needs a one-element value attribute")
        }
        value = stored.download()[0]
    }
    return [Tensor.full(shaped(dims), value)]
}

#: The explicit padding a window operator wants, as ONNX's
#: `[begin..., end...]`, working out what `auto_pad` meant when that is how
#: the file said it.
fn resolve_pads(
    node: GraphNode,
    spatial: list[int],
    window: list[int],
    strides: list[int],
    dilations: list[int]
) -> list[int] {
    let axes = spatial.len()
    let auto_pad = attr_str(node, "auto_pad", "NOTSET")
    if auto_pad == "NOTSET" {
        let given = attr_ints(node, "pads", [])
        if given.len() == 0 { return [0 for _slot in 0..2 * axes] }
        if given.len() != 2 * axes {
            throw ValueError(f"{node.label()} has {given.len()} pads for {axes} spatial axes")
        }
        return given
    }
    if auto_pad == "VALID" { return [0 for _slot in 0..2 * axes] }
    if auto_pad != "SAME_UPPER" and auto_pad != "SAME_LOWER" {
        throw ValueError(f"{node.label()} asks for auto_pad {auto_pad}, which is not implemented")
    }
    var pads = [0 for _slot in 0..2 * axes]
    for axis in 0..axes {
        let reach = dilations[axis] * (window[axis] - 1) + 1
        let covered = (spatial[axis] + strides[axis] - 1) // strides[axis]
        let total = max(0, (covered - 1) * strides[axis] + reach - spatial[axis])
        let front = if auto_pad == "SAME_UPPER" { total // 2 } else { total - total // 2 }
        pads[axis] = front
        pads[axes + axis] = total - front
    }
    return pads
}

#: Everything here convolves and pools in two dimensions, so a per-axis
#: attribute that is not a pair belongs to a 1-D or 3-D operator and would
#: otherwise fail somewhere further in with a confusing index.
fn two_of(values: list[int], node: GraphNode, what: str) -> list[int] {
    if values.len() != 2 {
        throw ValueError(
            f"{node.label()} has {what} {values}, and this executor only does " +
                "two-dimensional windows"
        )
    }
    return values
}

fn only_square(values: list[int], node: GraphNode, what: str) -> int {
    for value in values {
        if value != values[0] {
            throw ValueError(
                f"{node.label()} has {what} {values}, and jaitensor's window operators " +
                    "take one value for both axes"
            )
        }
    }
    return values[0]
}

#: NCHW `[N, C, H, W]` grown by ONNX's `[top, left, bottom, right]`.
fn pad_spatial(input: Tensor, pads: list[int], value: float) -> Tensor {
    if pads[0] == 0 and pads[1] == 0 and pads[2] == 0 and pads[3] == 0 { return input }
    return pad_nd_of(input, [0, 0, pads[0], pads[1]], [0, 0, pads[2], pads[3]], 0, value)
}

fn op_conv(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let x = need(ins, 0, node)
    let w = need(ins, 1, node)
    let bias = maybe(ins, 2)
    if x.rank() != 4 {
        throw ValueError(f"{node.label()} needs a 4-D NCHW input, got {x.shape}")
    }
    if w.rank() != 4 {
        throw ValueError(f"{node.label()} needs 4-D weights, got {w.shape}")
    }
    let dilations = two_of(attr_ints(node, "dilations", [1, 1]), node, "dilations")
    for spread in dilations {
        if spread != 1 {
            throw ValueError(
                f"{node.label()} has dilations {dilations}, which jaitensor's convolution " +
                    "does not take"
            )
        }
    }
    let window = two_of(
        attr_ints(node, "kernel_shape", [w.shape[2], w.shape[3]]),
        node,
        "kernel_shape"
    )
    if window != [w.shape[2], w.shape[3]] {
        throw ValueError(f"{node.label()} declares kernel_shape {window} for weights {w.shape}")
    }
    let strides = two_of(attr_ints(node, "strides", [1, 1]), node, "strides")
    let step = only_square(strides, node, "strides")
    let pads = resolve_pads(node, [x.shape[2], x.shape[3]], window, strides, [1, 1])
    let groups = attr_int(node, "group", 1)
    let out_channels = w.shape[0]
    let in_per_group = w.shape[1]
    if groups <= 0 or x.shape[1] % groups != 0 or out_channels % groups != 0 {
        throw ValueError(f"{node.label()} splits {x.shape[1]} channels into {groups} groups")
    }
    if x.shape[1] // groups != in_per_group {
        throw ValueError(
            f"{node.label()} has weights {w.shape} for {x.shape[1]} input channels in " +
                f"{groups} groups"
        )
    }
    if bias is not null and bias.size != out_channels {
        throw ValueError(f"{node.label()} has {bias.size} biases for {out_channels} filters")
    }

    #: One padding value goes to jaitensor when the four agree, because its
    #: convolution already skips the holes; anything else is materialised
    #: first, which costs a copy but is the only way to say "one row on top".
    let even = pads[0] == pads[1] and pads[1] == pads[2] and pads[2] == pads[3]
    let padded = if even { x } else { pad_spatial(x, pads, 0.0) }
    let inner_pad = if even { pads[0] } else { 0 }

    #: Everything below stays in the layout the file already uses. ONNX is
    #: NCHW activations and OIHW weights, which is exactly what `conv2d_nchw`
    #: takes, so nothing is transposed on the way in or out -- the round trip
    #: this used to make was two thirds of what convolution appeared to cost.
    var pieces: list[Tensor] = []
    let out_per_group = out_channels // groups
    for group in 0..groups {
        let slice_in = if groups == 1 {
            padded
        } else {
            slice_axis(padded, 1, group * in_per_group, in_per_group)
        }
        #: A group's filters are a contiguous run of OIHW's first axis, and
        #: they are the same run on every frame, so the slice is kept on the
        #: node. Ungrouped convolutions -- nearly all of them -- slice nothing
        #: and hand the initializer straight through.
        let slice_w = if groups == 1 { w } else { group_weights(node, w, group, out_per_group) }
        var slice_bias: Tensor? = null
        if bias is not null {
            slice_bias = if groups == 1 {
                bias
            } else {
                slice_axis(bias, 0, group * out_per_group, out_per_group)
            }
        }
        pieces.push(
            jt.conv2d_nchw(slice_in, slice_w, bias: slice_bias, stride: step, padding: inner_pad)
        )
        if slice_in is not padded { slice_in.free() }
        if slice_bias is not null and slice_bias is not bias { slice_bias.free() }
    }
    if padded is not x { padded.free() }
    let output = if pieces.len() == 1 { pieces[0] } else { concat_along(pieces, 1, node.label()) }
    if output is not pieces[0] {
        for piece in pieces { piece.free() }
    }
    return [output]
}

#: One group's filters out of an OIHW weight tensor, memoised on the node.
#:
#: The slice depends only on the weights, which are an initializer, so it is
#: the same tensor every run -- but it is recomputed if the caller ever feeds
#: different weights under the same name.
fn group_weights(node: GraphNode, w: Tensor, group: int, out_per_group: int) -> Tensor {
    let key = f"w{group}"
    let kept = node.cached.get(key, null)
    if kept is not null and node.cached_of.get(key, null) is w { return kept }
    if kept is not null { kept.free() }
    let cut = slice_axis(w, 0, group * out_per_group, out_per_group)
    node.cached[key] = cut
    node.cached_of[key] = w
    return cut
}

fn op_conv_transpose(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let x = need(ins, 0, node)
    let w = need(ins, 1, node)
    let bias = maybe(ins, 2)
    if x.rank() != 4 { throw ValueError(f"{node.label()} needs a 4-D NCHW input, got {x.shape}") }
    if w.rank() != 4 { throw ValueError(f"{node.label()} needs 4-D weights, got {w.shape}") }
    let groups = attr_int(node, "group", 1)
    let in_channels = x.shape[1]
    let out_per_group = w.shape[1]
    if groups <= 0 or in_channels % groups != 0 or w.shape[0] != in_channels {
        throw ValueError(f"{node.label()} has weights {w.shape} for input {x.shape}")
    }
    let window = two_of(
        attr_ints(node, "kernel_shape", [w.shape[2], w.shape[3]]),
        node,
        "kernel_shape"
    )
    let strides = two_of(attr_ints(node, "strides", [1, 1]), node, "strides")
    let dilations = two_of(attr_ints(node, "dilations", [1, 1]), node, "dilations")
    let extra = two_of(attr_ints(node, "output_padding", [0, 0]), node, "output_padding")
    let pads = resolve_pads(node, [x.shape[2], x.shape[3]], window, strides, dilations)
    let out_channels = out_per_group * groups
    if bias is not null and bias.size != out_channels {
        throw ValueError(f"{node.label()} has {bias.size} biases for {out_channels} filters")
    }
    var spatial: list[int] = []
    for axis in 0..2 {
        let reach = dilations[axis] * (window[axis] - 1) + 1
        spatial.push(
            (x.shape[2 + axis] - 1) * strides[axis] - pads[axis] - pads[2 + axis] +
                reach +
                extra[axis]
        )
    }
    #: `output_shape` may name the spatial axes alone or the whole output, and
    #: officially it regenerates the padding rather than describing it. What
    #: is safe is to check it, and to say so when it disagrees, instead of
    #: quietly producing a differently shaped tensor.
    let declared = attr_ints(node, "output_shape", [])
    let named = if declared.len() == 4 { [declared[2], declared[3]] } else { declared }
    if named.len() > 0 and named != spatial {
        throw ValueError(
            f"{node.label()} declares output_shape {declared}, but its pads and strides " +
                f"give {spatial}"
        )
    }
    let in_per_group = in_channels // groups
    var pieces: list[Tensor] = []
    for group in 0..groups {
        let part_x = if groups == 1 {
            x
        } else {
            slice_axis(x, 1, group * in_per_group, in_per_group)
        }
        let part_w = if groups == 1 {
            w
        } else {
            slice_axis(w, 0, group * in_per_group, in_per_group)
        }
        var part_bias: Tensor? = null
        if bias is not null {
            part_bias = if groups == 1 {
                bias
            } else {
                slice_axis(bias, 0, group * out_per_group, out_per_group)
            }
        }
        let output = Tensor([x.shape[0], out_per_group, spatial[0], spatial[1]])
        let bias_buffer = if part_bias is null { output.buffer() } else { part_bias.buffer() }
        kernel_for("conv_transpose_nchw").dispatch_async([part_x.buffer(), part_w.buffer(), bias_buffer, output.buffer()], output.size, [output.size, in_per_group, x.shape[2], x.shape[3], out_per_group, spatial[0], spatial[1], window[0], window[1], strides[0], strides[1], pads[0], pads[1], dilations[0], dilations[1], if part_bias is null {
            0
        } else {
            1
        }])
        pieces.push(output)
        if part_x is not x { part_x.free() }
        if part_w is not w { part_w.free() }
        if part_bias is not null and part_bias is not bias { part_bias.free() }
    }
    if pieces.len() == 1 { return [pieces[0]] }
    let joined = concat_along(pieces, 1, node.label())
    for piece in pieces { piece.free() }
    return [joined]
}

fn op_pool(node: GraphNode, ins: list[Tensor?], is_max: bool) -> list[Tensor] {
    let x = need(ins, 0, node)
    if x.rank() != 4 { throw ValueError(f"{node.label()} needs a 4-D NCHW input, got {x.shape}") }
    if node.outputs.len() > 1 {
        throw ValueError(f"{node.label()} is asked for pooling indices, which is not implemented")
    }
    let window = two_of(attr_ints(node, "kernel_shape", []), node, "kernel_shape")
    let dilations = two_of(attr_ints(node, "dilations", [1, 1]), node, "dilations")
    for spread in dilations {
        if spread != 1 {
            throw ValueError(f"{node.label()} has dilations {dilations}, which is not implemented")
        }
    }
    let strides = two_of(attr_ints(node, "strides", [1, 1]), node, "strides")
    let size = only_square(window, node, "kernel_shape")
    let step = only_square(strides, node, "strides")
    var pads = resolve_pads(node, [x.shape[2], x.shape[3]], window, strides, [1, 1])

    #: `ceil_mode` keeps the window that hangs off the end, which is the same
    #: shape as padding the end far enough that the floor formula finds it.
    if attr_int(node, "ceil_mode", 0) != 0 {
        for axis in 0..2 {
            let extent = x.shape[2 + axis] + pads[axis] + pads[2 + axis]
            let covered = (extent - window[axis] + step - 1) // step + 1
            pads[2 + axis] += max(0, (covered - 1) * step + window[axis] - extent)
        }
    }
    let empty = pads[0] == 0 and pads[1] == 0 and pads[2] == 0 and pads[3] == 0
    let filler = if is_max { LOWEST_FLOAT } else { 0.0 }
    let padded = pad_spatial(x, pads, filler)
    let channel_last = jt.nhwc(padded)
    if padded is not x { padded.free() }
    var pooled = if is_max {
        jt.max_pool2d(channel_last, size, stride: step, padding: 0)
    } else {
        jt.avg_pool2d(channel_last, size, stride: step, padding: 0)
    }

    #: An average pool that does not count the padding is the padded sum over
    #: the count of real values, and pooling a mask of ones is exactly that
    #: count -- so the correction is one more pool and a divide.
    if not is_max and not empty and attr_int(node, "count_include_pad", 0) == 0 {
        let mask = Tensor.ones(x.shape)
        let spread = pad_spatial(mask, pads, 0.0)
        mask.free()
        let mask_last = jt.nhwc(spread)
        spread.free()
        let share = jt.avg_pool2d(mask_last, size, stride: step, padding: 0)
        mask_last.free()
        let corrected = jt.divide(pooled, share)
        pooled.free()
        share.free()
        pooled = corrected
    }
    channel_last.free()
    let output = jt.nchw(pooled)
    pooled.free()
    return [output]
}

fn op_max_pool(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return op_pool(node, ins, true)
}

fn op_average_pool(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return op_pool(node, ins, false)
}

fn op_global_pool(node: GraphNode, ins: list[Tensor?], op: int) -> list[Tensor] {
    let x = need(ins, 0, node)
    if x.rank() < 3 {
        throw ValueError(f"{node.label()} needs at least a 3-D NCHW input, got {x.shape}")
    }
    let planes = x.shape[0] * x.shape[1]
    var dims = [x.shape[0], x.shape[1]]
    for _axis in 2..x.rank() { dims.push(1) }
    return [row_reduce_of(x, planes, x.size // planes, op, shape: dims)]
}

fn op_global_average_pool(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return op_global_pool(node, ins, 1)
}

fn op_global_max_pool(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return op_global_pool(node, ins, 2)
}

fn op_batch_norm(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let x = need(ins, 0, node)
    let scale = need(ins, 1, node)
    let shift = need(ins, 2, node)
    let mean = need(ins, 3, node)
    let variance = need(ins, 4, node)
    if x.rank() < 2 {
        throw ValueError(f"{node.label()} needs at least a 2-D input, got {x.shape}")
    }
    if node.outputs.len() > 1 {
        throw ValueError(
            f"{node.label()} is asked for its running statistics, which only training " +
                "produces and this executor does not"
        )
    }
    let channels = x.shape[1]
    for parameter in [scale, shift, mean, variance] {
        if parameter.size != channels {
            throw ValueError(
                f"{node.label()} has a {parameter.size}-wide parameter for {channels} channels"
            )
        }
    }
    let output = Tensor(x.shape)
    kernel_for("affine_norm_nchw").dispatch_async(
        [
            x.buffer(),
            mean.buffer(),
            variance.buffer(),
            scale.buffer(),
            shift.buffer(),
            output.buffer(),
        ],
        x.size,
        [
            x.size,
            channels,
            x.size // (x.shape[0] * channels),
            f32_bits(attr_float(node, "epsilon", 1e-5)),
            0,
        ]
    )
    return [output]
}

#: ONNX `LayerNormalization`: normalise each row over the axes from `axis` on.
#:
#: Every transformer export -- ViT, CvT, and anything with an attention block
#: -- opens with one of these, so a graph runner without it cannot load them at
#: all. The axes from `axis` onward are flattened into one row, which is what
#: normalising over them means, and the scale and shift run along that row.
fn op_layer_norm(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let x = need(ins, 0, node)
    let scale = need(ins, 1, node)
    let shift = maybe(ins, 2)
    let rank = x.rank()
    let raw_axis = attr_int(node, "axis", -1)
    let axis = if raw_axis < 0 { rank + raw_axis } else { raw_axis }
    if axis < 0 or axis >= rank {
        throw ValueError(f"{node.label()} normalises axis {raw_axis} of a rank-{rank} input")
    }
    var width = 1
    for other in axis..rank { width *= x.shape[other] }
    if width <= 0 { throw ValueError(f"{node.label()} normalises an empty axis of {x.shape}") }
    let rows = x.size // width
    if scale.size != width {
        throw ValueError(f"{node.label()} scales {scale.size} of {width} normalised values")
    }
    if shift is not null and shift.size != width {
        throw ValueError(f"{node.label()} shifts {shift.size} of {width} normalised values")
    }
    #: The shift is optional in the operator and not in the kernel, so an
    #: absent one becomes zeros rather than a second code path.
    let bias = if shift is not null { shift } else { jt.Tensor.zeros([width]) }
    let output = jt.Tensor(x.shape)
    let mean = jt.Tensor([rows])
    let inv_std = jt.Tensor([rows])
    kernel_for("layer_norm_rows").dispatch_async(
        [
            x.buffer(),
            scale.buffer(),
            bias.buffer(),
            output.buffer(),
            mean.buffer(),
            inv_std.buffer(),
        ],
        rows * 256,
        [width, f32_bits(attr_float(node, "epsilon", 1.0e-5))],
        group_size: 256
    )
    if shift is null { bias.free() }
    #: `Mean` and `InvStdDev` are optional outputs, shaped like the input with
    #: the normalised axes collapsed to one. The ones a graph does not name are
    #: dropped by the caller.
    var moment_shape: list[int] = []
    for other in 0..rank { moment_shape.push(if other < axis { x.shape[other] } else { 1 }) }
    return [output, jt.reshape(mean, moment_shape), jt.reshape(inv_std, moment_shape)]
}

#: One direction of an LSTM, from `[seq, batch, input]` to `[seq, batch, H]`.
#:
#: `x_gates` already holds every timestep's `Xt W' + b`, because that half does
#: not depend on the recurrence and so is one matrix product for the whole
#: sequence rather than one per step. What is left per step is the recurrent
#: product and the gates.
fn lstm_direction(
    x_gates: Tensor,
    recurrent: Tensor,
    initial_h: Tensor,
    initial_c: Tensor,
    steps: int,
    batch: int,
    hidden: int,
    clip: float,
    reverse: bool
) -> tuple[Tensor, Tensor, Tensor] {
    #: `steps` owns every hidden state it produces, because they are also the
    #: output sequence. The cell state is not kept, so each new one replaces
    #: and frees the one before -- except the caller's, which the caller frees.
    var state_h = initial_h
    var state_c = initial_c
    var own_cell = false
    var produced: list[Tensor] = []
    for index in 0..steps {
        let step = if reverse { steps - 1 - index } else { index }
        let recurred = jt.matmul(state_h, recurrent, trans_b: true)
        let next_c = jt.Tensor([batch, hidden])
        let next_h = jt.Tensor([batch, hidden])
        kernel_for("lstm_gates").dispatch_async(
            [
                x_gates.buffer(),
                recurred.buffer(),
                state_c.buffer(),
                next_c.buffer(),
                next_h.buffer(),
            ],
            batch * hidden,
            [batch * hidden, hidden, f32_bits(clip), step * batch * 4 * hidden]
        )
        recurred.free()
        if own_cell { state_c.free() }
        state_c = next_c
        own_cell = true
        state_h = next_h
        produced.push(next_h)
    }

    #: A reversed pass produces its steps back to front, and the output is
    #: indexed by time whichever way it was walked.
    var ordered = produced
    if reverse {
        ordered = []
        for index in 0..produced.len() { ordered.push(produced[produced.len() - 1 - index]) }
    }
    var rows: list[Tensor] = []
    for part in ordered { rows.push(copy_of(part, [1, batch, hidden])) }
    let sequence = if rows.len() == 1 {
        rows[0]
    } else {
        concat_along(rows, 0, "an LSTM's output sequence")
    }
    if rows.len() > 1 {
        for row in rows { row.free() }
    }
    let final_h = copy_of(state_h, [batch, hidden])
    let final_c = copy_of(state_c, [batch, hidden])
    for part in produced { part.free() }
    if own_cell { state_c.free() }
    return (sequence, final_h, final_c)
}

#: ONNX `LSTM`.
#:
#: Weights arrive as `[directions, 4H, input]` and `[directions, 4H, H]` with
#: the gates ordered i, o, f, c, and the bias as `[directions, 8H]` holding the
#: input biases and then the recurrent ones. Outputs are the whole sequence
#: `[seq, directions, batch, H]` and the final hidden and cell states.
fn op_lstm(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let x = need(ins, 0, node)
    let w = need(ins, 1, node)
    let r = need(ins, 2, node)
    let bias = maybe(ins, 3)
    if maybe(ins, 4) is not null {
        throw ValueError(
            f"{node.label()} gives per-sequence lengths, which this executor does not read; " +
                "every sequence in the batch has to be the same length"
        )
    }
    if maybe(ins, 7) is not null {
        throw ValueError(f"{node.label()} has peephole weights, which this executor does not apply")
    }
    if node.attrs.get("activations", null) is not null {
        throw ValueError(
            f"{node.label()} names its own activations; this executor uses the sigmoid and " +
                "tanh the operator defaults to"
        )
    }
    if attr_int(node, "layout", 0) != 0 {
        throw ValueError(
            f"{node.label()} uses the batch-first layout, which this executor does not read"
        )
    }
    if attr_int(node, "input_forget", 0) != 0 {
        throw ValueError(
            f"{node.label()} couples its input and forget gates, which this executor does not"
        )
    }
    if x.rank() != 3 { throw ValueError(f"{node.label()} needs a 3-D input, got {x.shape}") }
    if w.rank() != 3 or r.rank() != 3 {
        throw ValueError(f"{node.label()} needs 3-D weights, got {w.shape} and {r.shape}")
    }
    let steps = x.shape[0]
    let batch = x.shape[1]
    let input_size = x.shape[2]
    let directions = w.shape[0]
    let hidden = attr_int(node, "hidden_size", r.shape[2])
    let facing = attr_str(node, "direction", "forward")
    let wanted = if facing == "bidirectional" { 2 } else { 1 }
    if directions != wanted or r.shape[0] != directions {
        throw ValueError(
            f"{node.label()} is {facing} but carries {directions} directions of weights"
        )
    }
    if w.shape[1] != 4 * hidden or r.shape[1] != 4 * hidden or r.shape[2] != hidden {
        throw ValueError(
            f"{node.label()} has weights {w.shape} and {r.shape} for {hidden} hidden units"
        )
    }
    if w.shape[2] != input_size {
        throw ValueError(f"{node.label()} has weights {w.shape} for {input_size} inputs")
    }
    let clip = attr_float(node, "clip", 0.0)

    let flat_x = copy_of(x, [steps * batch, input_size])
    var sequences: list[Tensor] = []
    var last_h: list[Tensor] = []
    var last_c: list[Tensor] = []
    for direction in 0..directions {
        let wd = taken(slice_axis(w, 0, direction, 1), [4 * hidden, input_size])
        let rd = taken(slice_axis(r, 0, direction, 1), [4 * hidden, hidden])
        #: Every timestep's input half at once: it does not depend on the
        #: recurrence, so it is one product for the sequence rather than one
        #: for each step of it.
        var x_gates = jt.matmul(flat_x, wd, trans_b: true)
        wd.free()
        if bias is not null {
            #: `[directions, 8H]` is the input biases followed by the recurrent
            #: ones, and both land on the same gates.
            let row = taken(slice_axis(bias, 0, direction, 1), [8 * hidden])
            let wb = taken(slice_axis(row, 0, 0, 4 * hidden), [4 * hidden])
            let rb = taken(slice_axis(row, 0, 4 * hidden, 4 * hidden), [4 * hidden])
            let both = jt.add(wb, rb)
            let shifted = jt.add(x_gates, both)
            x_gates.free()
            x_gates = shifted
            row.free()
            wb.free()
            rb.free()
            both.free()
        }
        let given_h = maybe(ins, 5)
        let given_c = maybe(ins, 6)
        let start_h = if given_h is null {
            jt.Tensor.zeros([batch, hidden])
        } else {
            taken(slice_axis(given_h, 0, direction, 1), [batch, hidden])
        }
        let start_c = if given_c is null {
            jt.Tensor.zeros([batch, hidden])
        } else {
            taken(slice_axis(given_c, 0, direction, 1), [batch, hidden])
        }
        let (sequence, final_h, final_c) = lstm_direction(
            x_gates,
            rd,
            start_h,
            start_c,
            steps,
            batch,
            hidden,
            clip,
            facing == "reverse" or direction == 1
        )
        x_gates.free()
        rd.free()
        start_h.free()
        start_c.free()
        sequences.push(copy_of(sequence, [steps, 1, batch, hidden]))
        sequence.free()
        last_h.push(copy_of(final_h, [1, batch, hidden]))
        last_c.push(copy_of(final_c, [1, batch, hidden]))
        final_h.free()
        final_c.free()
    }
    flat_x.free()

    let all_y = if sequences.len() == 1 {
        sequences[0]
    } else {
        concat_along(sequences, 1, node.label())
    }
    let all_h = if last_h.len() == 1 { last_h[0] } else { concat_along(last_h, 0, node.label()) }
    let all_c = if last_c.len() == 1 { last_c[0] } else { concat_along(last_c, 0, node.label()) }
    if sequences.len() > 1 {
        for part in sequences { part.free() }
        for part in last_h { part.free() }
        for part in last_c { part.free() }
    }
    return [all_y, all_h, all_c]
}

#: One direction of a GRU, from `[seq, batch, input]` to `[seq, batch, H]`.
fn gru_direction(
    x_gates: Tensor,
    recurrent: Tensor,
    recurrent_h: Tensor,
    recurrent_bias: Tensor,
    initial_h: Tensor,
    steps: int,
    batch: int,
    hidden: int,
    clip: float,
    reverse: bool,
    linear_before_reset: bool
) -> tuple[Tensor, Tensor] {
    var state_h = initial_h
    var produced: list[Tensor] = []
    for index in 0..steps {
        let step = if reverse { steps - 1 - index } else { index }
        let recurred = jt.matmul(state_h, recurrent, trans_b: true)
        let offset = step * batch * 3 * hidden
        let next_h = jt.Tensor([batch, hidden])
        if linear_before_reset {
            kernel_for("gru_step_linear").dispatch_async(
                [
                    x_gates.buffer(),
                    recurred.buffer(),
                    recurrent_bias.buffer(),
                    state_h.buffer(),
                    next_h.buffer(),
                ],
                batch * hidden,
                [batch * hidden, hidden, f32_bits(clip), offset]
            )
        } else {
            #: The reset gate scales the hidden state before the weights see
            #: it, so this half stops to take a second product.
            let gate_z = jt.Tensor([batch, hidden])
            let scaled = jt.Tensor([batch, hidden])
            kernel_for("gru_reset").dispatch_async(
                [
                    x_gates.buffer(),
                    recurred.buffer(),
                    recurrent_bias.buffer(),
                    state_h.buffer(),
                    gate_z.buffer(),
                    scaled.buffer(),
                ],
                batch * hidden,
                [batch * hidden, hidden, f32_bits(clip), offset]
            )
            let reset_product = jt.matmul(scaled, recurrent_h, trans_b: true)
            kernel_for("gru_finish").dispatch_async(
                [
                    x_gates.buffer(),
                    reset_product.buffer(),
                    gate_z.buffer(),
                    state_h.buffer(),
                    next_h.buffer(),
                ],
                batch * hidden,
                [batch * hidden, hidden, f32_bits(clip), offset]
            )
            gate_z.free()
            scaled.free()
            reset_product.free()
        }
        recurred.free()
        state_h = next_h
        produced.push(next_h)
    }

    var ordered = produced
    if reverse {
        ordered = []
        for index in 0..produced.len() { ordered.push(produced[produced.len() - 1 - index]) }
    }
    var rows: list[Tensor] = []
    for part in ordered { rows.push(copy_of(part, [1, batch, hidden])) }
    let sequence = if rows.len() == 1 {
        rows[0]
    } else {
        concat_along(rows, 0, "a GRU's output sequence")
    }
    if rows.len() > 1 {
        for row in rows { row.free() }
    }
    let final_h = copy_of(state_h, [batch, hidden])
    for part in produced { part.free() }
    return (sequence, final_h)
}

#: ONNX `GRU`.
#:
#: Weights are `[directions, 3H, input]` and `[directions, 3H, H]` with the
#: gates ordered z, r, h, and the bias `[directions, 6H]` holds the input set
#: then the recurrent one. `linear_before_reset` decides whether the reset gate
#: scales the hidden state before or after the recurrent weights, which are
#: different answers -- and it decides where the recurrent bias for the
#: candidate gate belongs, since only one of the two puts it inside the reset.
fn op_gru(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let x = need(ins, 0, node)
    let w = need(ins, 1, node)
    let r = need(ins, 2, node)
    let bias = maybe(ins, 3)
    if maybe(ins, 4) is not null {
        throw ValueError(
            f"{node.label()} gives per-sequence lengths, which this executor does not read; " +
                "every sequence in the batch has to be the same length"
        )
    }
    if node.attrs.get("activations", null) is not null {
        throw ValueError(
            f"{node.label()} names its own activations; this executor uses the sigmoid and " +
                "tanh the operator defaults to"
        )
    }
    if attr_int(node, "layout", 0) != 0 {
        throw ValueError(
            f"{node.label()} uses the batch-first layout, which this executor does not read"
        )
    }
    if x.rank() != 3 { throw ValueError(f"{node.label()} needs a 3-D input, got {x.shape}") }
    if w.rank() != 3 or r.rank() != 3 {
        throw ValueError(f"{node.label()} needs 3-D weights, got {w.shape} and {r.shape}")
    }
    let steps = x.shape[0]
    let batch = x.shape[1]
    let input_size = x.shape[2]
    let directions = w.shape[0]
    let hidden = attr_int(node, "hidden_size", r.shape[2])
    let facing = attr_str(node, "direction", "forward")
    let wanted = if facing == "bidirectional" { 2 } else { 1 }
    if directions != wanted or r.shape[0] != directions {
        throw ValueError(
            f"{node.label()} is {facing} but carries {directions} directions of weights"
        )
    }
    if w.shape[1] != 3 * hidden or r.shape[1] != 3 * hidden or r.shape[2] != hidden {
        throw ValueError(
            f"{node.label()} has weights {w.shape} and {r.shape} for {hidden} hidden units"
        )
    }
    if w.shape[2] != input_size {
        throw ValueError(f"{node.label()} has weights {w.shape} for {input_size} inputs")
    }
    let clip = attr_float(node, "clip", 0.0)
    let linear = attr_int(node, "linear_before_reset", 0) != 0

    let flat_x = copy_of(x, [steps * batch, input_size])
    var sequences: list[Tensor] = []
    var last_h: list[Tensor] = []
    for direction in 0..directions {
        let wd = taken(slice_axis(w, 0, direction, 1), [3 * hidden, input_size])
        let rd = taken(slice_axis(r, 0, direction, 1), [3 * hidden, hidden])
        let rh = taken(slice_axis(rd, 0, 2 * hidden, hidden), [hidden, hidden])
        var x_gates = jt.matmul(flat_x, wd, trans_b: true)
        wd.free()
        #: Always a real vector, zeros when the node carries none, so the
        #: kernels have one shape of input rather than two.
        var recurrent_bias = jt.Tensor.zeros([3 * hidden])
        if bias is not null {
            let row = taken(slice_axis(bias, 0, direction, 1), [6 * hidden])
            let wb = taken(slice_axis(row, 0, 0, 3 * hidden), [3 * hidden])
            let rb = taken(slice_axis(row, 0, 3 * hidden, 3 * hidden), [3 * hidden])
            #: Without `linear_before_reset` the candidate's recurrent bias
            #: falls outside the reset gate, which makes it a constant like the
            #: input biases -- so it moves to the input half, and the recurrent
            #: half is left carrying only the update and reset gates.
            var input_bias = wb
            var carried = rb
            if not linear {
                let head = taken(slice_axis(rb, 0, 0, 2 * hidden), [2 * hidden])
                let tail = taken(slice_axis(rb, 0, 2 * hidden, hidden), [hidden])
                let pad_tail = jt.Tensor.zeros([hidden])
                let pad_head = jt.Tensor.zeros([2 * hidden])
                carried = concat_along([head, pad_tail], 0, node.label())
                let widened = concat_along([pad_head, tail], 0, node.label())
                input_bias = jt.add(wb, widened)
                head.free()
                tail.free()
                pad_tail.free()
                pad_head.free()
                widened.free()
            }
            let shifted = jt.add(x_gates, input_bias)
            x_gates.free()
            x_gates = shifted
            recurrent_bias.free()
            recurrent_bias = carried
            if input_bias is not wb { input_bias.free() }
            wb.free()
            if carried is not rb { rb.free() }
            row.free()
        }
        let given_h = maybe(ins, 5)
        let start_h = if given_h is null {
            jt.Tensor.zeros([batch, hidden])
        } else {
            taken(slice_axis(given_h, 0, direction, 1), [batch, hidden])
        }
        let (sequence, final_h) = gru_direction(
            x_gates,
            rd,
            rh,
            recurrent_bias,
            start_h,
            steps,
            batch,
            hidden,
            clip,
            facing == "reverse" or direction == 1,
            linear
        )
        x_gates.free()
        rd.free()
        rh.free()
        start_h.free()
        recurrent_bias.free()
        sequences.push(copy_of(sequence, [steps, 1, batch, hidden]))
        sequence.free()
        last_h.push(copy_of(final_h, [1, batch, hidden]))
        final_h.free()
    }
    flat_x.free()

    let all_y = if sequences.len() == 1 {
        sequences[0]
    } else {
        concat_along(sequences, 1, node.label())
    }
    let all_h = if last_h.len() == 1 { last_h[0] } else { concat_along(last_h, 0, node.label()) }
    if sequences.len() > 1 {
        for part in sequences { part.free() }
        for part in last_h { part.free() }
    }
    return [all_y, all_h]
}

#: `copy_of` on a freshly cut slice, freeing the slice. Slicing allocates, and
#: every use of it here wants the copy and not the cut.
fn taken(cut: Tensor, shape: list[int]) -> Tensor {
    let kept = copy_of(cut, shape)
    cut.free()
    return kept
}

fn op_instance_norm(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let x = need(ins, 0, node)
    let scale = need(ins, 1, node)
    let shift = need(ins, 2, node)
    if x.rank() < 3 {
        throw ValueError(f"{node.label()} needs at least a 3-D NCHW input, got {x.shape}")
    }
    let channels = x.shape[1]
    if scale.size != channels or shift.size != channels {
        throw ValueError(f"{node.label()} has parameters for {scale.size} of {channels} channels")
    }
    let planes = x.shape[0] * channels
    let spatial = x.size // planes
    let mean = Tensor([planes])
    let variance = Tensor([planes])
    kernel_for("channel_moments_nchw").dispatch_async(
        [x.buffer(), mean.buffer(), variance.buffer()],
        planes,
        [planes, spatial]
    )
    let output = Tensor(x.shape)
    kernel_for("affine_norm_nchw").dispatch_async(
        [
            x.buffer(),
            mean.buffer(),
            variance.buffer(),
            scale.buffer(),
            shift.buffer(),
            output.buffer(),
        ],
        x.size,
        [x.size, channels, spatial, f32_bits(attr_float(node, "epsilon", 1e-5)), 1]
    )
    mean.free()
    variance.free()
    return [output]
}

fn op_lrn(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let x = need(ins, 0, node)
    if x.rank() < 3 {
        throw ValueError(f"{node.label()} needs at least a 3-D NCHW input, got {x.shape}")
    }
    let size = attr_int(node, "size", 0)
    if size <= 0 { throw ValueError(f"{node.label()} needs a positive size, got {size}") }
    let channels = x.shape[1]
    let output = Tensor(x.shape)
    kernel_for("lrn_nchw").dispatch_async(
        [x.buffer(), output.buffer()],
        x.size,
        [
            x.size,
            channels,
            x.size // (x.shape[0] * channels),
            size,
            f32_bits(attr_float(node, "alpha", 0.0001)),
            f32_bits(attr_float(node, "beta", 0.75)),
            f32_bits(attr_float(node, "bias", 1.0)),
        ]
    )
    return [output]
}

fn op_gemm(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let a = need(ins, 0, node)
    let b = need(ins, 1, node)
    if a.rank() != 2 or b.rank() != 2 {
        throw ValueError(f"{node.label()} needs rank-two operands, got {a.shape} and {b.shape}")
    }
    let alpha = attr_float(node, "alpha", 1.0)
    let beta = attr_float(node, "beta", 1.0)
    let product = jt.matmul(
        a,
        b,
        trans_a: attr_int(node, "transA", 0) != 0,
        trans_b: attr_int(node, "transB", 0) != 0
    )
    var output = product
    if alpha != 1.0 {
        output = jt.scale(product, alpha)
        product.free()
    }
    let c = maybe(ins, 2)
    if c is null or beta == 0.0 { return [output] }
    let stretched = if c.shape == output.shape { c } else { expand_to(c, output.shape) }
    let weighted = if beta == 1.0 { stretched } else { jt.scale(stretched, beta) }
    let summed = jt.add(output, weighted)
    if weighted is not stretched { weighted.free() }
    if stretched is not c { stretched.free() }
    output.free()
    return [summed]
}

#: One `[rows, inner] x [inner, columns]` product per page of the broadcast
#: batch, written straight into a window of the output.
fn batched_matmul(
    left: Tensor,
    right: Tensor,
    batch: list[int],
    rows: int,
    inner: int,
    columns: int
) -> Tensor {
    var left_shape = [dimension for dimension in batch]
    left_shape.push(rows)
    left_shape.push(inner)
    var right_shape = [dimension for dimension in batch]
    right_shape.push(inner)
    right_shape.push(columns)
    var dims = [dimension for dimension in batch]
    dims.push(rows)
    dims.push(columns)
    let spread_left = if left.shape == left_shape { left } else { expand_to(left, left_shape) }
    let spread_right = if right.shape == right_shape {
        right
    } else {
        expand_to(right, right_shape)
    }
    let output = Tensor(dims)
    for index in 0..volume(batch) {
        let page_a = Tensor(
            [rows, inner],
            storage: spread_left.buffer().view(index * rows * inner, rows * inner)
        )
        let page_b = Tensor(
            [inner, columns],
            storage: spread_right.buffer().view(index * inner * columns, inner * columns)
        )
        let page_out = Tensor(
            [rows, columns],
            storage: output.buffer().view(index * rows * columns, rows * columns)
        )
        jt.matmul(page_a, page_b, out: page_out)
        page_a.free()
        page_b.free()
        page_out.free()
    }
    if spread_left is not left { spread_left.free() }
    if spread_right is not right { spread_right.free() }
    return output
}

fn op_matmul(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let a = need(ins, 0, node)
    let b = need(ins, 1, node)
    #: ONNX promotes a vector operand to a matrix for the product and drops
    #: the axis again afterwards, so both shapes are normalised here and the
    #: result is trimmed at the end.
    let left = if a.rank() == 1 { copy_of(a, [1, a.size]) } else { a }
    let right = if b.rank() == 1 { copy_of(b, [b.size, 1]) } else { b }
    let rows = left.shape[left.rank() - 2]
    let inner = left.shape[left.rank() - 1]
    let columns = right.shape[right.rank() - 1]
    if right.shape[right.rank() - 2] != inner {
        throw ValueError(f"{node.label()} cannot multiply {a.shape} by {b.shape}")
    }
    var batch: list[int] = []
    if left.rank() > 2 or right.rank() > 2 {
        batch = broadcast_shape(
            [left.shape[axis] for axis in 0..left.rank() - 2],
            [right.shape[axis] for axis in 0..right.rank() - 2],
            node.label()
        )
    }
    let product = if batch.len() == 0 {
        jt.matmul(left, right)
    } else {
        batched_matmul(left, right, batch, rows, inner, columns)
    }
    if left is not a { left.free() }
    if right is not b { right.free() }
    if a.rank() != 1 and b.rank() != 1 { return [product] }
    var trimmed = [dimension for dimension in batch]
    if a.rank() != 1 { trimmed.push(rows) }
    if b.rank() != 1 { trimmed.push(columns) }
    let squeezed = copy_of(product, shaped(trimmed))
    product.free()
    return [squeezed]
}

#: Reduce `axes` away, by moving them to the end and reducing the rows that
#: leaves. `op` is `row_reduce`'s: sum, mean, max, min, or an extreme's index.
fn reduce_over(x: Tensor, axes: list[int], keepdims: bool, op: int) -> Tensor {
    let rank = x.rank()
    var kept: list[int] = []
    for axis in 0..rank {
        if not axis in axes { kept.push(axis) }
    }
    var perm = [axis for axis in kept]
    for axis in axes { perm.push(axis) }
    var ordered = true
    for (slot, axis) in perm.enumerate() {
        if slot != axis { ordered = false }
    }
    let moved = if ordered { x } else { transpose_to(x, perm) }
    var inner = 1
    for axis in axes { inner *= x.shape[axis] }
    var dims: list[int] = []
    if keepdims {
        for axis in 0..rank { dims.push(if axis in axes { 1 } else { x.shape[axis] }) }
    } else {
        for axis in kept { dims.push(x.shape[axis]) }
    }
    let output = row_reduce_of(moved, x.size // inner, inner, op, shape: shaped(dims))
    if moved is not x { moved.free() }
    return output
}

fn sorted_axes(node: GraphNode, asked: list[int], rank: int) -> list[int] {
    var resolved: list[int] = []
    for axis in asked {
        let normalized = normalize_axis(axis, rank, node.label())
        if not normalized in resolved { resolved.push(normalized) }
    }
    resolved.sort()
    return resolved
}

fn op_reduce(node: GraphNode, ins: list[Tensor?], op: int) -> list[Tensor] {
    let x = need(ins, 0, node)
    let asked = ints_from(node, ins, 1, "axes") ?? []
    if asked.len() == 0 and attr_int(node, "noop_with_empty_axes", 0) != 0 {
        return [copy_of(x, x.shape)]
    }
    let wanted = if asked.len() == 0 { [axis for axis in 0..x.rank()] } else { asked }
    let axes = sorted_axes(node, wanted, x.rank())
    return [reduce_over(x, axes, attr_int(node, "keepdims", 1) != 0, op)]
}

fn op_reduce_sum(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return op_reduce(node, ins, 0)
}

fn op_reduce_mean(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return op_reduce(node, ins, 1)
}

fn op_reduce_max(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return op_reduce(node, ins, 2)
}

fn op_reduce_min(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return op_reduce(node, ins, 3)
}

fn op_arg_extreme(node: GraphNode, ins: list[Tensor?], want_max: bool) -> list[Tensor] {
    let x = need(ins, 0, node)
    let axis = normalize_axis(attr_int(node, "axis", 0), x.rank(), node.label())
    var op = if want_max { 4 } else { 5 }
    if attr_int(node, "select_last_index", 0) != 0 { op += 2 }
    return [reduce_over(x, [axis], attr_int(node, "keepdims", 1) != 0, op)]
}

fn op_argmax(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return op_arg_extreme(node, ins, true)
}

fn op_argmin(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return op_arg_extreme(node, ins, false)
}

fn softmax_like(node: GraphNode, ins: list[Tensor?], want_log: bool) -> list[Tensor] {
    let x = need(ins, 0, node)
    let rank = x.rank()
    let axis = normalize_axis(attr_int(node, "axis", -1), rank, node.label())
    let columns = x.shape[axis]
    if axis == rank - 1 { return [softmax_rows(x, x.size // columns, columns, want_log)] }
    var perm: list[int] = []
    for other in 0..rank {
        if other != axis { perm.push(other) }
    }
    perm.push(axis)
    let moved = transpose_to(x, perm)
    let normalized = softmax_rows(moved, x.size // columns, columns, want_log)
    moved.free()
    var back = [0 for _axis in 0..rank]
    for (slot, other) in perm.enumerate() { back[other] = slot }
    let output = transpose_to(normalized, back)
    normalized.free()
    return [output]
}

fn op_softmax(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return softmax_like(node, ins, false)
}

fn op_log_softmax(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return softmax_like(node, ins, true)
}

fn resample(node: GraphNode, x: Tensor, height: int, width: int, coord: int, rule: int) -> Tensor {
    if x.rank() != 4 { throw ValueError(f"{node.label()} needs a 4-D NCHW input, got {x.shape}") }
    let name = attr_str(node, "mode", "nearest")
    let mode = match name {
        "nearest" => 0,
        "linear" => 1,
        "bilinear" => 1,
        _ => throw ValueError(f"{node.label()} asks for {name} resampling, which is not implemented"),
    }
    if height <= 0 or width <= 0 {
        throw ValueError(f"{node.label()} resamples {x.shape} to {height}x{width}")
    }
    let output = Tensor([x.shape[0], x.shape[1], height, width])
    kernel_for("resize_nchw").dispatch_async(
        [x.buffer(), output.buffer()],
        output.size,
        [output.size, x.shape[2], x.shape[3], height, width, mode, coord, rule]
    )
    return output
}

fn scaled_extent(node: GraphNode, x: Tensor, scales: list[float]) -> tuple[int, int] {
    if scales.len() != 4 {
        throw ValueError(f"{node.label()} needs one scale an axis, got {scales.len()}")
    }
    if scales[0] != 1.0 or scales[1] != 1.0 {
        throw ValueError(
            f"{node.label()} scales batch and channels by {scales[0]} and {scales[1]}, and " +
                "this executor only resamples the spatial axes"
        )
    }
    return (int(float(x.shape[2]) * scales[2]), int(float(x.shape[3]) * scales[3]))
}

fn op_resize(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let x = need(ins, 0, node)
    let rounding = attr_str(node, "nearest_mode", "round_prefer_floor")
    let rule = match rounding {
        "round_prefer_floor" => 0,
        "round_prefer_ceil" => 1,
        "floor" => 2,
        "ceil" => 3,
        _ => throw ValueError(f"{node.label()} asks for nearest_mode {rounding}, which is not implemented"),
    }
    let transform = attr_str(node, "coordinate_transformation_mode", "half_pixel")
    let coord = match transform {
        "half_pixel" => 0,
        "asymmetric" => 1,
        "align_corners" => 2,
        "pytorch_half_pixel" => 3,
        _ => throw ValueError(f"{node.label()} asks for {transform} coordinates, which is not implemented"),
    }
    #: Resize-10 took only `(X, scales)`; every version since puts `roi`
    #: second, so a two-input node is the old shape and anything longer is
    #: the new one.
    let scales_at = if ins.len() == 2 { 1 } else { 2 }
    let sizes = maybe(ins, 3)
    if sizes is not null {
        let dims = host_ints(sizes)
        if dims.len() != 4 {
            throw ValueError(f"{node.label()} needs one size an axis, got {dims.len()}")
        }
        if dims[0] != x.shape[0] or dims[1] != x.shape[1] {
            throw ValueError(
                f"{node.label()} resizes {x.shape} to {dims}, changing batch or channels"
            )
        }
        return [resample(node, x, dims[2], dims[3], coord, rule)]
    }
    let scales = maybe(ins, scales_at)
    if scales is null { throw ValueError(f"{node.label()} needs either scales or sizes") }
    let (height, width) = scaled_extent(node, x, scales.download())
    return [resample(node, x, height, width, coord, rule)]
}

fn op_upsample(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let x = need(ins, 0, node)
    let supplied = maybe(ins, 1)
    let scales = if supplied is not null {
        supplied.download()
    } else {
        attr_floats(node, "scales", [])
    }
    let (height, width) = scaled_extent(node, x, scales)
    #: Upsample predates both `coordinate_transformation_mode` and
    #: `nearest_mode`: it always read the source coordinate as `out / scale`
    #: and took the floor of it.
    return [resample(node, x, height, width, 1, 2)]
}

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
