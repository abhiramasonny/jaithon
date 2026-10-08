#: The operators that move values around rather than compute with them:
#: reshaping, slicing, joining, padding, and the ones that produce a shape or
#: a constant outright.
from jaitensor import Tensor

from .kernels import kernel_for
from .nodes import Attr, GraphNode, attr_float, attr_int, attr_ints, attr_str, ints_from, maybe
from .nodes import need
from .tensors import broadcast_shape, concat_along, contiguous_strides, copy_of, expand_to
from .tensors import gather_strided, host_ints
from .tensors import normalize_axis, pad_nd_of, shaped, slice_axis, transpose_to, unary_ex_of

pub fn op_reshape(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
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

pub fn op_flatten(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
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

pub fn op_squeeze(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
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

pub fn op_unsqueeze(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
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

pub fn op_transpose(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
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

pub fn op_concat(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    if ins.len() == 0 { throw ValueError(f"{node.label()} needs at least one input") }
    var parts: list[Tensor] = []
    for index in 0..ins.len() { parts.push(need(ins, index, node)) }
    let axis = normalize_axis(attr_int(node, "axis", 0), parts[0].rank(), node.label())
    return [concat_along(parts, axis, node.label())]
}

pub fn op_split(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
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

pub fn op_slice(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
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

#: The attribute an importer puts on a `Gather` whose indices the file wrote
#: as a rank-zero scalar. jaitensor has no rank-zero tensor, so the index
#: arrives as `[1]` either way, and this is what says the axis should go.
pub let RANK_ZERO_INDICES = "jaicv_rank_zero_indices"

#: Mark every `Gather` indexed by one of `scalars` -- the names a file wrote
#: with no dimensions -- so that it drops the axis it picks from, as ONNX
#: says. Without it `x[:, 0]` keeps a width of one and the `Gemm` after a
#: ViT's class token refuses a rank-three operand.
pub fn mark_rank_zero(nodes: list[GraphNode], scalars: dict[str, bool]) -> void {
    for node in nodes {
        if node.op != "Gather" or node.inputs.len() < 2 { continue }
        if scalars.get(node.inputs[1], false) {
            node.attrs[RANK_ZERO_INDICES] = Attr.of_int(1)
        }
    }
}

pub fn op_gather(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let data = need(ins, 0, node)
    let indices = need(ins, 1, node)
    let axis = normalize_axis(attr_int(node, "axis", 0), data.rank(), node.label())
    var inner = 1
    for other in axis + 1..data.rank() { inner *= data.shape[other] }
    var dims: list[int] = []
    for other in 0..axis { dims.push(data.shape[other]) }
    #: A rank-zero index drops the axis; a vector of them replaces it. The
    #: importer says which, since both arrive as a tensor of shape `[1]`. A
    #: scalar picked out of a vector is `[1]` again, which is what a scalar is
    #: here.
    let dropped = indices.size == 1 and attr_int(node, RANK_ZERO_INDICES, 0) == 1
    if not dropped {
        for dimension in indices.shape { dims.push(dimension) }
    }
    for other in axis + 1..data.rank() { dims.push(data.shape[other]) }
    let output = Tensor(shaped(dims))
    kernel_for("gather_axis").dispatch_async(
        [data.buffer(), indices.buffer(), output.buffer()],
        output.size,
        [output.size, data.shape[axis], inner, indices.size]
    )
    return [output]
}

#: `data` stretched to the shape its second input names -- by numpy's rules,
#: and in both directions: a `1` in the target keeps the axis the data has,
#: and the data's own width-one axes are the ones repeated, so the result is
#: the broadcast of the two shapes rather than the target alone.
pub fn op_expand(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let data = need(ins, 0, node)
    let target = host_ints(need(ins, 1, node))
    for dimension in target {
        if dimension < 0 {
            throw ValueError(f"{node.label()} expands to {target}, which has a negative width")
        }
        if dimension == 0 {
            throw ValueError(
                f"{node.label()} expands to {target}, and an empty tensor has no place here"
            )
        }
    }
    let shape = broadcast_shape(data.shape, target, node.label())
    if shape == data.shape { return [copy_of(data, shape)] }
    return [expand_to(data, shape)]
}

pub fn op_pad(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
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

pub fn op_shape(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
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

pub fn op_cast(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    let data = need(ins, 0, node)
    let target = attr_int(node, "to", 1)
    if target in FLOAT_TYPES { return [copy_of(data, data.shape)] }
    if target in INTEGER_TYPES { return [unary_ex_of(data, 6)] }
    if target == 9 { return [unary_ex_of(data, 7)] }
    throw ValueError(f"{node.label()} casts to ONNX type {target}, which is not implemented")
}

pub fn op_constant(node: GraphNode, _ins: list[Tensor?]) -> list[Tensor] {
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

pub fn op_constant_of_shape(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
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
