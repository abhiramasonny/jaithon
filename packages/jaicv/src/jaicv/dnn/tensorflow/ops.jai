#: One TensorFlow operator at a time, translated into the shared IR.
from ..graph import Attr
from .graphdef import TfNode, attr_bool, attr_float, attr_int, attr_ints, attr_string, attr_type
from .importer import Importer
from .layout import NCHW_AXIS, TF_AXIS, lands_in_nhwc, permuted, survives_in_order
from .tensors import DTYPE_NAMES, DT_FLOAT, ONNX_DTYPES, TfTensor

#: TensorFlow writes a per-axis attribute in the data format's own order, so
#: `NHWC` strides arrive as `[1, sh, sw, 1]`. A window that moves over the
#: batch or the channels is not something the executor's operators can
#: express, and quietly dropping the outer two would be the wrong answer
#: rather than an unsupported one.
fn spatial_pair(node: TfNode, name: str, fallback: list[int]) -> list[int] {
    let values = attr_ints(node, name)
    if values is null { return fallback }
    let given: list[int] = values
    if given.len() == 0 { return fallback }
    if given.len() != 4 {
        throw ValueError(
            f"{node.label()} has {name} {given}, and this importer reads two spatial axes"
        )
    }
    if given[0] != 1 or given[3] != 1 {
        throw ValueError(f"{node.label()} has {name} {given}, which steps over batch or channels")
    }
    return [given[1], given[2]]
}

fn require_nhwc(node: TfNode) -> void {
    let format = attr_string(node, "data_format", "NHWC")
    if format != "NHWC" {
        throw ValueError(
            f"{node.label()} is laid out {format}, and this importer reads the NHWC graphs " +
                "TensorFlow freezes on a host without a GPU"
        )
    }
}

#: TensorFlow's `SAME` divides the padding with the remainder at the end,
#: which is exactly what ONNX calls `SAME_UPPER`. `SAME_LOWER` would put it at
#: the front and quietly shift the whole feature map by a pixel.
fn add_padding(node: TfNode, attrs: dict[str, Attr]) -> void {
    let padding = attr_string(node, "padding", "VALID")
    if padding == "SAME" {
        attrs["auto_pad"] = Attr.of_str("SAME_UPPER")
        return
    }
    if padding == "VALID" {
        attrs["auto_pad"] = Attr.of_str("VALID")
        return
    }
    if padding == "EXPLICIT" {
        let given = attr_ints(node, "explicit_paddings") ?? []
        if given.len() != 8 {
            throw ValueError(f"{node.label()} has {given.len()} explicit paddings, expected eight")
        }
        #: `[batch, batch, top, bottom, left, right, channel, channel]` in
        #: `NHWC`, and ONNX wants `[top, left, bottom, right]`.
        attrs["pads"] = Attr.of_ints([given[2], given[4], given[3], given[5]])
        return
    }
    throw ValueError(f"{node.label()} pads {padding}, which is not SAME, VALID or EXPLICIT")
}

fn normalized_axes(asked: list[int], rank: int, node: TfNode, what: str) -> list[int] {
    var out: list[int] = []
    for axis in asked {
        let resolved = if axis < 0 { axis + rank } else { axis }
        if resolved < 0 or resolved >= rank {
            throw ValueError(f"{node.label()} names {what} {axis} of a rank-{rank} tensor")
        }
        if not resolved in out { out.push(resolved) }
    }
    return out
}

pub fn translate_conv(im: Importer, node: TfNode, depthwise: bool) -> void {
    require_nhwc(node)
    if node.inputs.len() < 2 { throw ValueError(f"{node.label()} needs an input and a filter") }
    let held = im.const_of(node.inputs[1])
    if held is null {
        throw ValueError(
            f"{node.label()} takes its filter from {node.inputs[1]}, which is not a constant -- " +
                "this importer reads frozen weights only"
        )
    }
    let filter: TfTensor = held
    if filter.rank() != 4 {
        throw ValueError(f"{node.label()} has a rank-{filter.rank()} filter, expected four")
    }
    if im.rank_of(node.inputs[0], node) != 4 {
        throw ValueError(f"{node.label()} convolves a tensor that is not rank four")
    }
    let x = im.value_input(node.inputs[0], node)
    let kh = filter.shape[0]
    let kw = filter.shape[1]
    var attrs: dict[str, Attr] = {
        "kernel_shape": Attr.of_ints([kh, kw]),
        "strides": Attr.of_ints(spatial_pair(node, "strides", [1, 1])),
        "dilations": Attr.of_ints(spatial_pair(node, "dilations", [1, 1])),
    }
    add_padding(node, attrs)
    var weight = ""
    if depthwise {
        #: A depthwise filter is `[kH, kW, C, M]` and produces `C * M`
        #: channels, output `c * M + m` coming from input channel `c`. A
        #: grouped convolution with `C` groups numbers its outputs the same
        #: way, so the weights only have to be laid out `[C, M, kH, kW]` and
        #: read as `[C * M, 1, kH, kW]`.
        let channels = filter.shape[2]
        let multiplier = filter.shape[3]
        attrs["group"] = Attr.of_int(channels)
        weight = im.intern(
            node.inputs[1] + "$depthwise",
            [channels * multiplier, 1, kh, kw],
            permuted(filter.values(), filter.shape, [2, 3, 0, 1])
        )
    } else {
        #: `[kH, kW, I, O]` to ONNX's `[O, I, kH, kW]`.
        attrs["group"] = Attr.of_int(1)
        weight = im.intern(
            node.inputs[1] + "$oihw",
            [filter.shape[3], filter.shape[2], kh, kw],
            permuted(filter.values(), filter.shape, [3, 2, 0, 1])
        )
    }
    let _emitted = im.emit("Conv", node.name, [x, weight], attrs)
    im.ranks[node.name] = 4
}

pub fn translate_pool(im: Importer, node: TfNode, is_max: bool) -> void {
    require_nhwc(node)
    if node.inputs.len() < 1 { throw ValueError(f"{node.label()} needs an input") }
    if im.rank_of(node.inputs[0], node) != 4 {
        throw ValueError(f"{node.label()} pools a tensor that is not rank four")
    }
    let x = im.value_input(node.inputs[0], node)
    var attrs: dict[str, Attr] = {
        "kernel_shape": Attr.of_ints(spatial_pair(node, "ksize", [1, 1])),
        "strides": Attr.of_ints(spatial_pair(node, "strides", [1, 1])),
    }
    add_padding(node, attrs)
    #: TensorFlow's average pool divides by the number of real values in the
    #: window, never by the window's area, which is the executor's default.
    let _emitted = im.emit(if is_max { "MaxPool" } else { "AveragePool" }, node.name, [x], attrs)
    im.ranks[node.name] = 4
}

pub fn translate_batch_norm(im: Importer, node: TfNode) -> void {
    require_nhwc(node)
    #: The attribute defaults to true, so a file that does not say is a file
    #: that has not been frozen for inference.
    if attr_bool(node, "is_training", true) {
        throw ValueError(
            f"{node.label()} is still in training mode, so the file carries no running mean " +
                "or variance to normalise with"
        )
    }
    if node.inputs.len() < 5 {
        throw ValueError(
            f"{node.label()} needs an input, a scale, an offset, a mean and a variance"
        )
    }
    if im.rank_of(node.inputs[0], node) != 4 {
        throw ValueError(f"{node.label()} normalises a tensor that is not rank four")
    }
    var ins: list[str] = []
    for slot in 0..5 { ins.push(im.value_input(node.inputs[slot], node)) }
    let _emitted = im.emit(
        "BatchNormalization",
        node.name,
        ins,
        {"epsilon": Attr.of_float(attr_float(node, "epsilon", 0.0001))}
    )
    im.ranks[node.name] = 4
}

pub fn translate_matmul(im: Importer, node: TfNode) -> void {
    if node.inputs.len() < 2 { throw ValueError(f"{node.label()} needs two operands") }
    for slot in 0..2 {
        if im.rank_of(node.inputs[slot], node) != 2 {
            throw ValueError(f"{node.label()} multiplies a tensor that is not rank two")
        }
    }
    let a = im.value_input(node.inputs[0], node)
    let b = im.value_input(node.inputs[1], node)
    let _emitted = im.emit("Gemm", node.name, [a, b], {"transA": Attr.of_int(if attr_bool(node, "transpose_a", false) {
        1
    } else {
        0
    }), "transB": Attr.of_int(if attr_bool(node, "transpose_b", false) {
        1
    } else {
        0
    })})
    im.ranks[node.name] = 2
}

pub fn translate_elementwise(im: Importer, node: TfNode, op: str) -> void {
    #: `BiasAdd` is an `Add` whose bias is declared to sit on the channels, so
    #: its data format has to agree with the one the rest of the graph is in.
    if node.op == "BiasAdd" or node.op == "BiasAddV1" { require_nhwc(node) }
    var want = 0
    for name in node.inputs { want = max(want, im.rank_of(name, node)) }
    var ins: list[str] = []
    for name in node.inputs { ins.push(im.broadcast_input(name, node, want)) }
    let _emitted = im.emit(op, node.name, ins)
    im.ranks[node.name] = want
}

pub fn translate_unary(im: Importer, node: TfNode, op: str, attrs: dict[str, Attr] = {}) -> void {
    if node.inputs.len() < 1 { throw ValueError(f"{node.label()} needs an input") }
    let x = im.value_input(node.inputs[0], node)
    let _emitted = im.emit(op, node.name, [x], attrs)
    im.ranks[node.name] = im.loose_rank(node.inputs[0])
}

pub fn translate_softmax(im: Importer, node: TfNode, op: str) -> void {
    if node.inputs.len() < 1 { throw ValueError(f"{node.label()} needs an input") }
    let rank = im.rank_of(node.inputs[0], node)
    #: TensorFlow normalises the last axis, which is the channels, and the
    #: channels are axis one here.
    let axis = if rank == 4 { 1 } else { -1 }
    let x = im.value_input(node.inputs[0], node)
    let _emitted = im.emit(op, node.name, [x], {"axis": Attr.of_int(axis)})
    im.ranks[node.name] = rank
}

pub fn translate_lrn(im: Importer, node: TfNode) -> void {
    if node.inputs.len() < 1 { throw ValueError(f"{node.label()} needs an input") }
    if im.rank_of(node.inputs[0], node) != 4 {
        throw ValueError(f"{node.label()} normalises a tensor that is not rank four")
    }
    let radius = attr_int(node, "depth_radius", 5)
    if radius < 0 { throw ValueError(f"{node.label()} has a depth radius of {radius}") }
    let size = 2 * radius + 1
    #: ONNX divides the squared sum by the window size and TensorFlow does
    #: not, so the alpha that reproduces the file is the file's times the size.
    let x = im.value_input(node.inputs[0], node)
    let _emitted = im.emit(
        "LRN",
        node.name,
        [x],
        {
            "size": Attr.of_int(size),
            "alpha": Attr.of_float(attr_float(node, "alpha", 1.0) * float(size)),
            "beta": Attr.of_float(attr_float(node, "beta", 0.5)),
            "bias": Attr.of_float(attr_float(node, "bias", 1.0)),
        }
    )
    im.ranks[node.name] = 4
}

pub fn translate_reduce(im: Importer, node: TfNode, op: str) -> void {
    if node.inputs.len() < 2 {
        throw ValueError(f"{node.label()} needs a value and reduction indices")
    }
    let rank = im.rank_of(node.inputs[0], node)
    let axes = normalized_axes(
        im.const_ints(node.inputs[1], node, "reduction indices"),
        rank,
        node,
        "the axis"
    )
    let keep = attr_bool(node, "keep_dims", false)
    let x = im.value_input(node.inputs[0], node)
    #: Reducing no axes is the identity in TensorFlow, where in ONNX an empty
    #: axis list means every axis unless it is told otherwise.
    if axes.len() == 0 {
        let _same = im.emit("Identity", node.name, [x])
        im.ranks[node.name] = rank
        return
    }
    if rank != 4 {
        let final_rank = if keep { rank } else { rank - axes.len() }
        #: A reduction that keeps its axes never changes rank, so only the
        #: dropping form can land at rank four out of a starting rank that
        #: was not -- and when it does, the result is still TensorFlow's own
        #: order and has to be turned into NCHW, per `lands_in_nhwc`.
        let grown = lands_in_nhwc(rank, final_rank)
        let inner = if grown { node.name + "$nhwc" } else { node.name }
        let _reduced = im.emit(op, inner, [x], {"axes": Attr.of_ints(axes), "keepdims": Attr.of_int(if keep {
            1
        } else {
            0
        }), "noop_with_empty_axes": Attr.of_int(1)})
        if grown { let _back = im.to_nchw(inner, node.name) }
        im.ranks[node.name] = final_rank
        return
    }
    let mapped = [NCHW_AXIS[axis] for axis in axes]
    if keep {
        #: Averaging both spatial axes and keeping them is a global average
        #: pool, which is the name the rest of the vision world knows it by.
        var spatial = [axis for axis in axes]
        spatial.sort()
        if op == "ReduceMean" and spatial == [1, 2] {
            let _pooled = im.emit("GlobalAveragePool", node.name, [x])
        } else {
            let _reduced = im.emit(
                op,
                node.name,
                [x],
                {
                    "axes": Attr.of_ints(mapped),
                    "keepdims": Attr.of_int(1),
                    "noop_with_empty_axes": Attr.of_int(1),
                }
            )
        }
        im.ranks[node.name] = 4
        return
    }
    #: Dropping the axes can leave what is left in a different order here than
    #: it leaves it in TensorFlow, and when it does the tensor has to be put
    #: back into TensorFlow's order before the reduction rather than after.
    var source = x
    var wanted = mapped
    if not survives_in_order(axes) {
        source = im.to_nhwc(x, node.name + "$nhwc")
        wanted = axes
    }
    let _reduced = im.emit(
        op,
        node.name,
        [source],
        {
            "axes": Attr.of_ints(wanted),
            "keepdims": Attr.of_int(0),
            "noop_with_empty_axes": Attr.of_int(1),
        }
    )
    im.ranks[node.name] = rank - axes.len()
}

pub fn translate_arg_extreme(im: Importer, node: TfNode, op: str) -> void {
    if node.inputs.len() < 2 { throw ValueError(f"{node.label()} needs a value and a dimension") }
    let rank = im.rank_of(node.inputs[0], node)
    let asked = im.const_ints(node.inputs[1], node, "dimension")
    if asked.len() != 1 {
        throw ValueError(f"{node.label()} needs one dimension, got {asked.len()}")
    }
    let axes = normalized_axes(asked, rank, node, "the axis")
    let x = im.value_input(node.inputs[0], node)
    var source = x
    var axis = axes[0]
    if rank == 4 {
        if survives_in_order(axes) {
            axis = NCHW_AXIS[axes[0]]
        } else {
            source = im.to_nhwc(x, node.name + "$nhwc")
        }
    }
    #: `ArgMax`/`ArgMin` always drop exactly the one axis asked for, so a
    #: starting rank other than four can land at rank four the same way a
    #: dropping reduction can, and needs the same NCHW conversion.
    let final_rank = rank - 1
    let grown = lands_in_nhwc(rank, final_rank)
    let inner = if grown { node.name + "$nhwc" } else { node.name }
    let _found = im.emit(
        op,
        inner,
        [source],
        {"axis": Attr.of_int(axis), "keepdims": Attr.of_int(0)}
    )
    if grown { let _back = im.to_nchw(inner, node.name) }
    im.ranks[node.name] = final_rank
}

pub fn translate_squeeze(im: Importer, node: TfNode) -> void {
    if node.inputs.len() < 1 { throw ValueError(f"{node.label()} needs an input") }
    let rank = im.rank_of(node.inputs[0], node)
    let asked = attr_ints(node, "squeeze_dims") ?? []
    let axes = normalized_axes(asked, rank, node, "the axis")
    let x = im.value_input(node.inputs[0], node)
    if axes.len() == 0 {
        if rank == 4 {
            throw ValueError(
                f"{node.label()} names no squeeze_dims, so which axes it drops depends on the " +
                    "shape at run time -- and this importer needs to know whether what is left " +
                    "is still an image"
            )
        }
        let _squeezed = im.emit("Squeeze", node.name, [x])
        im.ranks[node.name] = -1
        return
    }
    let final_rank = rank - axes.len()
    if rank == 4 {
        var source = x
        var wanted = axes
        if survives_in_order(axes) {
            wanted = [NCHW_AXIS[axis] for axis in axes]
        } else {
            source = im.to_nhwc(x, node.name + "$nhwc")
        }
        let _squeezed = im.emit("Squeeze", node.name, [source], {"axes": Attr.of_ints(wanted)})
        im.ranks[node.name] = final_rank
        return
    }
    #: The input was already exactly TensorFlow's own order, and a squeeze
    #: only drops axes -- it never reorders the survivors -- so a result that
    #: lands at rank four here is still that same TensorFlow order and has to
    #: be turned into NCHW before it is tagged rank four, per `lands_in_nhwc`.
    let grown = lands_in_nhwc(rank, final_rank)
    let inner = if grown { node.name + "$nhwc" } else { node.name }
    let _squeezed = im.emit("Squeeze", inner, [x], {"axes": Attr.of_ints(axes)})
    if grown { let _back = im.to_nchw(inner, node.name) }
    im.ranks[node.name] = final_rank
}

pub fn translate_expand_dims(im: Importer, node: TfNode) -> void {
    if node.inputs.len() < 2 { throw ValueError(f"{node.label()} needs a value and an axis") }
    let rank = im.rank_of(node.inputs[0], node)
    let asked = im.const_ints(node.inputs[1], node, "axis")
    if asked.len() != 1 { throw ValueError(f"{node.label()} needs one axis, got {asked.len()}") }
    let grown = rank + 1
    var axis = asked[0]
    if axis < 0 { axis += grown }
    if axis < 0 or axis >= grown {
        throw ValueError(f"{node.label()} inserts axis {asked[0]} into a rank-{grown} result")
    }
    let x = im.value_input(node.inputs[0], node)
    #: A TensorFlow scalar is already carried as `[1]`, so growing it to a
    #: one-element vector is what the executor has been holding all along.
    if rank == 0 {
        let _same = im.emit("Identity", node.name, [x])
        im.ranks[node.name] = 1
        return
    }
    if rank == 4 {
        let flat = im.to_nhwc(x, node.name + "$nhwc")
        let _wider = im.emit("Unsqueeze", node.name, [flat], {"axes": Attr.of_ints([axis])})
        im.ranks[node.name] = 5
        return
    }
    if grown == 4 {
        let wider = im.emit("Unsqueeze", node.name + "$grown", [x], {"axes": Attr.of_ints([axis])})
        let _back = im.to_nchw(wider, node.name)
        im.ranks[node.name] = 4
        return
    }
    let _wider = im.emit("Unsqueeze", node.name, [x], {"axes": Attr.of_ints([axis])})
    im.ranks[node.name] = grown
}

pub fn translate_transpose(im: Importer, node: TfNode) -> void {
    if node.inputs.len() < 2 {
        throw ValueError(f"{node.label()} needs a value and a permutation")
    }
    let rank = im.rank_of(node.inputs[0], node)
    let asked = im.const_ints(node.inputs[1], node, "permutation")
    let perm = normalized_axes(asked, rank, node, "the axis")
    if perm.len() != rank {
        throw ValueError(f"{node.label()} permutes {perm.len()} axes of a rank-{rank} tensor")
    }
    let x = im.value_input(node.inputs[0], node)
    if rank != 4 {
        let _moved = im.emit("Transpose", node.name, [x], {"perm": Attr.of_ints(perm)})
        im.ranks[node.name] = rank
        return
    }
    #: Going out to `NHWC`, permuting, and coming back is one permutation
    #: composed of three: result axis `k` holds TensorFlow's axis `TF_AXIS[k]`
    #: of the output, which is TensorFlow's `perm[TF_AXIS[k]]` of the input,
    #: which is this tensor's axis `NCHW_AXIS[perm[TF_AXIS[k]]]`.
    let composed = [NCHW_AXIS[perm[TF_AXIS[slot]]] for slot in 0..4]
    let _moved = im.emit("Transpose", node.name, [x], {"perm": Attr.of_ints(composed)})
    im.ranks[node.name] = 4
}

pub fn translate_reshape(im: Importer, node: TfNode) -> void {
    if node.inputs.len() < 2 { throw ValueError(f"{node.label()} needs a value and a shape") }
    let rank = im.rank_of(node.inputs[0], node)
    let dims = im.const_ints(node.inputs[1], node, "shape")
    if dims.len() == 0 {
        throw ValueError(
            f"{node.label()} reshapes to a scalar, and the executor has no rank-zero tensor"
        )
    }
    let x = im.value_input(node.inputs[0], node)
    #: A reshape reads the buffer in TensorFlow's own order, so this is the
    #: one operator that cannot be expressed by renumbering axes: the values
    #: really do have to be moved back to `NHWC` first.
    let source = if rank == 4 { im.to_nhwc(x, node.name + "$nhwc") } else { x }
    let shape = im.intern(
        node.inputs[1] + "$dims",
        [dims.len()],
        [float(dimension) for dimension in dims]
    )
    let grown = dims.len() == 4
    let inner = if grown { node.name + "$shaped" } else { node.name }
    #: TensorFlow has no "copy the input's dimension" convention for a zero,
    #: so the executor is told not to apply ONNX's.
    let _shaped = im.emit("Reshape", inner, [source, shape], {"allowzero": Attr.of_int(1)})
    if grown { let _back = im.to_nchw(inner, node.name) }
    im.ranks[node.name] = dims.len()
}

pub fn translate_concat(im: Importer, node: TfNode) -> void {
    if node.inputs.len() == 0 { throw ValueError(f"{node.label()} joins nothing") }
    var values: list[str] = []
    var axis_name = ""
    if node.op == "Concat" {
        axis_name = node.inputs[0]
        for slot in 1..node.inputs.len() { values.push(node.inputs[slot]) }
    } else {
        axis_name = node.inputs[node.inputs.len() - 1]
        for slot in 0..node.inputs.len() - 1 { values.push(node.inputs[slot]) }
    }
    if values.len() == 0 { throw ValueError(f"{node.label()} joins nothing") }
    let asked = im.const_ints(axis_name, node, "axis")
    if asked.len() != 1 { throw ValueError(f"{node.label()} needs one axis, got {asked.len()}") }
    let rank = im.rank_of(values[0], node)
    let axes = normalized_axes(asked, rank, node, "the axis")
    var ins: list[str] = []
    for name in values {
        if im.rank_of(name, node) != rank {
            throw ValueError(f"{node.label()} joins tensors of different ranks")
        }
        ins.push(im.value_input(name, node))
    }
    let axis = if rank == 4 { NCHW_AXIS[axes[0]] } else { axes[0] }
    let _joined = im.emit("Concat", node.name, ins, {"axis": Attr.of_int(axis)})
    im.ranks[node.name] = rank
}

pub fn translate_pad(im: Importer, node: TfNode) -> void {
    if node.inputs.len() < 2 { throw ValueError(f"{node.label()} needs a value and paddings") }
    let rank = im.rank_of(node.inputs[0], node)
    let widths = im.const_ints(node.inputs[1], node, "paddings")
    if widths.len() != 2 * rank {
        throw ValueError(f"{node.label()} has {widths.len()} pad widths for a rank-{rank} tensor")
    }
    var before: list[int] = []
    var after: list[int] = []
    for slot in 0..rank {
        #: The paddings are one row an axis in TensorFlow's order, and the
        #: rows have to follow the axes into `NCHW`.
        let axis = if rank == 4 { TF_AXIS[slot] } else { slot }
        before.push(widths[2 * axis])
        after.push(widths[2 * axis + 1])
    }
    var pads = [width for width in before]
    for width in after { pads.push(width) }
    var mode = "constant"
    if node.op == "MirrorPad" {
        let asked = attr_string(node, "mode", "REFLECT")
        if asked != "REFLECT" {
            throw ValueError(f"{node.label()} mirrors {asked}, and only REFLECT is implemented")
        }
        mode = "reflect"
    }
    var value = 0.0
    if node.op == "PadV2" and node.inputs.len() > 2 {
        let held = im.const_of(node.inputs[2])
        if held is null {
            throw ValueError(f"{node.label()} computes its pad value rather than holding it")
        }
        let filler: TfTensor = held
        if filler.count() != 1 {
            throw ValueError(f"{node.label()} pads with {filler.count()} values, expected one")
        }
        value = filler.values()[0]
    }
    let x = im.value_input(node.inputs[0], node)
    let _padded = im.emit(
        "Pad",
        node.name,
        [x],
        {
            "pads": Attr.of_ints(pads),
            "mode": Attr.of_str(mode),
            "value": Attr.of_float(value),
        }
    )
    im.ranks[node.name] = rank
}

pub fn translate_cast(im: Importer, node: TfNode) -> void {
    let target = attr_type(node, "DstT", DT_FLOAT)
    let onnx = ONNX_DTYPES.get(target, 0)
    if onnx == 0 {
        throw ValueError(
            f"{node.label()} casts to {DTYPE_NAMES.get(target, f'data type {target}')}, which " +
                "the executor has no numbering for"
        )
    }
    translate_unary(im, node, "Cast", {"to": Attr.of_int(onnx)})
}
