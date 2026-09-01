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

from .graph.kernels import LOWEST_FLOAT, f32_bits, kernel_for
from .graph.nodes import Attr, Graph, GraphNode, attr_float, attr_floats, attr_int, attr_ints
from .graph.nodes import attr_str, maybe, need
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
from .graph.tensors import concat_along, copy_of, host_ints, normalize_axis, pad_nd_of
from .graph.tensors import row_reduce_of, slice_axis

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
