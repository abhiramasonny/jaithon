#: The layer operators: convolution and deconvolution, the pools, the four
#: normalisations, and spatial resampling -- everything that reads an NCHW
#: image through a window or along its channels.
import jaitensor as jt
from jaitensor import Tensor

from .kernels import LOWEST_FLOAT, f32_bits, kernel_for
from .nodes import GraphNode, attr_float, attr_floats, attr_int, attr_ints, attr_str, maybe, need
from .tensors import concat_along, host_ints, pad_nd_of, row_reduce_of, slice_axis

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

pub fn op_conv(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
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

pub fn op_conv_transpose(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
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

pub fn op_max_pool(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return op_pool(node, ins, true)
}

pub fn op_average_pool(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
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

pub fn op_global_average_pool(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return op_global_pool(node, ins, 1)
}

pub fn op_global_max_pool(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
    return op_global_pool(node, ins, 2)
}

pub fn op_batch_norm(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
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
pub fn op_layer_norm(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
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

pub fn op_instance_norm(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
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

pub fn op_lrn(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
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

pub fn op_resize(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
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

pub fn op_upsample(node: GraphNode, ins: list[Tensor?]) -> list[Tensor] {
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
