#: The layers that slide a window, or a weight matrix, over their input.
from ..graph import Attr
from .blobs import blobs_of
from .extent import axis_pair, conv_pair, convolved, pool_pair, pooled_pair
from .reading import Reading, Step, floats_of_ints, need_shape, resolve_axis

pub fn layer_conv(reading: Reading, step: Step) -> void {
    let param = step.layer.param("convolution_param", "ConvolutionParameter")
    if param.int_of("axis", 1) != 1 {
        throw ValueError(
            f"{step.label()} convolves over axis {param.int_of('axis', 1)}, and this importer " +
                "only does the usual channel axis"
        )
    }
    let outputs = param.int_of("num_output", 0)
    if outputs <= 0 { throw ValueError(f"{step.label()} declares num_output {outputs}") }
    let groups = param.int_of("group", 1)
    if groups <= 0 or outputs % groups != 0 {
        throw ValueError(f"{step.label()} splits {outputs} filters into {groups} groups")
    }
    let window = conv_pair(param, "kernel_size", "kernel_h", "kernel_w", 0)
    if window[0] <= 0 or window[1] <= 0 {
        throw ValueError(f"{step.label()} declares a {window[0]}x{window[1]} kernel")
    }
    let stride = conv_pair(param, "stride", "stride_h", "stride_w", 1)
    let pad = conv_pair(param, "pad", "pad_h", "pad_w", 0)
    let dilation = axis_pair(param.ints_of("dilation"), 1)
    let biased = param.bool_of("bias_term", true)
    let up = step.kind == "Deconvolution"

    let blobs = blobs_of(step.layer, step.name)
    let wanted = if biased { 2 } else { 1 }
    if blobs.len() > 0 and blobs.len() != wanted {
        throw ValueError(
            f"{step.label()} carries {blobs.len()} blobs, and a convolution with " +
                f"bias_term {biased} has {wanted}"
        )
    }

    #: The one dimension the file does not spell out is the input channel
    #: count, and the blob's own length fixes it whichever way round the
    #: weights are laid out.
    let plane = window[0] * window[1]
    var weight_shape: list[int]? = null
    if blobs.len() > 0 {
        let total = blobs[0].values.len()
        let known = if up { outputs // groups } else { outputs }
        if known * plane == 0 or total % (known * plane) != 0 {
            throw ValueError(
                f"{step.label()} has {total} weights, which is not a whole number of " +
                    f"{window[0]}x{window[1]} filters"
            )
        }
        let other = total // (known * plane)
        weight_shape = if up {
            [other, outputs // groups, window[0], window[1]]
        } else {
            [outputs, other, window[0], window[1]]
        }
    }
    let source = reading.read(step, step.bottoms[0])
    var ins = [source, reading.weight(step, blobs, 0, "weight", weight_shape)]
    if biased { ins.push(reading.weight(step, blobs, 1, "bias", [outputs])) }

    var attrs: dict[str, Attr] = {
        "kernel_shape": Attr.of_ints(window),
        "strides": Attr.of_ints(stride),
        "pads": Attr.of_ints([pad[0], pad[1], pad[0], pad[1]]),
        "dilations": Attr.of_ints(dilation),
        "group": Attr.of_int(groups),
    }
    var shape: list[int]? = null
    let given = reading.shape_of(source)
    if given is not null {
        let dims: list[int] = given
        if dims.len() != 4 {
            throw ValueError(f"{step.label()} reads {dims}, and convolution needs four axes")
        }
        var spatial: list[int] = []
        for axis in 0..2 {
            spatial.push(if up {
                (dims[2 + axis] - 1) * stride[axis] - 2 * pad[axis] + dilation[axis] * (window[axis] - 1) + 1
            } else {
                convolved(dims[2 + axis], window[axis], stride[axis], pad[axis], dilation[axis])
            })
        }
        shape = [dims[0], outputs, spatial[0], spatial[1]]
    }
    reading.emit(if up {
        "ConvTranspose"
    } else {
        "Conv"
    }, step.name, ins, [reading.write(step.tops[0], shape)], attrs)
}

pub fn layer_pool(reading: Reading, step: Step) -> void {
    let param = step.layer.param("pooling_param", "PoolingParameter")
    let method = param.int_of("pool", 0)
    if method == 2 {
        throw ValueError(f"{step.label()} pools stochastically, which is training-only")
    }
    let biggest = method == 0
    let source = reading.read(step, step.bottoms[0])
    let given = reading.shape_of(source)

    if param.bool_of("global_pooling", false) {
        var shape: list[int]? = null
        if given is not null {
            let dims: list[int] = given
            if dims.len() != 4 {
                throw ValueError(f"{step.label()} reads {dims}, and pooling needs four axes")
            }
            shape = [dims[0], dims[1], 1, 1]
        }
        let op = if biggest { "GlobalMaxPool" } else { "GlobalAveragePool" }
        reading.emit(op, step.name, [source], [reading.write(step.tops[0], shape)])
        return
    }

    let window = pool_pair(param, "kernel_size", "kernel_h", "kernel_w", 0, true)
    if window[0] <= 0 or window[1] <= 0 {
        throw ValueError(f"{step.label()} declares a {window[0]}x{window[1]} window")
    }
    let stride = pool_pair(param, "stride", "stride_h", "stride_w", 1, false)
    let pad = pool_pair(param, "pad", "pad_h", "pad_w", 0, false)
    let ceiling = param.int_of("round_mode", 0) == 0

    #: With the input extent in hand the exact output extent is Caffe's, and
    #: the padding that reproduces it under the plain floor rule can be handed
    #: to the executor directly. Without it, `ceil_mode` is the closest thing
    #: available and differs only where Caffe would have dropped the last
    #: window.
    var trailing: list[int]? = null
    var shape: list[int]? = null
    if given is not null {
        let dims: list[int] = given
        if dims.len() != 4 {
            throw ValueError(f"{step.label()} reads {dims}, and pooling needs four axes")
        }
        let counts = pooled_pair(step, [dims[2], dims[3]], window, stride, pad, ceiling)
        var ends: list[int] = []
        for axis in 0..2 {
            ends.push((counts[axis] - 1) * stride[axis] + window[axis] - dims[2 + axis] - pad[axis])
        }
        trailing = ends
        shape = [dims[0], dims[1], counts[0], counts[1]]
    }

    var attrs: dict[str, Attr] = {
        "kernel_shape": Attr.of_ints(window),
        "strides": Attr.of_ints(stride),
    }
    if biggest {
        let ends = trailing ?? pad
        attrs["pads"] = Attr.of_ints([pad[0], pad[1], ends[0], ends[1]])
        if trailing is null and ceiling { attrs["ceil_mode"] = Attr.of_int(1) }
        reading.emit("MaxPool", step.name, [source], [reading.write(step.tops[0], shape)], attrs)
        return
    }

    #: Caffe's average divides by the window clipped to the *padded* image, so
    #: the padding is materialised first and the pool is then told to count
    #: only what is inside it. That is exactly `count_include_pad: 0` over the
    #: padded tensor, and it also gets the shrunken divisor of a window that
    #: hangs off the end right, which neither ONNX setting does on its own.
    var input = source
    if pad[0] > 0 or pad[1] > 0 {
        let widths = reading.constant(
            f"{step.name}/pads",
            [8],
            floats_of_ints([0, 0, pad[0], pad[1], 0, 0, pad[0], pad[1]])
        )
        let padded = reading.fresh(f"{step.name}/padded")
        reading.emit(
            "Pad",
            f"{step.name}/pad",
            [input, widths],
            [padded],
            {"mode": Attr.of_str("constant")}
        )
        input = padded
    }
    attrs["count_include_pad"] = Attr.of_int(0)
    if trailing is null {
        if ceiling { attrs["ceil_mode"] = Attr.of_int(1) }
    } else {
        let ends: list[int] = trailing
        attrs["pads"] = Attr.of_ints([0, 0, ends[0] - pad[0], ends[1] - pad[1]])
    }
    reading.emit("AveragePool", step.name, [input], [reading.write(step.tops[0], shape)], attrs)
}

pub fn layer_inner_product(reading: Reading, step: Step) -> void {
    let param = step.layer.param("inner_product_param", "InnerProductParameter")
    let outputs = param.int_of("num_output", 0)
    if outputs <= 0 { throw ValueError(f"{step.label()} declares num_output {outputs}") }
    let biased = param.bool_of("bias_term", true)
    let transposed = param.bool_of("transpose", false)
    let source = reading.read(step, step.bottoms[0])
    let given = reading.shape_of(source)
    var axis = param.int_of("axis", 1)
    if axis != 1 or given is not null {
        let dims = need_shape(reading, step)
        axis = resolve_axis(axis, dims.len(), step)
    }

    let blobs = blobs_of(step.layer, step.name)
    if blobs.len() > 0 and blobs.len() != (if biased { 2 } else { 1 }) {
        throw ValueError(f"{step.label()} carries {blobs.len()} blobs")
    }
    var weight_shape: list[int]? = null
    if blobs.len() > 0 {
        let total = blobs[0].values.len()
        if total % outputs != 0 {
            throw ValueError(f"{step.label()} has {total} weights for {outputs} outputs")
        }
        let inner = total // outputs
        weight_shape = if transposed { [inner, outputs] } else { [outputs, inner] }
    }

    #: Gemm takes matrices, so everything from `axis` on is folded into one
    #: axis first -- unless the input already is a matrix cut in the same
    #: place, in which case the fold would only be a copy.
    var flat = source
    var leading: list[int] = []
    var already = false
    if given is not null {
        let dims: list[int] = given
        leading = [dims[index] for index in 0..axis]
        already = dims.len() == 2 and axis == 1
    }
    if not already {
        flat = reading.fresh(f"{step.name}/flat")
        reading.emit(
            "Flatten",
            f"{step.name}/flatten",
            [source],
            [flat],
            {"axis": Attr.of_int(axis)}
        )
    }
    var ins = [flat, reading.weight(step, blobs, 0, "weight", weight_shape)]
    if biased { ins.push(reading.weight(step, blobs, 1, "bias", [outputs])) }

    #: Caffe keeps every axis before `axis` and puts `num_output` after them,
    #: which is the matrix Gemm produces only when exactly one axis leads.
    var tail: list[int] = []
    for dimension in leading { tail.push(dimension) }
    tail.push(outputs)
    let matrix = leading.len() <= 1
    var shape: list[int]? = null
    if given is not null { shape = tail }
    let product = if matrix {
        reading.write(step.tops[0], shape)
    } else {
        reading.fresh(f"{step.name}/matrix")
    }
    reading.emit("Gemm", step.name, ins, [product], {"transB": Attr.of_int(if transposed {
        0
    } else {
        1
    })})
    if matrix { return }
    let dims = reading.constant(f"{step.name}/shape", [tail.len()], floats_of_ints(tail))
    reading.emit(
        "Reshape",
        f"{step.name}/reshape",
        [product, dims],
        [reading.write(step.tops[0], shape)]
    )
}
