#: The activations and the normalisers, including the BatchNorm/Scale fold.
from ..graph import Attr
from .blobs import Blob, blobs_of
from .message import Message
from .reading import Reading, Step, floats_of_ints, need_shape, resolve_axis

#: The activations that are one operator with at most one number on them.
pub fn layer_activation(reading: Reading, step: Step) -> void {
    var op = ""
    var attrs: dict[str, Attr] = {}
    if step.kind == "ReLU" {
        let slope = step.layer.param("relu_param", "ReLUParameter").float_of("negative_slope", 0.0)
        if slope == 0.0 {
            op = "Relu"
        } else {
            op = "LeakyRelu"
            attrs["alpha"] = Attr.of_float(slope)
        }
    } elif step.kind == "ELU" {
        op = "Elu"
        attrs["alpha"] = Attr.of_float(
            step.layer.param("elu_param", "ELUParameter").float_of("alpha", 1.0)
        )
    } elif step.kind == "Sigmoid" {
        op = "Sigmoid"
    } elif step.kind == "TanH" {
        op = "Tanh"
    } elif step.kind == "AbsVal" {
        op = "Abs"
    } elif step.kind == "BNLL" {
        op = "Softplus"
    } else {
        op = "Dropout"
    }
    let source = reading.read(step, step.bottoms[0])
    reading.emit(
        op,
        step.name,
        [source],
        [reading.write(step.tops[0], reading.shape_of(source))],
        attrs
    )
}

pub fn layer_prelu(reading: Reading, step: Step) -> void {
    let param = step.layer.param("prelu_param", "PReLUParameter")
    let shared = param.bool_of("channel_shared", false)
    let source = reading.read(step, step.bottoms[0])
    let blobs = blobs_of(step.layer, step.name)

    #: ONNX broadcasts the slope against the input the way numpy does, from
    #: the last axis back, so a per-channel slope has to carry the trailing
    #: axes as ones or it lands on the wrong one.
    var slope_shape: list[int]? = if shared { [1] } else { null }
    if not shared {
        let dims = need_shape(reading, step)
        var padded = [dims[1]]
        for _axis in 2..dims.len() { padded.push(1) }
        slope_shape = padded
    }
    reading.emit(
        "PRelu",
        step.name,
        [source, reading.weight(step, blobs, 0, "slope", slope_shape)],
        [reading.write(step.tops[0], reading.shape_of(source))]
    )
}

pub fn layer_lrn(reading: Reading, step: Step) -> void {
    let param = step.layer.param("lrn_param", "LRNParameter")
    if param.int_of("norm_region", 0) != 0 {
        throw ValueError(
            f"{step.label()} normalises within a channel, and the executor's LRN is the " +
                "across-channel one"
        )
    }
    let source = reading.read(step, step.bottoms[0])
    reading.emit(
        "LRN",
        step.name,
        [source],
        [reading.write(step.tops[0], reading.shape_of(source))],
        {
            "size": Attr.of_int(param.int_of("local_size", 5)),
            "alpha": Attr.of_float(param.float_of("alpha", 1.0)),
            "beta": Attr.of_float(param.float_of("beta", 0.75)),
            "bias": Attr.of_float(param.float_of("k", 1.0)),
        }
    )
}

#: The scale factor Caffe's third blob carries, as the number the other two
#: must be multiplied by.
#:
#: Zero does not mean "divide by zero", it means no statistics were ever
#: accumulated, and Caffe reads the mean and the variance back as zero.
fn batch_norm_factor(blobs: list[Blob], step: Step) -> float {
    if blobs.len() < 3 { return 1.0 }
    if blobs[2].values.len() != 1 {
        throw ValueError(
            f"{step.label()} has {blobs[2].values.len()} numbers in its scale-factor blob, " +
                "which holds one"
        )
    }
    let factor = blobs[2].values[0]
    if factor == 0.0 { return 0.0 }
    return 1.0 / factor
}

pub fn layer_batch_norm(reading: Reading, step: Step) -> void {
    let param = step.layer.param("batch_norm_param", "BatchNormParameter")
    if param.has("use_global_stats") and not param.bool_of("use_global_stats", true) {
        throw ValueError(
            f"{step.label()} asks for the statistics of the batch in front of it, which is " +
                "training and not inference"
        )
    }
    let blobs = blobs_of(step.layer, step.name)
    if blobs.len() > 0 and blobs.len() != 3 {
        throw ValueError(
            f"{step.label()} carries {blobs.len()} blobs, and a BatchNorm has three: the " +
                "mean, the variance, and the factor they were both scaled by"
        )
    }
    let factor = batch_norm_factor(blobs, step)
    var scaled: list[Blob] = []
    for index in 0..(if blobs.len() == 3 { 2 } else { 0 }) {
        scaled.push(Blob(blobs[index].shape, [value * factor for value in blobs[index].values]))
    }
    let source = reading.read(step, step.bottoms[0])

    var channels = 0
    if scaled.len() > 0 {
        channels = scaled[0].values.len()
    } else {
        let given = reading.shape_of(source)
        if given is not null {
            let dims: list[int] = given
            if dims.len() >= 2 { channels = dims[1] }
        }
    }
    let mean = reading.weight(step, scaled, 0, "mean", if channels > 0 { [channels] } else { null })
    let variance = reading.weight(step, scaled, 1, "variance", if channels > 0 {
        [channels]
    } else {
        null
    })

    #: ONNX fuses the affine part into the same operator, so a `Scale` layer
    #: directly after this one is folded in and its blobs become the scale and
    #: the shift. Without one they are the identity, which still has to be
    #: spelled out because the operator has no default for them.
    var gamma = ""
    var beta = ""
    let folded = step.scale
    if folded is null {
        if channels == 0 {
            throw ValueError(
                f"{step.label()} carries no statistics and this file does not fix its " +
                    "channel count, so the unit scale ONNX wants cannot be sized"
            )
        }
        gamma = reading.constant(f"{step.name}/scale", [channels], [1.0 for _slot in 0..channels])
        beta = reading.constant(f"{step.name}/bias", [channels], [0.0 for _slot in 0..channels])
    } else {
        let scale: Message = folded
        let affine = Step(scale, "Scale", scale.str_of("name", step.name), [], [])
        let learned = blobs_of(scale, affine.name)
        let biased = scale.param("scale_param", "ScaleParameter").bool_of("bias_term", false)
        if learned.len() > 0 and learned.len() != (if biased { 2 } else { 1 }) {
            throw ValueError(f"{affine.label()} carries {learned.len()} blobs")
        }
        let width = if channels > 0 { [channels] } else { null }
        gamma = reading.weight(affine, learned, 0, "scale", width)
        beta = if biased {
            reading.weight(affine, learned, 1, "bias", width)
        } else {
            if channels == 0 {
                throw ValueError(f"{affine.label()} has no bias and this file does not fix its channel " + "count, so the zero ONNX wants cannot be sized")
            }
            reading.constant(f"{affine.name}/bias", [channels], [0.0 for _slot in 0..channels])
        }
    }
    reading.emit(
        "BatchNormalization",
        step.name,
        [source, gamma, beta, mean, variance],
        [reading.write(step.tops[0], reading.shape_of(source))],
        {"epsilon": Attr.of_float(param.float_of("eps", 1e-5))}
    )
}

#: `Scale` and `Bias`, which are the same layer with and without the multiply.
pub fn layer_affine(reading: Reading, step: Step) -> void {
    let scaling = step.kind == "Scale"
    let param = if scaling {
        step.layer.param("scale_param", "ScaleParameter")
    } else {
        step.layer.param("bias_param", "BiasParameter")
    }
    let source = reading.read(step, step.bottoms[0])
    let dims = need_shape(reading, step)
    let rank = dims.len()
    let axis = resolve_axis(param.int_of("axis", 1), rank, step)
    let biased = scaling and param.bool_of("bias_term", false)

    #: Caffe applies the parameter to the axes `[axis, axis + num_axes)` and
    #: broadcasts over the rest; ONNX broadcasts from the last axis back. The
    #: two agree once the trailing axes are spelled out as ones.
    var width = param.int_of("num_axes", 1)
    var operand = ""
    var operand_shape: list[int] = []
    if step.bottoms.len() > 1 {
        let other = reading.read(step, step.bottoms[1])
        let second = reading.shape_of(other)
        if second is null {
            throw ValueError(
                f"{step.label()} scales by {step.bottoms[1]}, whose shape this file does " +
                    "not fix"
            )
        }
        let given: list[int] = second
        width = given.len()
        operand_shape = given
        operand = other
    } else {
        if width < 0 { width = rank - axis }
        for index in 0..width { operand_shape.push(dims[axis + index]) }
    }
    if axis + width > rank {
        throw ValueError(
            f"{step.label()} covers {width} axes from axis {axis} of a rank-{rank} input"
        )
    }
    var stretched = [dimension for dimension in operand_shape]
    for _axis in axis + width..rank { stretched.push(1) }

    let blobs = blobs_of(step.layer, step.name)

    #: With a second bottom the multiplier is a blob of the net and not of the
    #: layer, so a learned bias is the layer's first blob rather than its
    #: second -- which is how Caffe numbers them too.
    var bias_index = 0
    if operand.len() == 0 {
        bias_index = 1
        let expected = if biased { 2 } else { 1 }
        if blobs.len() > 0 and blobs.len() != expected {
            throw ValueError(f"{step.label()} carries {blobs.len()} blobs, and needs {expected}")
        }
        operand = reading.weight(step, blobs, 0, if scaling { "scale" } else { "bias" }, stretched)
    } elif stretched != operand_shape {
        let dims_name = reading.constant(
            f"{step.name}/shape",
            [stretched.len()],
            floats_of_ints(stretched)
        )
        let reshaped = reading.fresh(f"{step.name}/spread")
        reading.emit("Reshape", f"{step.name}/reshape", [operand, dims_name], [reshaped])
        operand = reshaped
    }

    let last = if biased {
        reading.fresh(f"{step.name}/scaled")
    } else {
        reading.write(step.tops[0], dims)
    }
    reading.emit(if scaling { "Mul" } else { "Add" }, step.name, [source, operand], [last])
    if not biased { return }
    let shift = reading.weight(step, blobs, bias_index, "bias", stretched)
    reading.emit("Add", f"{step.name}/shift", [last, shift], [reading.write(step.tops[0], dims)])
}
