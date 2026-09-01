#: The layers that join, cut and reshape blobs, and the elementwise arithmetic.
from ..graph import Attr
from .reading import Reading, Step, floats_of_ints, resolve_axis

pub fn layer_concat(reading: Reading, step: Step) -> void {
    let param = step.layer.param("concat_param", "ConcatParameter")
    let asked = if param.has("axis") {
        param.int_of("axis", 1)
    } else {
        param.int_of("concat_dim", 1)
    }
    var ins: list[str] = []
    for blob in step.bottoms { ins.push(reading.read(step, blob)) }
    let given = reading.shape_of(ins[0])
    var axis = asked
    var shape: list[int]? = null
    if given is not null {
        let dims: list[int] = given
        axis = resolve_axis(asked, dims.len(), step)
        var joined = [dimension for dimension in dims]
        var width = 0
        var complete = true
        for name in ins {
            let part = reading.shape_of(name)
            if part is null {
                complete = false
                break
            }
            let other: list[int] = part
            width += other[axis]
        }
        if complete {
            joined[axis] = width
            shape = joined
        }
    }
    reading.emit(
        "Concat",
        step.name,
        ins,
        [reading.write(step.tops[0], shape)],
        {"axis": Attr.of_int(axis)}
    )
}

pub fn layer_slice(reading: Reading, step: Step) -> void {
    let param = step.layer.param("slice_param", "SliceParameter")
    let asked = if param.has("axis") {
        param.int_of("axis", 1)
    } else {
        param.int_of("slice_dim", 1)
    }
    let source = reading.read(step, step.bottoms[0])
    let points = param.ints_of("slice_point")
    let given = reading.shape_of(source)
    var axis = asked
    var widths: list[int] = []
    if given is not null {
        let dims: list[int] = given
        axis = resolve_axis(asked, dims.len(), step)
        var start = 0
        for point in points {
            if point <= start or point > dims[axis] {
                throw ValueError(f"{step.label()} cuts axis {axis} of {dims} at {points}")
            }
            widths.push(point - start)
            start = point
        }
        widths.push(dims[axis] - start)
        if widths.len() != step.tops.len() {
            throw ValueError(
                f"{step.label()} cuts into {widths.len()} pieces and names {step.tops.len()} tops"
            )
        }
    } elif points.len() > 0 {
        throw ValueError(
            f"{step.label()} cuts at {points}, and the last piece runs to the end of an axis " +
                "whose extent this file does not fix"
        )
    }

    var attrs: dict[str, Attr] = {"axis": Attr.of_int(axis)}
    if widths.len() > 0 { attrs["split"] = Attr.of_ints(widths) }
    var outs: list[str] = []
    for (index, top) in step.tops.enumerate() {
        var shape: list[int]? = null
        if given is not null and widths.len() > 0 {
            let dims: list[int] = given
            var piece = [dimension for dimension in dims]
            piece[axis] = widths[index]
            shape = piece
        }
        outs.push(reading.write(top, shape))
    }
    reading.emit("Split", step.name, [source], outs, attrs)
}

pub fn layer_softmax(reading: Reading, step: Step) -> void {
    let param = step.layer.param("softmax_param", "SoftmaxParameter")
    let source = reading.read(step, step.bottoms[0])
    reading.emit(
        "Softmax",
        step.name,
        [source],
        [reading.write(step.tops[0], reading.shape_of(source))],
        {"axis": Attr.of_int(param.int_of("axis", 1))}
    )
}

pub fn layer_eltwise(reading: Reading, step: Step) -> void {
    let param = step.layer.param("eltwise_param", "EltwiseParameter")
    let operation = param.int_of("operation", 1)
    let coefficients = param.floats_of("coeff")
    if coefficients.len() > 0 and operation != 1 {
        throw ValueError(f"{step.label()} gives coefficients to an operation that is not SUM")
    }
    if step.bottoms.len() < 2 {
        throw ValueError(f"{step.label()} combines {step.bottoms.len()} inputs, and needs two")
    }
    if coefficients.len() > 0 and coefficients.len() != step.bottoms.len() {
        throw ValueError(
            f"{step.label()} gives {coefficients.len()} coefficients to " +
                f"{step.bottoms.len()} inputs"
        )
    }
    var ins: list[str] = []
    for (index, blob) in step.bottoms.enumerate() {
        var name = reading.read(step, blob)
        if coefficients.len() > 0 and coefficients[index] != 1.0 {
            let factor = reading.constant(f"{step.name}/coeff{index}", [1], [coefficients[index]])
            let weighted = reading.fresh(f"{step.name}/weighted{index}")
            reading.emit("Mul", f"{step.name}/scale{index}", [name, factor], [weighted])
            name = weighted
        }
        ins.push(name)
    }
    let shape = reading.shape_of(ins[0])
    if operation == 1 {
        reading.emit("Sum", step.name, ins, [reading.write(step.tops[0], shape)])
        return
    }
    if operation == 2 {
        reading.emit("Max", step.name, ins, [reading.write(step.tops[0], shape)])
        return
    }

    #: The executor's `Mul` is binary, so a product of more than two inputs is
    #: a chain of them.
    var running = ins[0]
    for index in 1..ins.len() {
        let last = index == ins.len() - 1
        let out = if last {
            reading.write(step.tops[0], shape)
        } else {
            reading.fresh(f"{step.name}/product{index}")
        }
        let name = if last { step.name } else { f"{step.name}/mul{index}" }
        reading.emit("Mul", name, [running, ins[index]], [out])
        running = out
    }
}

pub fn layer_flatten(reading: Reading, step: Step) -> void {
    let param = step.layer.param("flatten_param", "FlattenParameter")
    let source = reading.read(step, step.bottoms[0])
    let given = reading.shape_of(source)
    let asked = param.int_of("axis", 1)
    let asked_end = param.int_of("end_axis", -1)

    #: `Flatten` folds one run of axes and keeps the rest, which is ONNX's
    #: operator only when the run reaches the last axis; anything else is a
    #: reshape, and needs the shape to write one.
    if given is null {
        if asked != 1 or asked_end != -1 {
            throw ValueError(
                f"{step.label()} folds axes {asked}..{asked_end}, which needs the input's rank"
            )
        }
        reading.emit(
            "Flatten",
            step.name,
            [source],
            [reading.write(step.tops[0], null)],
            {"axis": Attr.of_int(asked)}
        )
        return
    }
    let dims: list[int] = given
    let rank = dims.len()
    let first = resolve_axis(asked, rank, step)
    let last = resolve_axis(asked_end, rank, step)
    if last < first {
        throw ValueError(f"{step.label()} folds axes {first}..{last}, which run backwards")
    }
    var folded: list[int] = []
    for axis in 0..first { folded.push(dims[axis]) }
    var inner = 1
    for axis in first..last + 1 { inner *= dims[axis] }
    folded.push(inner)
    for axis in last + 1..rank { folded.push(dims[axis]) }
    #: ONNX's `Flatten` always produces a matrix, so it is Caffe's layer only
    #: when the run starts at axis one and reaches the end. Every other run
    #: keeps axes on both sides and has to be spelled as a reshape.
    if first == 1 and last == rank - 1 {
        reading.emit(
            "Flatten",
            step.name,
            [source],
            [reading.write(step.tops[0], folded)],
            {"axis": Attr.of_int(1)}
        )
        return
    }
    let shape = reading.constant(f"{step.name}/shape", [folded.len()], floats_of_ints(folded))
    reading.emit("Reshape", step.name, [source, shape], [reading.write(step.tops[0], folded)])
}

pub fn layer_reshape(reading: Reading, step: Step) -> void {
    let param = step.layer.param("reshape_param", "ReshapeParameter")
    let asked = param.param("shape", "BlobShape").ints_of("dim")
    let source = reading.read(step, step.bottoms[0])
    let given = reading.shape_of(source)
    let start = param.int_of("axis", 0)
    let span = param.int_of("num_axes", -1)

    #: Caffe replaces the axes `[axis, axis + num_axes)` and keeps the rest,
    #: and inside that window a zero copies the axis it sits over and a minus
    #: one is inferred -- which is ONNX's `Reshape` exactly when the window is
    #: the whole shape. It usually is.
    if given is null {
        if start != 0 or span != -1 {
            throw ValueError(
                f"{step.label()} reshapes only part of its input, which needs the input's shape"
            )
        }
        let dims = reading.constant(f"{step.name}/shape", [asked.len()], floats_of_ints(asked))
        reading.emit("Reshape", step.name, [source, dims], [reading.write(step.tops[0], null)])
        return
    }
    let shape: list[int] = given
    let rank = shape.len()
    let first = if start >= 0 { start } else { rank + start + 1 }
    if first < 0 or first > rank {
        throw ValueError(f"{step.label()} starts at axis {start} of a rank-{rank} input")
    }
    let width = if span >= 0 { span } else { rank - first }
    if first + width > rank {
        throw ValueError(f"{step.label()} replaces {width} axes from axis {first} of {shape}")
    }
    var dims: list[int] = []
    for axis in 0..first { dims.push(shape[axis]) }
    for (slot, value) in asked.enumerate() {
        dims.push(if value == 0 { shape[first + slot] } else { value })
    }
    for axis in first + width..rank { dims.push(shape[axis]) }
    var total = 1
    for dimension in shape { total *= dimension }
    var known = 1
    var infer = -1
    for (axis, dimension) in dims.enumerate() {
        if dimension == -1 {
            if infer >= 0 { throw ValueError(f"{step.label()} leaves two dimensions to infer") }
            infer = axis
        } else {
            known *= dimension
        }
    }
    if infer >= 0 {
        if known == 0 or total % known != 0 {
            throw ValueError(f"{step.label()} cannot fit {total} values into {dims}")
        }
        dims[infer] = total // known
    } elif known != total {
        throw ValueError(f"{step.label()} reshapes {total} values into {dims}")
    }
    let name = reading.constant(f"{step.name}/shape", [dims.len()], floats_of_ints(dims))
    reading.emit("Reshape", step.name, [source, name], [reading.write(step.tops[0], dims)])
}

pub fn layer_power(reading: Reading, step: Step) -> void {
    let param = step.layer.param("power_param", "PowerParameter")
    let exponent = param.float_of("power", 1.0)
    let factor = param.float_of("scale", 1.0)
    let offset = param.float_of("shift", 0.0)
    let source = reading.read(step, step.bottoms[0])
    let shape = reading.shape_of(source)

    #: `(shift + scale * x) ^ power`, with each of the three dropped when it
    #: is the identity -- which for a plain `Power` layer is all three.
    var stages: list[str] = []
    if factor != 1.0 { stages.push("Mul") }
    if offset != 0.0 { stages.push("Add") }
    if exponent != 1.0 { stages.push("Pow") }
    if stages.len() == 0 {
        reading.emit("Identity", step.name, [source], [reading.write(step.tops[0], shape)])
        return
    }
    var running = source
    for (index, op) in stages.enumerate() {
        let (value, stem) = match op {
            "Mul" => (factor, "factor"),
            "Add" => (offset, "offset"),
            _ => (exponent, "exponent"),
        }
        let operand = reading.constant(f"{step.name}/{stem}", [1], [value])
        let last = index == stages.len() - 1
        let out = if last {
            reading.write(step.tops[0], shape)
        } else {
            reading.fresh(f"{step.name}/{stem}ed")
        }
        let name = if last { step.name } else { f"{step.name}/{op.lower()}" }
        reading.emit(op, name, [running, operand], [out])
        running = out
    }
}

#: Caffe's `Split` hands one blob to several names, which the graph spells as
#: one copy apiece: two nodes may not produce the same name.
pub fn layer_split(reading: Reading, step: Step) -> void {
    let source = reading.read(step, step.bottoms[0])
    let shape = reading.shape_of(source)
    for (index, top) in step.tops.enumerate() {
        reading.emit("Identity", if index == 0 {
            step.name
        } else {
            f"{step.name}/{index}"
        }, [source], [reading.write(top, shape)])
    }
}
