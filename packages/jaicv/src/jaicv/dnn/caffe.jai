#: Reading a Caffe model.
#:
#: Caffe stores a network twice over. The `.prototxt` is one `NetParameter` in
#: protobuf's *text* format and holds the architecture; the `.caffemodel` is
#: the same message in the *binary* format with the weights filled in. Callers
#: have both, so both are read here and merged by layer name -- the prototxt
#: gives the structure, and a caffemodel layer's blobs replace whatever the
#: prototxt carried for it, which is the order Caffe itself uses.
#:
#: The binary half is `protobuf.jai`'s job. The text half is a parser in this
#: file, and it is the part a real prototxt is most likely to break: they are
#: hand-written files, so they arrive with comments in the middle of a message,
#: angle brackets instead of braces, a colon before a nested message, commas
#: and semicolons between fields, hex and exponent numbers, bare enum
#: identifiers, escapes in a quoted name, and -- most often -- fields this
#: importer has no use for. Unknown fields are skipped by syntax rather than by
#: schema, so a `weight_filler`, a `param { lr_mult: 1 }` or a `transform_param`
#: costs nothing. Both halves land in the same `Message` tree, so everything
#: below this line is written once and does not know which file it came from.
#:
#: Three things about the translation are worth naming.
#:
#: Caffe's `BatchNorm` blobs are not the mean and the variance. They are the
#: mean and the variance each multiplied by the scale factor in the third blob,
#: and the layer divides by it on the way in -- with a zero factor meaning
#: "there are no statistics", not a division by zero. A reader that hands the
#: blobs straight to a normaliser is wrong by whatever that factor happens to
#: be, which for a trained net is the number of batches it saw. Caffe also
#: splits into `BatchNorm` and `Scale` what ONNX fuses into one operator, so an
#: immediately following `Scale` is folded in here.
#:
#: Caffe pools with `ceil` and then, if there is any padding, drops the last
#: window when it would begin inside the padding; and its average pool divides
#: by the window clipped to the *padded* image, which is neither of ONNX's two
#: `count_include_pad` settings when a window hangs off the end. Both are got
#: right by computing the output extent here and handing the executor explicit
#: padding, which needs the input shape -- so this module infers shapes as it
#: goes. Where a shape cannot be inferred it falls back to `ceil_mode`, which
#: differs only by that one trailing window.
#:
#: A Caffe layer may write the blob it reads, and most deploy prototxts do it
#: for every activation. The graph the executor runs needs one producer per
#: name, so each rewrite gets a fresh name -- except the last, which keeps the
#: blob's own name. `graph.inputs` and `graph.outputs` are therefore Caffe's
#: names exactly, and the invented ones never surface.
#:
#: ```
#: let graph = read_net_from_caffe("deploy.prototxt", "weights.caffemodel")
#: defer { free_caffe_weights(graph) }
#: let outputs = run_graph(graph, {"data": blob})
#: ```
import std.io as io
from jaitensor import Tensor
from .graph import Attr, Graph, GraphNode
from .protobuf import WIRE_I32, WIRE_I64, WIRE_LEN, WIRE_VARINT, as_double, as_float, as_string, packed_doubles, packed_floats, packed_varints, parse_message
from .caffe.message import Message
from .caffe.schema import MAX_DEPTH, Spec, layout_of
from .caffe.text import parse_net_text

export { Message, parse_net_text }

#: The same message out of the binary format.
fn decode_message(kind: str, data: bytes, depth: int) -> Message {
    if depth > MAX_DEPTH {
        throw ValueError(f"this caffemodel nests more than {MAX_DEPTH} deep")
    }
    let table = layout_of(kind)
    let message = Message(kind)
    for entry in parse_message(data) {
        let spec = table.by_number.get(entry.number, null)
        if spec is null { continue }
        let field: Spec = spec
        let where = f"{kind}.{field.name}"
        if field.kind == "msg" {
            if entry.wire != WIRE_LEN {
                throw ValueError(f"{where} is a message and arrived as wire type {entry.wire}")
            }
            message.add_child(field.name, decode_message(field.of, entry.data, depth + 1))
        } elif field.kind == "str" {
            message.add_str(field.name, as_string(entry))
        } elif field.kind == "float" {
            if entry.wire == WIRE_LEN {
                message.add_floats(field.name, packed_floats(entry))
            } elif entry.wire == WIRE_I32 {
                message.add_floats(field.name, [as_float(entry)])
            } else {
                throw ValueError(f"{where} is a float and arrived as wire type {entry.wire}")
            }
        } elif field.kind == "double" {
            if entry.wire == WIRE_LEN {
                message.add_floats(field.name, packed_doubles(entry))
            } elif entry.wire == WIRE_I64 {
                message.add_floats(field.name, [as_double(entry)])
            } else {
                throw ValueError(f"{where} is a double and arrived as wire type {entry.wire}")
            }
        } else {
            if entry.wire == WIRE_LEN {
                message.add_ints(field.name, packed_varints(entry))
            } elif entry.wire == WIRE_VARINT {
                message.add_ints(field.name, [entry.varint])
            } else {
                throw ValueError(f"{where} is an integer and arrived as wire type {entry.wire}")
            }
        }
    }
    return message
}

#: One `BlobProto`: the numbers, and whatever shape the file put on them.
class Blob {
    pub let shape: list[int]
    pub let values: list[float]

    pub fn init(self, shape: list[int], values: list[float]) {
        self.shape = shape
        self.values = values
    }
}

#: Read a blob, in either the modern `shape` form or the four-number one that
#: Caffe deprecated but every older caffemodel still uses.
fn read_blob(message: Message, where: str) -> Blob {
    var values = message.floats_of("data")
    if values.len() == 0 { values = message.floats_of("double_data") }
    var shape: list[int] = []
    let declared = message.child("shape")
    if declared is not null {
        let holder: Message = declared
        shape = holder.ints_of("dim")
    } elif message.has("num") or
        message.has("channels") or
        message.has("height") or
        message.has("width") {
        shape = [
            message.int_of("num", 0),
            message.int_of("channels", 0),
            message.int_of("height", 0),
            message.int_of("width", 0),
        ]
    }
    if shape.len() > 0 {
        var volume = 1
        for dimension in shape { volume *= dimension }
        if volume != values.len() {
            throw ValueError(
                f"{where} says its shape is {shape} and carries {values.len()} numbers"
            )
        }
    }
    return Blob(shape, values)
}

#: Caffe writes one value for every spatial axis, or one per axis.
fn axis_pair(values: list[int], fallback: int) -> list[int] {
    if values.len() == 0 { return [fallback, fallback] }
    if values.len() == 1 { return [values[0], values[0]] }
    return [values[0], values[1]]
}

#: A convolution's per-axis value: the `_h`/`_w` pair wins over the repeated
#: field when the file sets either of them, which is Caffe's own order.
fn conv_pair(param: Message, plain: str, tall: str, wide: str, fallback: int) -> list[int] {
    if param.has(tall) or param.has(wide) {
        return [param.int_of(tall, fallback), param.int_of(wide, fallback)]
    }
    return axis_pair(param.ints_of(plain), fallback)
}

#: A pooling layer's, which Caffe reads by a different rule again: the square
#: field wins for the kernel, and only `_h` is tested for the other two.
fn pool_pair(
    param: Message,
    plain: str,
    tall: str,
    wide: str,
    fallback: int,
    square: bool
) -> list[int] {
    if square {
        if param.has(plain) {
            let value = param.int_of(plain, fallback)
            return [value, value]
        }
        return [param.int_of(tall, fallback), param.int_of(wide, fallback)]
    }
    if param.has(tall) { return [param.int_of(tall, fallback), param.int_of(wide, fallback)] }
    let value = param.int_of(plain, fallback)
    return [value, value]
}

#: A convolution's output extent along one axis.
fn convolved(size: int, window: int, stride: int, pad: int, dilation: int) -> int {
    return (size + 2 * pad - (dilation * (window - 1) + 1)) // stride + 1
}

#: Caffe's pooled extent, both axes at once because Caffe's correction is
#: guarded by whether *either* axis is padded.
#:
#: The ceiling is the easy half. The correction after it is the half that gets
#: missed: with padding, a window that would begin inside the trailing padding
#: is dropped, so an input that rounds up to one more window than it can
#: honestly fill produces one fewer.
fn pooled_pair(
    step: Step,
    size: list[int],
    window: list[int],
    stride: list[int],
    pad: list[int],
    ceiling: bool
) -> list[int] {
    var counts: list[int] = []
    for axis in 0..2 {
        if stride[axis] <= 0 {
            throw ValueError(f"{step.label()} pools with a stride of {stride[axis]}")
        }
        let extent = size[axis] + 2 * pad[axis] - window[axis]
        if extent < 0 {
            throw ValueError(
                f"{step.label()} slides a {window[0]}x{window[1]} window over a " +
                    f"{size[0]}x{size[1]} input padded by {pad[0]}x{pad[1]}, which does not fit"
            )
        }
        counts.push((if ceiling { -(-extent // stride[axis]) } else { extent // stride[axis] }) + 1)
    }
    if pad[0] > 0 or pad[1] > 0 {
        for axis in 0..2 {
            if (counts[axis] - 1) * stride[axis] >= size[axis] + pad[axis] { counts[axis] -= 1 }
        }
    }
    return counts
}

#: One Caffe layer, or a `BatchNorm` and the `Scale` folded into it.
class Step {
    pub let layer: Message
    pub let kind: str
    pub let name: str
    pub let bottoms: list[str]
    pub let tops: list[str]
    pub let scale: Message?

    pub fn init(
        self,
        layer: Message,
        kind: str,
        name: str,
        bottoms: list[str],
        tops: list[str],
        scale: Message? = null
    ) {
        self.layer = layer
        self.kind = kind
        self.name = name
        self.bottoms = bottoms
        self.tops = tops
        self.scale = scale
    }

    #: How this layer is named in an error.
    pub fn label(self) -> str {
        if self.name.len() > 0 { return f"the {self.kind} layer {self.name}" }
        return f"an unnamed {self.kind} layer"
    }
}

#: The graph as it is being put together.
class Reading {
    pub var nodes: list[GraphNode]
    pub var weights: dict[str, Tensor]
    pub var inputs: list[str]
    pub var live: dict[str, str]
    pub var shapes: dict[str, list[int]]
    pub var taken: dict[str, bool]
    pub var consumed: dict[str, bool]
    pub var written: dict[str, int]
    pub var expected: dict[str, int]
    pub var declared: dict[str, bool]
    pub var finished: list[str]

    pub fn init(self) {
        self.nodes = []
        self.weights = {}
        self.inputs = []
        self.live = {}
        self.shapes = {}
        self.taken = {}
        self.consumed = {}
        self.written = {}
        self.expected = {}
        self.declared = {}
        self.finished = []
    }

    #: Free everything allocated so far. Called when the import throws, so
    #: that a half-built graph does not leak a model's worth of GPU memory.
    pub fn discard(self) -> void {
        for (_name, tensor) in self.weights.items() { tensor.free() }
        self.weights = {}
    }

    #: A name nothing else has taken. Blob names are reserved up front, so the
    #: only thing this bumps is a name this importer invented.
    pub fn fresh(self, wanted: str) -> str {
        if not wanted in self.taken {
            self.taken[wanted] = true
            return wanted
        }
        var counter = 1
        loop {
            let candidate = f"{wanted}:{counter}"
            if not candidate in self.taken {
                self.taken[candidate] = true
                return candidate
            }
            counter += 1
        }
    }

    pub fn hold(self, name: str, shape: list[int], values: list[float]) -> void {
        self.weights[name] = Tensor(shape, values)
    }

    pub fn emit(
        self,
        op: str,
        name: str,
        ins: list[str],
        outs: list[str],
        attrs: dict[str, Attr] = {}
    ) -> void {
        self.nodes.push(GraphNode(op, name: name, inputs: ins, outputs: outs, attrs: attrs))
    }

    #: The graph name currently holding a Caffe blob.
    pub fn read(self, step: Step, blob: str) -> str {
        let name = self.live.get(blob, null)
        if name is null {
            throw ValueError(
                f"{step.label()} reads {blob}, which no earlier layer writes and the net " +
                    "does not declare as an input"
            )
        }
        let held: str = name
        self.consumed[held] = true
        return held
    }

    #: The graph name a Caffe blob's next value goes to. The last value a blob
    #: takes keeps the blob's own name, so an in-place chain leaves Caffe's
    #: name on the thing a caller asks for -- unless the net declared that name
    #: as an input, in which case nothing may produce it and every version has
    #: to be renamed.
    pub fn write(self, blob: str, shape: list[int]?) -> str {
        let seen = self.written.get(blob, 0) + 1
        self.written[blob] = seen
        let last = seen >= self.expected.get(blob, 1)
        let name = if last and not blob in self.declared {
            self.finished.push(blob)
            blob
        } else {
            let invented = self.fresh(f"{blob}:{seen}")
            if last {
                self.finished.push(invented)
            }
            invented
        }
        if shape is not null {
            let dims: list[int] = shape
            self.shapes[name] = dims
        }
        self.live[blob] = name
        return name
    }

    #: The shape of a graph name, when it was possible to work one out.
    pub fn shape_of(self, name: str) -> list[int]? { return self.shapes.get(name, null) }

    #: The shape of a layer's first input, or `null`.
    pub fn input_shape(self, step: Step, index: int = 0) -> list[int]? {
        if index >= step.bottoms.len() { return null }
        let name = self.live.get(step.bottoms[index], null)
        if name is null { return null }
        let held: str = name
        return self.shapes.get(held, null)
    }

    #: A constant this importer invented -- pad widths, a reshape's dimensions,
    #: a coefficient. Always an initializer, never an input.
    pub fn constant(self, stem: str, shape: list[int], values: list[float]) -> str {
        let name = self.fresh(stem)
        self.hold(name, shape, values)
        return name
    }

    #: One of a layer's weight blobs, reshaped to what the operator needs.
    #:
    #: A prototxt on its own carries no weights, and a graph that names them
    #: anyway is more useful than one that refuses to be built: the name goes
    #: into `graph.inputs` instead, so the structure is complete and the caller
    #: supplies the numbers.
    pub fn weight(
        self,
        step: Step,
        blobs: list[Blob],
        index: int,
        stem: str,
        shape: list[int]?
    ) -> str {
        let name = self.fresh(f"{step.name}/{stem}")
        if index >= blobs.len() {
            self.inputs.push(name)
            return name
        }
        let blob = blobs[index]
        var dims = shape ?? blob.shape
        if dims.len() == 0 { dims = [blob.values.len()] }
        var volume = 1
        for dimension in dims { volume *= dimension }
        if volume != blob.values.len() {
            throw ValueError(
                f"{step.label()} has {blob.values.len()} numbers in blob {index}, and this " +
                    f"layer needs {volume} laid out as {dims}"
            )
        }
        self.hold(name, dims, blob.values)
        return name
    }
}

#: A layer's blobs, in the order Caffe numbers them. By the time this runs the
#: caffemodel's blobs have already replaced whatever the prototxt held.
fn blobs_of(layer: Message, step_name: str) -> list[Blob] {
    var out: list[Blob] = []
    for (index, held) in layer.children_of("blobs").enumerate() {
        out.push(read_blob(held, f"blob {index} of the layer {step_name}"))
    }
    return out
}

fn resolve_axis(axis: int, rank: int, step: Step) -> int {
    let resolved = if axis < 0 { rank + axis } else { axis }
    if resolved < 0 or resolved >= rank {
        throw ValueError(f"{step.label()} names axis {axis}, which is outside rank {rank}")
    }
    return resolved
}

fn need_shape(reading: Reading, step: Step, index: int = 0) -> list[int] {
    let found = reading.input_shape(step, index)
    if found is null {
        throw ValueError(
            f"{step.label()} needs the shape of {step.bottoms[index]}, which this file does " +
                "not declare and nothing above it fixes"
        )
    }
    let dims: list[int] = found
    return dims
}

fn floats_of_ints(values: list[int]) -> list[float] {
    return [float(value) for value in values]
}

fn layer_conv(reading: Reading, step: Step) -> void {
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

fn layer_pool(reading: Reading, step: Step) -> void {
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

fn layer_inner_product(reading: Reading, step: Step) -> void {
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

#: The activations that are one operator with at most one number on them.
fn layer_activation(reading: Reading, step: Step) -> void {
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

fn layer_prelu(reading: Reading, step: Step) -> void {
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

fn layer_lrn(reading: Reading, step: Step) -> void {
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

fn layer_batch_norm(reading: Reading, step: Step) -> void {
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
fn layer_affine(reading: Reading, step: Step) -> void {
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

fn layer_concat(reading: Reading, step: Step) -> void {
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

fn layer_slice(reading: Reading, step: Step) -> void {
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

fn layer_softmax(reading: Reading, step: Step) -> void {
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

fn layer_eltwise(reading: Reading, step: Step) -> void {
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

fn layer_flatten(reading: Reading, step: Step) -> void {
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

fn layer_reshape(reading: Reading, step: Step) -> void {
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

fn layer_power(reading: Reading, step: Step) -> void {
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
fn layer_split(reading: Reading, step: Step) -> void {
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

var _layers: dict[str, fn(Reading, Step) -> void] = {}

#: Every Caffe layer type this importer translates, and what translates it.
fn translators() -> dict[str, fn(Reading, Step) -> void] {
    if _layers.len() > 0 { return _layers }
    _layers["Convolution"] = layer_conv
    _layers["Deconvolution"] = layer_conv
    _layers["Pooling"] = layer_pool
    _layers["InnerProduct"] = layer_inner_product
    _layers["ReLU"] = layer_activation
    _layers["ELU"] = layer_activation
    _layers["Sigmoid"] = layer_activation
    _layers["TanH"] = layer_activation
    _layers["AbsVal"] = layer_activation
    _layers["BNLL"] = layer_activation
    _layers["Dropout"] = layer_activation
    _layers["PReLU"] = layer_prelu
    _layers["LRN"] = layer_lrn
    _layers["BatchNorm"] = layer_batch_norm
    _layers["Scale"] = layer_affine
    _layers["Bias"] = layer_affine
    _layers["Concat"] = layer_concat
    _layers["Slice"] = layer_slice
    _layers["Softmax"] = layer_softmax
    _layers["Eltwise"] = layer_eltwise
    _layers["Flatten"] = layer_flatten
    _layers["Reshape"] = layer_reshape
    _layers["Power"] = layer_power
    _layers["Split"] = layer_split
    return _layers
}

#: The Caffe layer types this importer translates, in alphabetical order.
#:
#: Everything else -- the data layers, the loss layers, and the types a fork of
#: Caffe added to its own copy of the schema -- is refused by name rather than
#: dropped, because a graph quietly missing a step is worse than no graph.
pub fn translated_layers() -> list[str] {
    var names: list[str] = []
    for (name, _handler) in translators().items() { names.push(name) }
    names.sort()
    return names
}

fn convert(reading: Reading, step: Step) -> void {
    let handler = translators().get(step.kind, null)
    if handler is null {
        throw ValueError(f"{step.label()} is of a type this importer does not translate")
    }
    if step.tops.len() == 0 { throw ValueError(f"{step.label()} writes nothing") }
    if step.bottoms.len() == 0 { throw ValueError(f"{step.label()} reads nothing") }
    handler(reading, step)
}

#: Whether a `Scale` may be folded into the `BatchNorm` in front of it.
#:
#: It may when the `Scale` is the learned-parameter form over the channel axis
#: and the value it consumes reaches nobody else. In place -- which is how a
#: deploy prototxt almost always writes the pair -- later layers read the same
#: blob name and so read the scaled value either way, which is what the fused
#: operator produces.
fn foldable(norm: Message, scale: Message, last_read: dict[str, int], position: int) -> bool {
    if scale.str_of("type", "") != "Scale" { return false }
    let tops = norm.strs_of("top")
    let bottoms = scale.strs_of("bottom")
    let written = scale.strs_of("top")
    if tops.len() != 1 or bottoms.len() != 1 or written.len() != 1 { return false }
    if bottoms[0] != tops[0] { return false }
    let param = scale.param("scale_param", "ScaleParameter")
    if param.int_of("axis", 1) != 1 or param.int_of("num_axes", 1) != 1 { return false }
    if written[0] == bottoms[0] { return true }
    return last_read.get(tops[0], -1) <= position + 1
}

#: The layers, with folded pairs collapsed and `Input` layers taken out.
fn plan(layers: list[Message], declared: dict[str, list[int]], order: list[str]) -> list[Step] {
    #: The last layer to read each blob, so that a fold can ask in one lookup
    #: whether the value it would swallow is wanted below the pair.
    var last_read: dict[str, int] = {}
    for (position, layer) in layers.enumerate() {
        for blob in layer.strs_of("bottom") { last_read[blob] = position }
    }

    var steps: list[Step] = []
    var index = 0
    while index < layers.len() {
        let layer = layers[index]
        let kind = layer.str_of("type", "")
        let name = layer.str_of("name", "")
        if kind.len() == 0 {
            throw ValueError(f"the layer {name} does not say what type it is")
        }
        if kind == "Input" {
            let shapes = layer.param("input_param", "InputParameter").children_of("shape")
            for (slot, top) in layer.strs_of("top").enumerate() {
                if not top in order { order.push(top) }
                if shapes.len() == 0 { continue }
                if shapes.len() != 1 and slot >= shapes.len() {
                    throw ValueError(f"the Input layer {name} names more tops than shapes")
                }
                declared[top] = shapes[if shapes.len() == 1 { 0 } else { slot }].ints_of("dim")
            }
            index += 1
            continue
        }
        var scale: Message? = null
        if kind == "BatchNorm" and index + 1 < layers.len() {
            if foldable(layer, layers[index + 1], last_read, index) { scale = layers[index + 1] }
        }
        let tops = if scale is null {
            layer.strs_of("top")
        } else {
            let folded: Message = scale
            folded.strs_of("top")
        }
        steps.push(Step(layer, kind, name, layer.strs_of("bottom"), tops, scale: scale))
        index += if scale is null { 1 } else { 2 }
    }
    return steps
}

#: The inputs a NetParameter declares outside its layers, in either the
#: `input_shape` form or the four-numbers-per-blob one that came before it.
fn declared_inputs(net: Message, declared: dict[str, list[int]], order: list[str]) -> void {
    let names = net.strs_of("input")
    let shapes = net.children_of("input_shape")
    let flat = net.ints_of("input_dim")
    for (index, name) in names.enumerate() {
        order.push(name)
        if index < shapes.len() {
            declared[name] = shapes[index].ints_of("dim")
        } elif flat.len() == 4 * names.len() {
            declared[name] = [flat[4 * index + axis] for axis in 0..4]
        }
    }
}

fn assemble(net: Message) -> Graph {
    if net.children_of("layers").len() > 0 {
        throw ValueError(
            "this file uses the V1 `layers` field, which Caffe replaced with `layer` in " +
                "2015 and this importer does not read"
        )
    }
    var declared: dict[str, list[int]] = {}
    var order: list[str] = []
    declared_inputs(net, declared, order)
    let steps = plan(net.children_of("layer"), declared, order)

    let reading = Reading()
    for name in order {
        reading.taken[name] = true
        reading.declared[name] = true
        reading.live[name] = name
        reading.inputs.push(name)
        let shape = declared.get(name, null)
        if shape is not null {
            let dims: list[int] = shape
            if dims.len() > 0 { reading.shapes[name] = dims }
        }
    }
    for step in steps {
        for blob in step.bottoms { reading.taken[blob] = true }
        for blob in step.tops {
            reading.taken[blob] = true
            reading.expected[blob] = reading.expected.get(blob, 0) + 1
        }
    }

    try {
        for step in steps { convert(reading, step) }
    } catch failure: Error {
        reading.discard()
        throw failure
    }

    #: What comes out is every blob whose last value nothing read. Because the
    #: last write keeps the blob's own name, these are Caffe's names.
    var outputs: list[str] = []
    for name in reading.finished {
        if not reading.consumed.get(name, false) and not name in outputs { outputs.push(name) }
    }
    return Graph(
        nodes: reading.nodes,
        inputs: reading.inputs,
        outputs: outputs,
        initializers: reading.weights
    )
}

#: Put the caffemodel's blobs into the prototxt's layers, by name.
fn merge_weights(net: Message, trained: Message) -> void {
    var byname: dict[str, Message] = {}
    for layer in trained.children_of("layer") { byname[layer.str_of("name", "")] = layer }
    if trained.children_of("layers").len() > 0 {
        throw ValueError(
            "this caffemodel uses the V1 `layers` field, which Caffe replaced with `layer` " +
                "in 2015 and this importer does not read"
        )
    }
    for layer in net.children_of("layer") {
        let name = layer.str_of("name", "")
        let found = byname.get(name, null)
        if found is null { continue }
        let source: Message = found
        let blobs = source.children_of("blobs")
        if blobs.len() == 0 { continue }
        layer.children["blobs"] = blobs
    }
}

#: Free every tensor an imported graph holds.
#:
#: The importer allocates the initializers and nothing else, so this is the
#: whole of a model's GPU footprint.
pub fn free_caffe_weights(graph: Graph) -> void {
    for (_name, tensor) in graph.initializers.items() { tensor.free() }
}

#: Read a Caffe network from a prototxt already in memory, and optionally the
#: caffemodel that goes with it.
#:
#: Without the caffemodel the graph still has every node, and the weights it
#: could not find become names in `graph.inputs` for the caller to feed.
pub fn read_net_from_caffe_bytes(prototxt: bytes, caffemodel: bytes? = null) -> Graph {
    let net = parse_net_text(__prim__.str_decode(prototxt))
    if caffemodel is not null {
        let raw: bytes = caffemodel
        merge_weights(net, decode_message("NetParameter", raw, 0))
    }
    return assemble(net)
}

#: The same, from paths.
pub fn read_net_from_caffe(prototxt: str, caffemodel: str? = null) -> Graph {
    let text = io.read_bytes(prototxt)
    if caffemodel is null { return read_net_from_caffe_bytes(text) }
    let path: str = caffemodel
    return read_net_from_caffe_bytes(text, io.read_bytes(path))
}
