#: Reading a frozen TensorFlow graph.
#:
#: The file is a binary `GraphDef`, which is what `freeze_graph` and
#: `convert_variables_to_constants` write and what OpenCV's
#: `readNetFromTensorflow` takes. It is read with the schemaless wire reader in
#: `protobuf.jai` and translated into the shared IR in `graph.jai`, so a
#: TensorFlow `Conv2D` and a Caffe `Convolution` arrive at the executor as the
#: same `Conv` node.
#:
#: The whole difficulty of this importer is layout, and it is worth being
#: precise about the invariant it maintains, because a graph that imports
#: cleanly and computes plausible nonsense is the failure that costs a day.
#:
#: TensorFlow lays images out `NHWC` and convolution weights `HWIO`. Everything
#: downstream of here is ONNX's `NCHW` and `OIHW`. So:
#:
#:   **A rank-four tensor in the imported graph holds TensorFlow's axis
#:   `TF_AXIS[k]` at its own axis `k`. A tensor of any other rank is exactly
#:   TensorFlow's.**
#:
#: That single rule decides everything else. The graph's inputs and outputs are
#: `NCHW`, so a blob from `blob_from_image` feeds straight in. Convolution and
#: pooling and batch normalisation are `NCHW`-native and need no transpose at
#: all -- only the weights are permuted, once, at import. An operator whose
#: meaning is written in terms of axis *numbers* -- `Concat`, `Pad`, `Mean`,
#: `Squeeze`, `Softmax` -- has its axes renumbered instead of its data moved.
#: A `Transpose` composes its permutation with the two layout changes and stays
#: a single node.
#:
#: Only `Reshape` genuinely has to move data, because it reads the buffer in
#: TensorFlow's row-major order; and so does a reduction that drops some but
#: not all of the spatial axes, because dropping `H` alone leaves `NCW` here
#: where TensorFlow is left with `NWC`. Both are wrapped in explicit
#: `Transpose` nodes, and `survives_in_order` is the predicate that decides.
#:
#: The rule generalises past rank four, and every place that changes a
#: tensor's rank has to apply it explicitly rather than trust the number it
#: lands on. A tensor arrives at a node already obeying the invariant --
#: `NCHW` if its rank was four, TensorFlow's own order otherwise -- and an
#: operator that only renumbers or drops axes (`Squeeze`, a reduction,
#: `ArgMax`) never moves data, so its result is automatically correct in
#: whichever of those two regimes its INPUT was in. The one thing that has to
#: be checked by hand is which regime that result rank calls for: dropping
#: axes off a rank-four `NCHW` tensor keeps the result in the same
#: renumbered-axis world (handled by mapping through `NCHW_AXIS` or
#: transposing first, same as `Reshape`), but dropping axes off a tensor of
#: any OTHER rank leaves it in TensorFlow's own order regardless of what rank
#: it lands on -- and if that landing rank is exactly four, the invariant
#: requires `NCHW`, so a `to_nchw` transpose has to run before the result is
#: tagged rank four. Skipping that step is what left `translate_squeeze` and
#: `translate_reduce` tagging a still-`NHWC` tensor as rank four. Broadcasting
#: has the same rule read backwards: `broadcast_input` brings every operand of
#: an elementwise node into whatever regime the node's OWN result rank calls
#: for, so a rank-four operand has to be un-permuted back to TensorFlow's
#: order whenever the node's result is not itself rank four, the same as a
#: `Reshape` would.
#:
#: Renumbering axes needs the rank of a tensor, which is why this walks the
#: graph in topological order carrying a rank for every name. A rank it cannot
#: work out is an error naming the node, never a guess. So is a file this
#: importer will not read: it is turned away whole, with nothing left
#: allocated behind it, rather than half-translated. What it turns away, and
#: why refusing beats guessing in each case -- a graph frozen in `NCHW`, which
#: only a GPU host writes, would break the invariant above; a `Squeeze` with no
#: `squeeze_dims` on a rank-four tensor leaves a rank nothing static can work
#: out; and the `.pbtxt` configuration OpenCV takes beside the model exists to
#: substitute detection layers this executor has no operators for.
#:
#: `Graph.initializers` is the weights the file carried, already on the GPU,
#: and freeing them is the caller's once it has finished with the graph.
#:
#: ```
#: let net = read_net_from_tensorflow("frozen_inference_graph.pb")
#: defer { for (_name, weight) in net.initializers.items() { weight.free() } }
#: let outputs = run_graph(net, {"image": blob})
#: defer { for (_name, tensor) in outputs.items() { tensor.free() } }
#: ```
import std.io as io
from .graph import Attr, Graph, GraphNode
from .protobuf import Field, WIRE_I32, WIRE_I64, WIRE_LEN, WIRE_VARINT, as_bool, as_double, as_float, as_string, field_by_number, fields_by_number, packed_doubles, packed_floats, packed_varints, parse_message, parse_nested
from jaitensor import Tensor
from std.math import INF, NAN, ldexp

#: `GraphDef.node`.
let GRAPH_NODE = 1

#: `NodeDef`. Field 4 is the device the node was placed on, which inference
#: on this machine has no use for.
let NODE_NAME = 1
let NODE_OP = 2
let NODE_INPUT = 3
let NODE_ATTR = 5

#: `NodeDef.attr` is a `map<string, AttrValue>`, and protobuf encodes a map as
#: a repeated message of key and value.
let ENTRY_KEY = 1
let ENTRY_VALUE = 2

#: `AttrValue`. Exactly one of these is set on any given attribute. Field 9 is
#: a placeholder name and field 10 a function reference, neither of which
#: survives freezing.
let ATTR_LIST = 1
let ATTR_S = 2
let ATTR_I = 3
let ATTR_F = 4
let ATTR_B = 5
let ATTR_TYPE = 6
let ATTR_SHAPE = 7
let ATTR_TENSOR = 8

#: `AttrValue.ListValue` numbers its repeats the way `AttrValue` numbers its
#: singulars, and the only list this importer reads is of integers.
let LIST_I = 3

#: `TensorProto`. Note `int_val` is 7 and `double_val` is 6, which is the way
#: round that is easy to write down backwards.
let TENSOR_DTYPE = 1
let TENSOR_SHAPE = 2
let TENSOR_CONTENT = 4
let TENSOR_FLOAT_VAL = 5
let TENSOR_DOUBLE_VAL = 6
let TENSOR_INT_VAL = 7
let TENSOR_INT64_VAL = 10
let TENSOR_BOOL_VAL = 11
let TENSOR_HALF_VAL = 13
let TENSOR_UINT32_VAL = 16
let TENSOR_UINT64_VAL = 17

#: `TensorShapeProto` and its `Dim`. There is no field 1 on the shape.
let SHAPE_DIM = 2
let SHAPE_UNKNOWN_RANK = 3
let DIM_SIZE = 1

#: `tensorflow.DataType`, which is not ONNX's numbering and has to be mapped
#: before it can be handed to `Cast`.
let DT_FLOAT = 1
let DT_DOUBLE = 2
let DT_INT32 = 3
let DT_UINT8 = 4
let DT_INT16 = 5
let DT_INT8 = 6
let DT_STRING = 7
let DT_COMPLEX64 = 8
let DT_INT64 = 9
let DT_BOOL = 10
let DT_BFLOAT16 = 14
let DT_UINT16 = 17
let DT_HALF = 19
let DT_UINT32 = 22
let DT_UINT64 = 23

#: Bytes per element of `tensor_content`, for the dtypes a graph can hold
#: numbers in. A dtype missing here has no raw form this reader accepts.
let DTYPE_WIDTHS: dict[int, int] = {
    DT_FLOAT: 4,
    DT_DOUBLE: 8,
    DT_INT32: 4,
    DT_UINT8: 1,
    DT_INT16: 2,
    DT_INT8: 1,
    DT_INT64: 8,
    DT_BOOL: 1,
    DT_BFLOAT16: 2,
    DT_UINT16: 2,
    DT_HALF: 2,
    DT_UINT32: 4,
    DT_UINT64: 8,
}

#: The dtypes whose values are whole numbers, so that an axis or a shape read
#: out of one is exact rather than a float that happens to look like an
#: integer.
let WHOLE_DTYPES = [
    DT_INT32,
    DT_UINT8,
    DT_INT16,
    DT_INT8,
    DT_INT64,
    DT_BOOL,
    DT_UINT16,
    DT_UINT32,
    DT_UINT64,
]

#: What each `tensorflow.DataType` is called in an error.
let DTYPE_NAMES: dict[int, str] = {
    DT_FLOAT: "float32",
    DT_DOUBLE: "float64",
    DT_INT32: "int32",
    DT_UINT8: "uint8",
    DT_INT16: "int16",
    DT_INT8: "int8",
    DT_STRING: "string",
    DT_COMPLEX64: "complex64",
    DT_INT64: "int64",
    DT_BOOL: "bool",
    DT_BFLOAT16: "bfloat16",
    DT_UINT16: "uint16",
    DT_HALF: "float16",
    DT_UINT32: "uint32",
    DT_UINT64: "uint64",
}

#: `tensorflow.DataType` to the `onnx.TensorProto.DataType` that `Cast` reads.
let ONNX_DTYPES: dict[int, int] = {
    DT_FLOAT: 1,
    DT_DOUBLE: 11,
    DT_INT32: 6,
    DT_UINT8: 2,
    DT_INT16: 5,
    DT_INT8: 3,
    DT_INT64: 7,
    DT_BOOL: 9,
    DT_BFLOAT16: 16,
    DT_UINT16: 4,
    DT_HALF: 10,
    DT_UINT32: 12,
    DT_UINT64: 13,
}

#: Axis `k` of an imported rank-four tensor holds TensorFlow's axis
#: `TF_AXIS[k]`: batch, then channels, then the two spatial axes. Read the
#: other way round it is also the permutation that turns an `NHWC` tensor into
#: an `NCHW` one, which is what a `Transpose` node gets given.
let TF_AXIS = [0, 3, 1, 2]

#: The inverse. TensorFlow's axis `a` is axis `NCHW_AXIS[a]` here, and the
#: list is equally the permutation back to `NHWC`.
let NCHW_AXIS = [0, 2, 3, 1]

#: Nodes that carry no value and are dropped rather than translated. `NoOp` and
#: `Assert` exist for their control edges, which this importer discards along
#: with every other control dependency.
let DISCARDED = ["NoOp", "Assert", "Print", "PrintV2", "SaveV2", "RestoreV2"]

#: TensorFlow operators that are the identity at inference time.
let PASSTHROUGH = ["Identity", "StopGradient", "CheckNumerics", "Snapshot", "PreventGradient"]

#: Unary TensorFlow operators whose ONNX name is all that changes.
let UNARY_OPS: dict[str, str] = {
    "Relu": "Relu",
    "Elu": "Elu",
    "Sigmoid": "Sigmoid",
    "Tanh": "Tanh",
    "Softplus": "Softplus",
    "Exp": "Exp",
    "Log": "Log",
    "Sqrt": "Sqrt",
    "Abs": "Abs",
    "Neg": "Neg",
    "Erf": "Erf",
    "LogicalNot": "Not",
}

#: Binary TensorFlow operators that are elementwise with numpy broadcasting.
let BINARY_OPS: dict[str, str] = {
    "Add": "Add",
    "AddV2": "Add",
    "BiasAdd": "Add",
    "BiasAddV1": "Add",
    "Sub": "Sub",
    "Mul": "Mul",
    "Div": "Div",
    "RealDiv": "Div",
    "Maximum": "Max",
    "Minimum": "Min",
    "Pow": "Pow",
    "Equal": "Equal",
    "Greater": "Greater",
    "Less": "Less",
    "LogicalAnd": "And",
    "LogicalOr": "Or",
}

#: The four reductions, with the ONNX name and the `row_reduce` they become.
let REDUCE_OPS: dict[str, str] = {
    "Mean": "ReduceMean",
    "Sum": "ReduceSum",
    "Max": "ReduceMax",
    "Min": "ReduceMin",
}

#: Every value of a repeated varint field, whether the writer packed it or
#: not.
#:
#: TensorFlow's protos are proto3, so a repeated scalar is packed by default
#: and arrives as one length-delimited run -- but a `.pb` written by an older
#: exporter, or merged from two encodings, can carry the same field number
#: several times and unpacked. Reading only the first, or assuming the wire
#: type, loses values from real files.
fn repeated_varints(fields: list[Field], number: int, where: str) -> list[int] {
    var out: list[int] = []
    for field in fields_by_number(fields, number) {
        if field.wire == WIRE_LEN {
            for value in packed_varints(field) { out.push(value) }
        } elif field.wire == WIRE_VARINT {
            out.push(field.varint)
        } else {
            throw ValueError(f"{where} holds field {number} in a wire type varints cannot be in")
        }
    }
    return out
}

#: Every value of a repeated `float` field, packed or not.
fn repeated_floats(fields: list[Field], number: int, where: str) -> list[float] {
    var out: list[float] = []
    for field in fields_by_number(fields, number) {
        if field.wire == WIRE_LEN {
            for value in packed_floats(field) { out.push(value) }
        } elif field.wire == WIRE_I32 {
            out.push(as_float(field))
        } else {
            throw ValueError(f"{where} holds field {number} in a wire type floats cannot be in")
        }
    }
    return out
}

#: Every value of a repeated `double` field, packed or not.
fn repeated_doubles(fields: list[Field], number: int, where: str) -> list[float] {
    var out: list[float] = []
    for field in fields_by_number(fields, number) {
        if field.wire == WIRE_LEN {
            for value in packed_doubles(field) { out.push(value) }
        } elif field.wire == WIRE_I64 {
            out.push(as_double(field))
        } else {
            throw ValueError(f"{where} holds field {number} in a wire type doubles cannot be in")
        }
    }
    return out
}

#: `width` bytes of `data` at `at`, little-endian and unsigned.
#:
#: `tensor_content` is the raw memory of the tensor as the writing machine had
#: it, so this assumes a little-endian writer. Every machine TensorFlow ships
#: for is one; a graph frozen on a big-endian host would need its own reader
#: and would not be silently wrong here, because the exponents would come out
#: as noise.
fn little_endian(data: bytes, at: int, width: int) -> int {
    var value = 0
    for index in 0..width { value = value | data[at + index] << 8 * index }
    return value
}

#: The same bytes read as a two's-complement signed value.
fn signed_endian(data: bytes, at: int, width: int) -> int {
    let value = little_endian(data, at, width)
    let sign = 1 << 8 * width - 1
    if width < 8 and value & sign != 0 { return value - (sign << 1) }
    return value
}

#: The binary32 whose bit pattern is the low thirty-two bits of `bits`.
#:
#: A biased exponent of zero means the leading one is absent, so a denormal
#: scales by `2**-149` outright rather than having the hidden bit added first.
fn f32_from_bits(bits: int) -> float {
    let exponent = bits >> 23 & 0xFF
    let fraction = bits & 0x7FFFFF
    var value = 0.0
    if exponent == 0 {
        value = ldexp(float(fraction), -149)
    } elif exponent == 255 {
        value = if fraction == 0 { INF } else { NAN }
    } else {
        value = ldexp(float(fraction) + 8388608.0, exponent - 150)
    }
    if bits >> 31 & 1 == 1 { return -value }
    return value
}

#: The binary64 whose bit pattern is `bits`, which arrives as the signed
#: reinterpretation the eight wire bytes read as.
fn f64_from_bits(bits: int) -> float {
    let exponent = bits >> 52 & 0x7FF
    let fraction = bits & 0xFFFFFFFFFFFFF
    var value = 0.0
    if exponent == 0 {
        value = ldexp(float(fraction), -1074)
    } elif exponent == 2047 {
        value = if fraction == 0 { INF } else { NAN }
    } else {
        value = ldexp(float(fraction) + 4503599627370496.0, exponent - 1075)
    }
    if bits < 0 { return -value }
    return value
}

#: The binary16 whose bit pattern is the low sixteen bits of `bits`.
fn f16_from_bits(bits: int) -> float {
    let exponent = bits >> 10 & 0x1F
    let fraction = bits & 0x3FF
    var value = 0.0
    if exponent == 0 {
        value = ldexp(float(fraction), -24)
    } elif exponent == 31 {
        value = if fraction == 0 { INF } else { NAN }
    } else {
        value = ldexp(float(fraction) + 1024.0, exponent - 25)
    }
    if bits >> 15 & 1 == 1 { return -value }
    return value
}

#: `bfloat16` is the top half of a binary32, so widening it is a shift.
fn bf16_from_bits(bits: int) -> float { return f32_from_bits((bits & 0xFFFF) << 16) }

#: One tensor a `Const` node carried, decoded off the wire.
#:
#: Both readings are kept because both are needed and neither can be recovered
#: from the other once it is lost: an axis list has to be exact, and a weight
#: has to be a float. Which one is filled is `is_whole`.
class TfTensor {
    pub let dtype: int
    pub let shape: list[int]
    pub let floats: list[float]
    pub let whole: list[int]
    pub let is_whole: bool

    pub fn init(
        self,
        dtype: int,
        shape: list[int],
        floats: list[float],
        whole: list[int],
        is_whole: bool
    ) {
        self.dtype = dtype
        self.shape = shape
        self.floats = floats
        self.whole = whole
        self.is_whole = is_whole
    }

    pub fn rank(self) -> int { return self.shape.len() }

    pub fn count(self) -> int {
        var total = 1
        for dimension in self.shape { total *= dimension }
        return total
    }

    #: The elements as floats, which is the only thing a tensor on the GPU
    #: holds.
    pub fn values(self) -> list[float] {
        if not self.is_whole { return self.floats }
        return [float(value) for value in self.whole]
    }

    #: The elements as exact integers, for an axis, a shape or a pad width.
    pub fn ints(self, where: str) -> list[int] {
        if self.is_whole { return self.whole }
        throw ValueError(f"{where} needs whole numbers, and this constant holds {self.name()}")
    }

    pub fn name(self) -> str { return DTYPE_NAMES.get(self.dtype, f"data type {self.dtype}") }
}

#: The dimensions of a `TensorShapeProto`, with `-1` for an unknown one.
fn read_shape_proto(fields: list[Field], where: str) -> list[int]? {
    let unknown = field_by_number(fields, SHAPE_UNKNOWN_RANK)
    if unknown is not null and as_bool(unknown) { return null }
    var dims: list[int] = []
    for entry in fields_by_number(fields, SHAPE_DIM) {
        let inner = parse_nested(entry)
        #: An absent size is proto3's zero, not an unknown axis. TensorFlow
        #: writes `-1` for unknown, and writes it explicitly because it is not
        #: the default.
        let size = field_by_number(inner, DIM_SIZE)
        dims.push(if size is null { 0 } else { size.varint })
    }
    return dims
}

#: The values a `TensorProto` holds outside `tensor_content`, in whichever
#: type-specific field its dtype uses.
fn typed_values(fields: list[Field], dtype: int, where: str) -> tuple[list[float], list[int]] {
    if dtype == DT_FLOAT { return (repeated_floats(fields, TENSOR_FLOAT_VAL, where), []) }
    if dtype == DT_DOUBLE { return (repeated_doubles(fields, TENSOR_DOUBLE_VAL, where), []) }
    if dtype == DT_INT64 { return ([], repeated_varints(fields, TENSOR_INT64_VAL, where)) }
    if dtype == DT_BOOL { return ([], repeated_varints(fields, TENSOR_BOOL_VAL, where)) }
    #: `half_val` is a repeated `int32` holding the sixteen bits, not a float
    #: field, because protobuf has no half.
    if dtype == DT_HALF {
        return (
            [f16_from_bits(bits) for bits in repeated_varints(fields, TENSOR_HALF_VAL, where)],
            []
        )
    }
    if dtype == DT_BFLOAT16 {
        return (
            [bf16_from_bits(bits) for bits in repeated_varints(fields, TENSOR_HALF_VAL, where)],
            []
        )
    }
    #: `uint32` and `uint64` ride in their own fields -- `int_val` is an
    #: `int32` and cannot hold either without losing the top of the range --
    #: so these two have to be pulled out of `WHOLE_DTYPES` before it falls
    #: through to `int_val` below.
    if dtype == DT_UINT32 { return ([], repeated_varints(fields, TENSOR_UINT32_VAL, where)) }
    if dtype == DT_UINT64 { return ([], repeated_varints(fields, TENSOR_UINT64_VAL, where)) }
    #: Every remaining integer width rides in `int_val`, which is an `int32`.
    if dtype in WHOLE_DTYPES { return ([], repeated_varints(fields, TENSOR_INT_VAL, where)) }
    if dtype == DT_STRING {
        throw ValueError(f"{where} holds strings, which this importer has no tensor for")
    }
    throw ValueError(
        f"{where} holds {DTYPE_NAMES.get(dtype, f'data type {dtype}')}, which this importer does not read"
    )
}

#: `tensor_content`, which is the tensor's raw memory rather than a repeated
#: field, split into elements.
fn raw_values(data: bytes, dtype: int, where: str) -> tuple[list[float], list[int]] {
    let width = DTYPE_WIDTHS.get(dtype, 0)
    if width == 0 {
        throw ValueError(
            f"{where} holds {DTYPE_NAMES.get(dtype, f'data type {dtype}')} as raw bytes, which this importer does not read"
        )
    }
    if data.len() % width != 0 {
        throw ValueError(
            f"{where} has {data.len()} bytes, which is not whole {width}-byte elements"
        )
    }
    let count = data.len() // width
    var floats: list[float] = []
    var whole: list[int] = []
    for index in 0..count {
        let at = index * width
        if dtype == DT_FLOAT {
            floats.push(f32_from_bits(little_endian(data, at, 4)))
        } elif dtype == DT_DOUBLE {
            floats.push(f64_from_bits(little_endian(data, at, 8)))
        } elif dtype == DT_HALF {
            floats.push(f16_from_bits(little_endian(data, at, 2)))
        } elif dtype == DT_BFLOAT16 {
            floats.push(bf16_from_bits(little_endian(data, at, 2)))
        } elif dtype == DT_UINT8 or dtype == DT_UINT16 or dtype == DT_UINT32 or dtype == DT_BOOL {
            whole.push(little_endian(data, at, width))
        } else {
            whole.push(signed_endian(data, at, width))
        }
    }
    return (floats, whole)
}

#: A `TensorProto`, whichever of its three shapes it arrived in.
#:
#: TensorFlow writes a tensor in one of four shapes, and a reader that knows
#: only the first two rejects real files. There is raw memory in
#: `tensor_content`; there is one repeated typed field with an element each;
#: there is a single value standing for a tensor whose elements are all equal;
#: and there is *nothing at all*, which means every element is zero. The last
#: is not a curiosity -- grappler's constant folding writes a zero axis index
#: that way, and this reader was rejecting a frozen `reduce_max(x, axis=[0])`
#: until a fixture produced one.
#:
#: Both of those last two shapes turn a handful of bytes into `count`
#: materialised elements, and `count` is the declared shape's product with
#: nothing behind it yet -- exactly the number a hostile file gets to pick for
#: free. `total_bytes`, the size of this `TensorProto` on the wire, is what
#: bounds that: a real splat or a real all-zero tensor is small because the
#: encoding that produces one is small, and a declared count past what the
#: message could have paid for even at one byte an element is a shape this
#: reader will not honour.
fn read_tensor_proto(fields: list[Field], total_bytes: int, where: str) -> TfTensor {
    let dtype_field = field_by_number(fields, TENSOR_DTYPE)
    let dtype = if dtype_field is null { DT_FLOAT } else { dtype_field.varint }
    var shape: list[int] = []
    let shape_field = field_by_number(fields, TENSOR_SHAPE)
    if shape_field is not null {
        let dims = read_shape_proto(parse_nested(shape_field), where)
        if dims is null { throw ValueError(f"{where} has a tensor of unknown rank") }
        shape = dims
    }
    var count = 1
    for dimension in shape {
        if dimension < 0 { throw ValueError(f"{where} has a tensor of shape {shape}") }
        count *= dimension
    }
    let is_whole = dtype in WHOLE_DTYPES
    let content = field_by_number(fields, TENSOR_CONTENT)
    var floats: list[float] = []
    var whole: list[int] = []
    if content is not null and content.data.len() > 0 {
        let (raw_floats, raw_whole) = raw_values(content.data, dtype, where)
        floats = raw_floats
        whole = raw_whole
    } else {
        let (typed_floats, typed_whole) = typed_values(fields, dtype, where)
        floats = typed_floats
        whole = typed_whole
    }
    let held = if is_whole { whole.len() } else { floats.len() }
    if held == count { return TfTensor(dtype, shape, floats, whole, is_whole) }
    if held == 1 or held == 0 {
        if count > total_bytes {
            throw ValueError(
                f"{where} declares a tensor of shape {shape} -- {count} elements -- backed by " +
                    f"a message of only {total_bytes} bytes, which is not enough to have written " +
                    "even one byte per element"
            )
        }
        if held == 1 {
            if is_whole {
                return TfTensor(dtype, shape, [], [whole[0] for _slot in 0..count], true)
            }
            return TfTensor(dtype, shape, [floats[0] for _slot in 0..count], [], false)
        }
        if is_whole { return TfTensor(dtype, shape, [], [0 for _slot in 0..count], true) }
        return TfTensor(dtype, shape, [0.0 for _slot in 0..count], [], false)
    }
    throw ValueError(f"{where} holds {held} values for a tensor of shape {shape}")
}

#: One `NodeDef`, with its input names already canonicalised.
#:
#: A TensorFlow input is `node`, `node:k` for the `k`-th output of a node with
#: several, or `^node` for a control dependency. Control edges carry no value
#: and are dropped here: they order side effects, and there are none left in a
#: frozen inference graph. `node:0` is the same name as `node`, and anything
#: past `:0` is kept verbatim so that a graph reading a second output fails by
#: naming a value nothing produces rather than by silently reading the first.
class TfNode {
    pub let name: str
    pub let op: str
    pub let inputs: list[str]
    pub let attrs: dict[str, list[Field]]

    pub fn init(self, name: str, op: str, inputs: list[str], attrs: dict[str, list[Field]]) {
        self.name = name
        self.op = op
        self.inputs = inputs
        self.attrs = attrs
    }

    pub fn label(self) -> str { return f"the {self.op} node {self.name}" }
}

fn canonical_input(raw: str) -> str? {
    if raw.starts_with("^") { return null }
    let colon = raw.rfind(":")
    if colon < 0 { return raw }
    let index = raw.slice(colon + 1).to_int()
    if index is not null and index == 0 { return raw.slice(0, colon) }
    return raw
}

fn read_node(fields: list[Field], index: int) -> TfNode {
    let name_field = field_by_number(fields, NODE_NAME)
    let name = if name_field is null { f"node{index}" } else { as_string(name_field) }
    let op_field = field_by_number(fields, NODE_OP)
    if op_field is null { throw ValueError(f"the node {name} names no operator") }
    var inputs: list[str] = []
    for entry in fields_by_number(fields, NODE_INPUT) {
        let cleaned = canonical_input(as_string(entry))
        if cleaned is not null { inputs.push(cleaned) }
    }
    var attrs: dict[str, list[Field]] = {}
    for entry in fields_by_number(fields, NODE_ATTR) {
        let pair = parse_nested(entry)
        let key = field_by_number(pair, ENTRY_KEY)
        let value = field_by_number(pair, ENTRY_VALUE)
        if key is null { continue }
        attrs[as_string(key)] = if value is null { [] } else { parse_nested(value) }
    }
    return TfNode(name, as_string(op_field), inputs, attrs)
}

fn attr_of(node: TfNode, name: str) -> list[Field]? { return node.attrs.get(name, null) }

fn attr_string(node: TfNode, name: str, fallback: str) -> str {
    let found = attr_of(node, name)
    if found is null { return fallback }
    let value = field_by_number(found, ATTR_S)
    if value is null { return fallback }
    return as_string(value)
}

fn attr_int(node: TfNode, name: str, fallback: int) -> int {
    let found = attr_of(node, name)
    if found is null { return fallback }
    let value = field_by_number(found, ATTR_I)
    if value is null { return fallback }
    return value.varint
}

fn attr_float(node: TfNode, name: str, fallback: float) -> float {
    let found = attr_of(node, name)
    if found is null { return fallback }
    let value = field_by_number(found, ATTR_F)
    if value is null { return fallback }
    return as_float(value)
}

fn attr_bool(node: TfNode, name: str, fallback: bool) -> bool {
    let found = attr_of(node, name)
    if found is null { return fallback }
    let value = field_by_number(found, ATTR_B)
    if value is null { return fallback }
    return as_bool(value)
}

fn attr_type(node: TfNode, name: str, fallback: int) -> int {
    let found = attr_of(node, name)
    if found is null { return fallback }
    let value = field_by_number(found, ATTR_TYPE)
    if value is null { return fallback }
    return value.varint
}

#: An `AttrValue` holding a `ListValue` of integers, which is how every
#: per-axis attribute -- `strides`, `ksize`, `dilations`, `squeeze_dims` --
#: arrives.
fn attr_ints(node: TfNode, name: str) -> list[int]? {
    let found = attr_of(node, name)
    if found is null { return null }
    let listed = field_by_number(found, ATTR_LIST)
    if listed is null { return null }
    return repeated_varints(parse_nested(listed), LIST_I, node.label())
}

#: The `TensorShapeProto` a `Placeholder` declares, or null when it declares
#: nothing this reader can use.
fn attr_shape(node: TfNode, name: str) -> list[int]? {
    let found = attr_of(node, name)
    if found is null { return null }
    let value = field_by_number(found, ATTR_SHAPE)
    if value is null { return null }
    return read_shape_proto(parse_nested(value), node.label())
}

fn attr_tensor(node: TfNode, name: str) -> TfTensor? {
    let found = attr_of(node, name)
    if found is null { return null }
    let value = field_by_number(found, ATTR_TENSOR)
    if value is null { return null }
    return read_tensor_proto(parse_nested(value), value.data.len(), node.label())
}

#: `values` laid out row-major for `shape`, reordered so that the source's
#: axis `perm[k]` becomes axis `k`.
fn permuted(values: list[float], shape: list[int], perm: list[int]) -> list[float] {
    let rank = shape.len()
    var strides = [1 for _axis in 0..rank]
    var axis = rank - 2
    while axis >= 0 {
        strides[axis] = strides[axis + 1] * shape[axis + 1]
        axis -= 1
    }
    let dims = [shape[perm[slot]] for slot in 0..rank]
    let steps = [strides[perm[slot]] for slot in 0..rank]
    var total = 1
    for dimension in dims { total *= dimension }
    var index = [0 for _axis in 0..rank]
    var out: list[float] = []
    for _slot in 0..total {
        var offset = 0
        for k in 0..rank { offset += index[k] * steps[k] }
        out.push(values[offset])
        var k = rank - 1
        while k >= 0 {
            index[k] += 1
            if index[k] < dims[k] { break }
            index[k] = 0
            k -= 1
        }
    }
    return out
}

fn reordered(shape: list[int], perm: list[int]) -> list[int] {
    return [shape[perm[slot]] for slot in 0..perm.len()]
}

#: Whether dropping TensorFlow's axes `dropped` out of a rank-four tensor
#: leaves the survivors in the same order here as it does in TensorFlow.
#:
#: Reducing or squeezing away `H` and `W` together leaves `NC` both ways, and
#: so the operator runs on the stored tensor with nothing but its axis numbers
#: changed. Dropping `H` alone leaves `NCW` here and `NWC` there, and then the
#: tensor has to be put back into TensorFlow's order first. Getting this wrong
#: produces a correctly shaped output with its values shuffled, which is the
#: hardest kind of wrong to notice.
fn survives_in_order(dropped: list[int]) -> bool {
    var stored: list[int] = []
    for axis in TF_AXIS {
        if not axis in dropped { stored.push(axis) }
    }
    var native: list[int] = []
    for axis in 0..4 {
        if not axis in dropped { native.push(axis) }
    }
    return stored == native
}

#: Whether a value that started at `from_rank` and had axes dropped down to
#: `to_rank` needs a transpose into `NCHW` before its new rank is recorded.
#:
#: Only a starting rank of four is ever stored `NCHW`; every other rank is
#: exactly TensorFlow's own order by the invariant this module keeps, and
#: dropping axes never reorders the survivors. So a result landing at rank
#: four out of a starting rank that was NOT four is still in TensorFlow's
#: order -- NHWC in everything but name -- and has to be turned into `NCHW`
#: before anything downstream treats it as an image, the same as `Reshape`
#: already does. A starting rank of four is handled by its own caller, by
#: renumbering axes or transposing the input first, so this only ever answers
#: yes for the other case.
fn lands_in_nhwc(from_rank: int, to_rank: int) -> bool { return from_rank != 4 and to_rank == 4 }

#: The translation, one TensorFlow node at a time.
#:
#: Nodes are walked in topological order so that the rank of every name is
#: known before the node that needs to renumber its axes is reached. Constants
#: never become nodes: a weight becomes an initializer, and a shape, an axis
#: list or a pad width is read at import and folded into an attribute, which
#: is why nothing downstream has to evaluate a `Const`.
class Importer {
    pub var ranks: dict[str, int]
    pub var consts: dict[str, TfTensor]
    pub var dropped: dict[str, bool]
    pub var produced: list[GraphNode]
    pub var weights: dict[str, Tensor]
    pub var graph_inputs: list[str]

    pub fn init(self) {
        self.ranks = {}
        self.consts = {}
        self.dropped = {}
        self.produced = []
        self.weights = {}
        self.graph_inputs = []
    }

    #: Release every initializer built so far. Called when translation throws
    #: part of the way through, so that a rejected file leaves no tensors on
    #: the GPU.
    pub fn discard(self) -> void {
        for (_name, tensor) in self.weights.items() { tensor.free() }
        self.weights = {}
    }

    pub fn emit(self, op: str, name: str, ins: list[str], attrs: dict[str, Attr] = {}) -> str {
        self.produced.push(GraphNode(op, name: name, inputs: ins, outputs: [name], attrs: attrs))
        return name
    }

    #: An initializer under a name derived from the constant and the shape it
    #: was needed in, so that one `Const` feeding a convolution and an addition
    #: becomes two tensors rather than one that is wrong for one of them.
    pub fn intern(self, key: str, shape: list[int], values: list[float]) -> str {
        if key in self.weights { return key }
        self.weights[key] = Tensor(shape, values)
        return key
    }

    pub fn const_of(self, name: str) -> TfTensor? { return self.consts.get(name, null) }

    #: The rank the file gives a name, refusing to guess when it has none.
    pub fn rank_of(self, name: str, node: TfNode) -> int {
        if self.dropped.get(name, false) {
            throw ValueError(f"{node.label()} reads {name}, which carries no value")
        }
        let found = self.ranks.get(name, -2)
        if found == -2 {
            throw ValueError(f"{node.label()} reads {name}, which nothing in the file produces")
        }
        if found < 0 {
            throw ValueError(
                f"{node.label()} reads {name}, whose rank the file does not say -- and this " +
                    "importer needs it to know which axes are which"
            )
        }
        return found
    }

    #: The rank of a name, or -1 when it is not known. For the places that can
    #: carry on without it.
    pub fn loose_rank(self, name: str) -> int { return self.ranks.get(name, -1) }

    #: A constant read as exact integers, for an axis list or a shape.
    pub fn const_ints(self, name: str, node: TfNode, what: str) -> list[int] {
        let held = self.const_of(name)
        if held is null {
            throw ValueError(
                f"{node.label()} computes its {what} rather than holding it as a constant, " +
                    "which this importer does not follow"
            )
        }
        let tensor: TfTensor = held
        return tensor.ints(f"{node.label()}'s {what}")
    }

    #: The name to read a value input by.
    #:
    #: A constant becomes an initializer laid out to the invariant: rank four
    #: goes to `NCHW`, and TensorFlow's rank-zero scalar becomes the `[1]` that
    #: the executor uses in its place.
    pub fn value_input(self, name: str, node: TfNode) -> str {
        let held = self.const_of(name)
        if held is null {
            if self.dropped.get(name, false) {
                throw ValueError(f"{node.label()} reads {name}, which carries no value")
            }
            if self.ranks.get(name, -2) == -2 {
                throw ValueError(f"{node.label()} reads {name}, which nothing in the file produces")
            }
            return name
        }
        let tensor: TfTensor = held
        #: One role, one name: a constant has one rank, so the shape this
        #: hands back is the same every time and interning it once is safe.
        #: A constant read as weights instead goes under a name of its own,
        #: which is what keeps one `Const` feeding a convolution and an
        #: addition from having to be laid out two ways at once.
        if tensor.rank() == 4 {
            return self.intern(
                name + "$data",
                reordered(tensor.shape, TF_AXIS),
                permuted(tensor.values(), tensor.shape, TF_AXIS)
            )
        }
        if tensor.rank() == 0 { return self.intern(name + "$data", [1], tensor.values()) }
        return self.intern(name + "$data", tensor.shape, tensor.values())
    }

    pub fn to_nhwc(self, name: str, tag: str) -> str {
        return self.emit("Transpose", tag, [name], {"perm": Attr.of_ints(NCHW_AXIS)})
    }

    pub fn to_nchw(self, name: str, tag: str) -> str {
        return self.emit("Transpose", tag, [name], {"perm": Attr.of_ints(TF_AXIS)})
    }

    #: An operand of an elementwise node, put where numpy broadcasting will
    #: line it up once every operand agrees on which of the two regimes the
    #: node's result -- rank `want` -- is in.
    #:
    #: When `want` is four the result is `NCHW`, and TensorFlow aligns a
    #: shorter shape against the *trailing* axes, so a bias of `[C]` meets
    #: `[N, H, W, C]` on the channels. Stored `NCHW`, the channels are axis
    #: one, so the bias has to become `[1, C, 1, 1]`. Padding the shape with
    #: leading ones and then permuting it is the same rule for every rank, and
    #: for a constant it is done to the initializer rather than with nodes.
    #:
    #: When `want` is not four the result stays in TensorFlow's own order, and
    #: the executor's own broadcasting already aligns trailing axes exactly
    #: the way TensorFlow does -- so every operand can be handed over as
    #: `value_input` gives it, MINUS the one case `value_input` gets backwards
    #: for this purpose: a rank-four operand, which `value_input` always
    #: permutes into `NCHW` on the assumption it is going somewhere that wants
    #: that. Here it is not, so a rank-four operand has to be put back into
    #: TensorFlow's order first, the same conversion `to_nhwc` does for an
    #: intermediate value, or read straight out of the constant's own
    #: un-permuted shape.
    pub fn broadcast_input(self, name: str, node: TfNode, want: int) -> str {
        let rank = self.rank_of(name, node)
        if want != 4 {
            if rank != 4 { return self.value_input(name, node) }
            let held = self.const_of(name)
            if held is not null {
                let tensor: TfTensor = held
                return self.intern(name + "$flat", tensor.shape, tensor.values())
            }
            let source = self.value_input(name, node)
            return self.to_nhwc(source, f"{node.name}$flat${name}")
        }
        if rank == 4 { return self.value_input(name, node) }
        if rank > 4 {
            throw ValueError(
                f"{node.label()} broadcasts a rank-{rank} operand against a rank-four one, " +
                    "which this importer's layout does not cover"
            )
        }
        #: The executor has no rank-zero tensor, so a scalar is already `[1]`
        #: and needs one fewer axis inserted than its TensorFlow rank suggests.
        let stored = max(rank, 1)
        let held = self.const_of(name)
        if held is not null {
            let tensor: TfTensor = held
            var padded = [1 for _slot in 0..4 - stored]
            for dimension in tensor.shape { padded.push(dimension) }
            if rank == 0 { padded.push(1) }
            return self.intern(
                name + "$bcast",
                reordered(padded, TF_AXIS),
                permuted(tensor.values(), padded, TF_AXIS)
            )
        }
        let source = self.value_input(name, node)
        let axes = [axis for axis in 0..4 - stored]
        let grown = self.emit(
            "Unsqueeze",
            f"{node.name}$grow${name}",
            [source],
            {"axes": Attr.of_ints(axes)}
        )
        return self.to_nchw(grown, f"{node.name}$align${name}")
    }
}

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

fn translate_conv(im: Importer, node: TfNode, depthwise: bool) -> void {
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

fn translate_pool(im: Importer, node: TfNode, is_max: bool) -> void {
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

fn translate_batch_norm(im: Importer, node: TfNode) -> void {
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

fn translate_matmul(im: Importer, node: TfNode) -> void {
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

fn translate_elementwise(im: Importer, node: TfNode, op: str) -> void {
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

fn translate_unary(im: Importer, node: TfNode, op: str, attrs: dict[str, Attr] = {}) -> void {
    if node.inputs.len() < 1 { throw ValueError(f"{node.label()} needs an input") }
    let x = im.value_input(node.inputs[0], node)
    let _emitted = im.emit(op, node.name, [x], attrs)
    im.ranks[node.name] = im.loose_rank(node.inputs[0])
}

fn translate_softmax(im: Importer, node: TfNode, op: str) -> void {
    if node.inputs.len() < 1 { throw ValueError(f"{node.label()} needs an input") }
    let rank = im.rank_of(node.inputs[0], node)
    #: TensorFlow normalises the last axis, which is the channels, and the
    #: channels are axis one here.
    let axis = if rank == 4 { 1 } else { -1 }
    let x = im.value_input(node.inputs[0], node)
    let _emitted = im.emit(op, node.name, [x], {"axis": Attr.of_int(axis)})
    im.ranks[node.name] = rank
}

fn translate_lrn(im: Importer, node: TfNode) -> void {
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

fn translate_reduce(im: Importer, node: TfNode, op: str) -> void {
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

fn translate_arg_extreme(im: Importer, node: TfNode, op: str) -> void {
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

fn translate_squeeze(im: Importer, node: TfNode) -> void {
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

fn translate_expand_dims(im: Importer, node: TfNode) -> void {
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

fn translate_transpose(im: Importer, node: TfNode) -> void {
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

fn translate_reshape(im: Importer, node: TfNode) -> void {
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

fn translate_concat(im: Importer, node: TfNode) -> void {
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

fn translate_pad(im: Importer, node: TfNode) -> void {
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

fn translate_cast(im: Importer, node: TfNode) -> void {
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

fn translate(im: Importer, node: TfNode) -> void {
    let op = node.op
    if op in PASSTHROUGH {
        if node.inputs.len() == 0 { throw ValueError(f"{node.label()} passes nothing through") }
        translate_unary(im, node, "Identity")
        return
    }
    if op == "Conv2D" {
        translate_conv(im, node, false)
        return
    }
    if op == "DepthwiseConv2dNative" {
        translate_conv(im, node, true)
        return
    }
    if op == "MaxPool" {
        translate_pool(im, node, true)
        return
    }
    if op == "AvgPool" {
        translate_pool(im, node, false)
        return
    }
    if op == "FusedBatchNorm" or op == "FusedBatchNormV2" or op == "FusedBatchNormV3" {
        translate_batch_norm(im, node)
        return
    }
    if op == "MatMul" {
        translate_matmul(im, node)
        return
    }
    if op == "Softmax" {
        translate_softmax(im, node, "Softmax")
        return
    }
    if op == "LogSoftmax" {
        translate_softmax(im, node, "LogSoftmax")
        return
    }
    if op == "LRN" {
        translate_lrn(im, node)
        return
    }
    if op == "Squeeze" {
        translate_squeeze(im, node)
        return
    }
    if op == "ExpandDims" {
        translate_expand_dims(im, node)
        return
    }
    if op == "Transpose" {
        translate_transpose(im, node)
        return
    }
    if op == "Reshape" {
        translate_reshape(im, node)
        return
    }
    if op == "ConcatV2" or op == "Concat" {
        translate_concat(im, node)
        return
    }
    if op == "Pad" or op == "PadV2" or op == "MirrorPad" {
        translate_pad(im, node)
        return
    }
    if op == "Cast" {
        translate_cast(im, node)
        return
    }
    if op == "ArgMax" {
        translate_arg_extreme(im, node, "ArgMax")
        return
    }
    if op == "ArgMin" {
        translate_arg_extreme(im, node, "ArgMin")
        return
    }
    let reduce = REDUCE_OPS.get(op, "")
    if reduce != "" {
        translate_reduce(im, node, reduce)
        return
    }
    if op == "Relu6" {
        #: `Relu6` is the clamp MobileNet is built out of, and ONNX has no
        #: operator for it beyond the general one.
        translate_unary(im, node, "Clip", {"min": Attr.of_float(0.0), "max": Attr.of_float(6.0)})
        return
    }
    if op == "LeakyRelu" {
        translate_unary(
            im,
            node,
            "LeakyRelu",
            {"alpha": Attr.of_float(attr_float(node, "alpha", 0.2))}
        )
        return
    }
    if op == "Square" or op == "Rsqrt" {
        if node.inputs.len() < 1 { throw ValueError(f"{node.label()} needs an input") }
        #: Squaring by multiplying would give the node one name in two input
        #: slots, and the executor frees an intermediate by the slot it was
        #: read in. An exponent costs one initializer and no such question.
        let exponent = if op == "Square" { 2.0 } else { -0.5 }
        let power = im.intern(f"{node.name}$exponent", [1], [exponent])
        let x = im.value_input(node.inputs[0], node)
        let _raised = im.emit("Pow", node.name, [x, power])
        im.ranks[node.name] = im.loose_rank(node.inputs[0])
        return
    }
    if op == "AddN" {
        translate_elementwise(im, node, "Sum")
        return
    }
    let unary = UNARY_OPS.get(op, "")
    if unary != "" {
        translate_unary(im, node, unary)
        return
    }
    let binary = BINARY_OPS.get(op, "")
    if binary != "" {
        translate_elementwise(im, node, binary)
        return
    }
    throw ValueError(f"{node.label()} is an operator this importer does not translate")
}

#: The nodes in an order where every name is written before it is read.
#:
#: A `GraphDef` usually already is in that order, but nothing in the format
#: says it has to be, and this importer needs the rank of a name before the
#: node that reads it. Kahn's, with the readers of each name indexed once so
#: the walk is linear in edges.
fn sorted_nodes(nodes: list[TfNode]) -> list[TfNode] {
    var index: dict[str, int] = {}
    for (slot, node) in nodes.enumerate() {
        if node.name in index { throw ValueError(f"the file has two nodes called {node.name}") }
        index[node.name] = slot
    }
    var waiting = [0 for _slot in 0..nodes.len()]
    var readers: dict[str, list[int]] = {}
    for (slot, node) in nodes.enumerate() {
        for name in node.inputs {
            let producer = index.get(name, -1)
            if producer < 0 {
                throw ValueError(f"{node.label()} reads {name}, which nothing in the file produces")
            }
            waiting[slot] += 1
            if readers.get(name, null) is null { readers[name] = [] }
            readers[name].push(slot)
        }
    }
    var ready: list[int] = []
    for (slot, count) in waiting.enumerate() {
        if count == 0 { ready.push(slot) }
    }
    var order: list[TfNode] = []
    var next = 0
    while next < ready.len() {
        let slot = ready[next]
        next += 1
        order.push(nodes[slot])
        for reader in readers.get(nodes[slot].name, []) {
            waiting[reader] -= 1
            if waiting[reader] == 0 { ready.push(reader) }
        }
    }
    if order.len() != nodes.len() {
        var stuck: list[str] = []
        for (slot, count) in waiting.enumerate() {
            if count > 0 { stuck.push(nodes[slot].name) }
        }
        throw ValueError(f"these nodes are in a cycle and cannot be ordered: {stuck}")
    }
    return order
}

#: Whether a `Const` declares `DT_STRING`, checked without decoding the rest
#: of the tensor so that string is the only reason one gets dropped silently.
#: Anything else that keeps a constant from decoding -- a data type this
#: importer has no tensor for, a declared shape its bytes cannot back -- has
#: to say so, not be folded into the same "carries no value" a dropped string
#: gets.
fn const_is_string(node: TfNode) -> bool {
    let found = attr_of(node, "value")
    if found is null { return false }
    let value = field_by_number(found, ATTR_TENSOR)
    if value is null { return false }
    let dtype_field = field_by_number(parse_nested(value), TENSOR_DTYPE)
    let dtype = if dtype_field is null { DT_FLOAT } else { dtype_field.varint }
    return dtype == DT_STRING
}

fn build_graph(nodes: list[TfNode]) -> Graph {
    let order = sorted_nodes(nodes)
    let im = Importer()
    for node in order {
        if node.op in DISCARDED {
            im.dropped[node.name] = true
            continue
        }
        if node.op != "Const" { continue }
        #: A frozen graph carries string constants for things this importer
        #: never runs -- an assertion's message, a file name. Refusing the
        #: whole file over one is wrong; refusing it only if something reads
        #: it is right.
        if const_is_string(node) {
            im.dropped[node.name] = true
            continue
        }
        let held = attr_tensor(node, "value")
        if held is null { throw ValueError(f"{node.label()} carries no value") }
        let tensor: TfTensor = held
        im.consts[node.name] = tensor
        im.ranks[node.name] = tensor.rank()
    }
    var outputs: list[str] = []
    try {
        for node in order {
            if node.op == "Const" or node.op in DISCARDED { continue }
            if node.op == "Placeholder" or node.op == "PlaceholderV2" {
                let dims = attr_shape(node, "shape")
                im.ranks[node.name] = if dims is null { -1 } else { dims.len() }
                im.graph_inputs.push(node.name)
                continue
            }
            translate(im, node)
        }
        var consumed: dict[str, bool] = {}
        for node in order {
            if node.op == "Const" or node.op in DISCARDED { continue }
            for name in node.inputs { consumed[name] = true }
        }
        #: What nothing else reads is what the graph is for. A frozen file
        #: names no outputs of its own, so this is the only thing to go on,
        #: and it is what every tool that reads one of these does.
        for node in order {
            if node.op in DISCARDED { continue }
            if node.op == "Const" or node.op == "Placeholder" or node.op == "PlaceholderV2" {
                continue
            }
            if not consumed.get(node.name, false) { outputs.push(node.name) }
        }
        if outputs.len() == 0 {
            throw ValueError(
                "every node in this file feeds another one, so it names no output to run it for"
            )
        }
    } catch failure: Error {
        im.discard()
        throw failure
    }
    let graph = Graph(
        im.produced,
        inputs: im.graph_inputs,
        outputs: outputs,
        initializers: im.weights
    )
    try {
        let _order = graph.topological()
    } catch failure: Error {
        im.discard()
        throw failure
    }
    return graph
}

#: What to say when a caller passes the text graph OpenCV takes beside the
#: binary one. Refusing is the honest answer: the configuration exists to
#: replace a subgraph with layers -- `PriorBox`, `DetectionOutput` -- that this
#: executor's operator set does not have, so a parser for it would buy nothing
#: but a later and more confusing failure.
let NO_CONFIG = "this importer reads the binary GraphDef alone; the text graph OpenCV takes beside it exists to substitute detection layers that this executor has no operators for"

#: Read a frozen TensorFlow `GraphDef` from `path`.
#:
#: The graph's inputs and outputs are `NCHW`, whatever the file's own layout
#: was, so a blob from `blob_from_image` feeds one directly. The names are the
#: file's own: a `Placeholder` keeps its name as a graph input, and every node
#: nothing else reads becomes a graph output under its own name.
pub fn read_net_from_tensorflow(path: str, config: str? = null) -> Graph {
    if config is not null { throw ValueError(NO_CONFIG) }
    return read_net_from_tensorflow_bytes(io.read_bytes(path))
}

#: The same, for a file already in memory.
pub fn read_net_from_tensorflow_bytes(data: bytes, config: bytes? = null) -> Graph {
    if config is not null { throw ValueError(NO_CONFIG) }
    var nodes: list[TfNode] = []
    for (slot, entry) in fields_by_number(parse_message(data), GRAPH_NODE).enumerate() {
        nodes.push(read_node(parse_nested(entry), slot))
    }
    if nodes.len() == 0 {
        throw ValueError("this file holds no nodes, so it is not a frozen GraphDef")
    }
    return build_graph(nodes)
}
