#: The weights: what data type a `TensorProto` holds, and its values off the wire.
from ..protobuf import Field, WIRE_I32, WIRE_I64, WIRE_LEN, WIRE_VARINT
from ..protobuf import as_bool, as_double, as_float, field_by_number, fields_by_number
from ..protobuf import packed_doubles, packed_floats, packed_varints, parse_nested
from std.math import INF, NAN, ldexp

#: `TensorProto`. Note `int_val` is 7 and `double_val` is 6, which is the way
#: round that is easy to write down backwards.
pub let TENSOR_DTYPE = 1
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
pub let DT_FLOAT = 1
let DT_DOUBLE = 2
let DT_INT32 = 3
let DT_UINT8 = 4
let DT_INT16 = 5
let DT_INT8 = 6
pub let DT_STRING = 7
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
pub let DTYPE_NAMES: dict[int, str] = {
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
pub let ONNX_DTYPES: dict[int, int] = {
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

#: Every value of a repeated varint field, whether the writer packed it or
#: not.
#:
#: TensorFlow's protos are proto3, so a repeated scalar is packed by default
#: and arrives as one length-delimited run -- but a `.pb` written by an older
#: exporter, or merged from two encodings, can carry the same field number
#: several times and unpacked. Reading only the first, or assuming the wire
#: type, loses values from real files.
pub fn repeated_varints(fields: list[Field], number: int, where: str) -> list[int] {
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
pub class TfTensor {
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
pub fn read_shape_proto(fields: list[Field], where: str) -> list[int]? {
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
pub fn read_tensor_proto(fields: list[Field], total_bytes: int, where: str) -> TfTensor {
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
