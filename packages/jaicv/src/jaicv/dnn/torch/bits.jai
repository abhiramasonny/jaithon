#: Storage classes, element widths, and the little-endian and IEEE-754 readers
#: every other module here decodes bytes through.
import std.math

#: What one element of each torch storage class is, in this module's names.
#:
#: This doubles as the whitelist of storage classes: a `torch.` global is
#: allowed exactly when it is a key here.
pub let STORAGE_DTYPES: dict[str, str] = {
    "FloatStorage": "float32",
    "DoubleStorage": "float64",
    "HalfStorage": "float16",
    "BFloat16Storage": "bfloat16",
    "LongStorage": "int64",
    "IntStorage": "int32",
    "ShortStorage": "int16",
    "CharStorage": "int8",
    "ByteStorage": "uint8",
    "BoolStorage": "bool",
}

#: Bytes per element.
let DTYPE_SIZES: dict[str, int] = {
    "float32": 4,
    "float64": 8,
    "float16": 2,
    "bfloat16": 2,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "uint8": 1,
    "bool": 1,
}

#: How wide one element of `dtype` is, in bytes.
pub fn dtype_size(dtype: str) -> int {
    if not DTYPE_SIZES.contains(dtype) {
        throw ValueError(f"'{dtype}' is not a dtype this reader knows")
    }
    return DTYPE_SIZES[dtype]
}

pub fn le_u16(data: bytes, at: int) -> int {
    return data[at] | data[at + 1] << 8
}

pub fn le_u32(data: bytes, at: int) -> int {
    return data[at] | data[at + 1] << 8 | data[at + 2] << 16 | data[at + 3] << 24
}

pub fn be_u32(data: bytes, at: int) -> int {
    return data[at] << 24 | data[at + 1] << 16 | data[at + 2] << 8 | data[at + 3]
}

#: A little-endian integer `width` bytes wide, read as two's complement when
#: `signed`.
#:
#: The top byte is folded in by multiplication rather than by a shift, because
#: an unsigned `0xFF` in the eighth byte is past `INT_MAX` and shifting it
#: there raises an overflow before the sign can be applied.
pub fn le_int(data: bytes, at: int, width: int, signed: bool) -> int {
    if width == 0 { return 0 }
    var value = 0
    var scale = 1
    for index in 0..width - 1 {
        value += data[at + index] * scale
        scale *= 256
    }
    var top = data[at + width - 1]
    if signed and top >= 128 { top -= 256 }
    if not signed and width >= 8 {
        throw ValueError("an unsigned 64-bit field does not fit in an int")
    }
    return value + top * scale
}

pub fn le_u64(data: bytes, at: int) -> int {
    if data[at + 7] >= 128 {
        throw ValueError("a ZIP field claims a size past 2^63, which cannot be a real file")
    }
    return le_int(data, at, 8, true)
}

#: An IEEE-754 binary32 out of its 32 bits.
#:
#: Assembled with `ldexp` rather than by any arithmetic on the value, so a
#: subnormal, a zero of either sign, and the exact mantissa of a large value
#: all survive: every step is exact in a double.
pub fn f32_from_bits(bits: int) -> float {
    let exponent = bits >> 23 & 0xFF
    let mantissa = bits & 0x7FFFFF
    var value = 0.0
    if exponent == 0 {
        value = math.ldexp(float(mantissa), -149)
    } elif exponent == 0xFF {
        value = mantissa == 0 ? math.INF : math.NAN
    } else {
        value = math.ldexp(float(mantissa + 0x800000), exponent - 150)
    }
    return bits >> 31 & 1 == 1 ? -value : value
}

#: An IEEE-754 binary64 out of its two halves.
#:
#: The 53-bit significand is built as `high * 2^32 + low`, which is at most
#: `2^53 - 1` and so still exact in a double before `ldexp` scales it.
pub fn f64_from_halves(high: int, low: int) -> float {
    let exponent = high >> 20 & 0x7FF
    let fraction = high & 0xFFFFF
    var value = 0.0
    if exponent == 0 {
        value = math.ldexp(float(fraction) * 4294967296.0 + float(low), -1074)
    } elif exponent == 0x7FF {
        value = fraction == 0 and low == 0 ? math.INF : math.NAN
    } else {
        value = math.ldexp(
            (float(fraction) + 1048576.0) * 4294967296.0 + float(low),
            exponent - 1075
        )
    }
    return high >> 31 & 1 == 1 ? -value : value
}

#: An IEEE-754 binary16 out of its 16 bits.
pub fn f16_from_bits(bits: int) -> float {
    let exponent = bits >> 10 & 0x1F
    let mantissa = bits & 0x3FF
    var value = 0.0
    if exponent == 0 {
        value = math.ldexp(float(mantissa), -24)
    } elif exponent == 0x1F {
        value = mantissa == 0 ? math.INF : math.NAN
    } else {
        value = math.ldexp(float(mantissa + 1024), exponent - 25)
    }
    return bits >> 15 & 1 == 1 ? -value : value
}
