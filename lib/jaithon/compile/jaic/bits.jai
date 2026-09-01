#: The bit-level conversions the container is built out of: its CRC-32, the
#: FNV-1a source hash, UTF-8 in both directions, and the IEEE-754 binary64
#: bit pattern a float is stored as.
#:
#: Pure functions over ints, floats and strings. Nothing here knows what a
#: `.jaic` file looks like, which is why `Writer` and `Reader` are built on
#: top of this file rather than beside it.
#:
#: `float_bits` and `bits_to_float` are arithmetic rather than a
#: reinterpretation because this language cannot reinterpret a float as an
#: integer, and `_is_negative` exists because `atan2` is the only primitive
#: that tells -0.0 from 0.0.
#:
#: ```
#: crc32([0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39])   # 3421780262
#: float_bits(1.0)                                                 # 4607182418800017408
#: bits_to_float(float_bits(-2.5))                                 # -2.5
#: ```

#: Bit 63 of a 64-bit integer, on its own. Public because `Reader.u64` sets it
#: from the top byte it read, the same way `_assemble` sets it from a sign.
pub let _SIGN_BIT: int = -9223372036854775807 - 1
let _CRC_TABLE: list[int] = _build_crc_table()

fn _build_crc_table() -> list[int] {
    var table: list[int] = []
    for index in range(256) {
        var value = index
        for _bit in range(8) {
            value = if value & 1 == 1 { 0xEDB88320 ^ value >> 1 } else { value >> 1 }
        }
        table.push(value)
    }
    return table
}

#: CRC-32 (IEEE 802.3, reflected, polynomial `0xEDB88320`) of `data` — the
#: checksum held in the container's last four bytes, computed over everything
#: before them.
#:
#: ```
#: crc32([])          # 0
#: crc32([0x61])      # 3904355907
#: ```
pub fn crc32(data: list[int], length: int = -1) -> int {
    let stop = if length < 0 or length > data.len() { data.len() } else { length }
    var crc = 0xFFFFFFFF
    for index in range(stop) {
        crc = _CRC_TABLE[(crc ^ data[index]) & 0xFF] ^ crc >> 8
    }
    return crc ^ 0xFFFFFFFF
}

#: FNV-1a 64 of `text`, which is the container's `srcHash`.
#:
#: Signed 64-bit with the same bit pattern the C side writes, since the
#: multiply wraps.
#:
#: ```
#: source_hash("") == source_hash("")     # true
#: source_hash("a") != source_hash("b")   # true
#: ```
pub fn source_hash(text: str) -> int {
    var hash = -3750763034362895579  # 14695981039346656037 as i64
    for byte in utf8_bytes(text) {
        hash = (hash ^ byte) *% 1099511628211
    }
    return hash
}

#: The UTF-8 bytes of `text`, as integers in 0..255.
#:
#: ```
#: utf8_bytes("hi")      # [104, 105]
#: ```
pub fn utf8_bytes(text: str) -> list[int] {
    let encoded = __prim__.str_encode(text)
    var out: list[int] = []
    for index in range(encoded.len()) { out.push(encoded[index]) }
    return out
}

#: Decode UTF-8 `data` back into a string.
#:
#: Raises `ValueError` on invalid UTF-8: a pool that can't decode is corrupt,
#: not lossy.
pub fn utf8_text(data: list[int]) -> str {
    return __prim__.str_decode(__prim__.bytes_new(data))
}

#: The IEEE-754 binary64 bit pattern of `value`, as a signed 64-bit integer.
#:
#: ```
#: float_bits(0.0)       # 0
#: float_bits(2.0)       # 4611686018427387904
#: ```
pub fn float_bits(value: float) -> int {
    if value != value { return 0x7FF8000000000000 }  # a quiet NaN

    var magnitude = value
    var sign = 0
    if _is_negative(value) {
        sign = 1
        magnitude = -value
    }
    if magnitude > 1.7976931348623157e308 {
        return _assemble(sign, 2047, 0)  # infinity
    }
    if magnitude == 0.0 { return _assemble(sign, 0, 0) }

    let split = __prim__.f64_frexp(magnitude)  # (m in [0.5,1), exp)
    let exponent = split[1] + 1022
    if exponent <= 0 {
        # Subnormal: the value is fraction * 2**-1074 exactly.
        return _assemble(sign, 0, int(__prim__.f64_ldexp(magnitude, 1074)))
    }
    return _assemble(sign, exponent, int((split[0] * 2.0 - 1.0) * 4503599627370496.0))
}

#: The binary64 value whose bit pattern is `bits`.
#:
#: ```
#: bits_to_float(4607182418800017408)      # 1.0
#: bits_to_float(0)                        # 0.0
#: ```
pub fn bits_to_float(bits: int) -> float {
    let exponent = bits >> 52 & 2047
    let fraction = bits & 4503599627370495
    var value = 0.0
    if exponent == 0 {
        value = __prim__.f64_ldexp(float(fraction), -1074)
    } elif exponent == 2047 {
        value = if fraction == 0 { 1.0e308 * 10.0 } else { _nan() }
    } else {
        value = __prim__.f64_ldexp(float(fraction) + 4503599627370496.0, exponent - 1075)
    }
    if bits < 0 { return -value }
    return value
}

fn _nan() -> float {
    let infinity = 1.0e308 * 10.0
    return infinity - infinity
}

# atan2 is the only primitive that distinguishes -0.0 from 0.0: atan2(-0, -1)
# is -pi where atan2(+0, -1) is +pi.
fn _is_negative(value: float) -> bool {
    if value < 0.0 { return true }
    if value > 0.0 { return false }
    return __prim__.f64_atan2(value, -1.0) < 0.0
}

fn _assemble(sign: int, exponent: int, fraction: int) -> int {
    let low = (exponent & 2047) << 52 | fraction & 4503599627370495
    if sign == 1 { return low | _SIGN_BIT }
    return low
}
