#: The dtype tags, the arithmetic and cast codes a kernel is told in, and what
#: a pair of dtypes promotes to.

#: Elements per threadgroup in every kernel here that folds through
#: threadgroup memory. 256 is eight simdgroups, which is what the scan's
#: per-warp table is sized for.
pub const GROUP = 256

#: Largest whole number float32 represents exactly. An integer dtype is a
#: promise about rounding, not about width, so a count past this silently
#: stops being a count.
pub const MAX_EXACT_INT = 16777216

#: dtype: 0.0 or 1.0, nothing else. A comparison produces one of these.
pub const BOOL = 0
#: dtype: whole numbers, exact to `MAX_EXACT_INT`.
pub const INT32 = 1
#: dtype: whole numbers. Storage is float32, so this is exact to
#: `MAX_EXACT_INT` too — an alias that keeps source compatible, not a wider
#: element.
pub const INT64 = 2
#: dtype: 32-bit float, the storage everything actually is.
pub const FLOAT32 = 3
#: dtype: 64-bit float. Storage is 32-bit, so this is an alias that keeps
#: source compatible rather than a wider element.
pub const FLOAT64 = 4
#: dtype: interleaved real and imaginary float32, two slots per element.
pub const COMPLEX64 = 5

#: Arithmetic selector shared by the four kernels the operators own.
pub const OP_ADD = 0
#: Arithmetic selector: subtraction.
pub const OP_SUB = 1
#: Arithmetic selector: multiplication.
pub const OP_MUL = 2
#: Arithmetic selector: true division.
pub const OP_DIV = 3

#: Conversion mode for the copy kernels: take the value as it stands.
pub const CAST_NONE = 0
#: Conversion mode: round to nearest even, then clamp — what an integer dtype
#: means.
pub const CAST_ROUND = 1
#: Conversion mode: anything non-zero becomes 1.0.
pub const CAST_BOOL = 2

#: The name numpy prints for a dtype.
pub fn dtype_name(dtype: int) -> str {
    let names = ["bool", "int32", "int64", "float32", "float64", "complex64"]
    if dtype < 0 or dtype >= names.len() { return f"dtype{dtype}" }
    return names[dtype]
}

#: Float slots one element of `dtype` occupies: 2 for complex, 1 for the rest.
pub fn dtype_lanes(dtype: int) -> int { return if dtype == COMPLEX64 { 2 } else { 1 } }

#: Whether `dtype` stores whole numbers, and therefore rounds and saturates on
#: the way in.
pub fn dtype_is_integer(dtype: int) -> bool { return dtype <= INT64 }

#: Whether `dtype` carries an imaginary part.
pub fn dtype_is_complex(dtype: int) -> bool { return dtype == COMPLEX64 }

#: Whether `dtype` is the two-valued one.
pub fn dtype_is_bool(dtype: int) -> bool { return dtype == BOOL }

#: Lowest value `dtype` holds, for a saturating cast.
pub fn dtype_min(dtype: int) -> float {
    if dtype == BOOL { return 0.0 }
    return -2147483648.0
}

#: Highest value `dtype` holds, for a saturating cast.
#:
#: 2147483520 and not 2147483647: a float32 near 2^31 steps in 128s, so the
#: true int32 ceiling rounds *up* to 2^31 and a clamp against it lets through
#: a value the type cannot hold. This is the largest float32 below 2^31.
#:
#: `INT64` saturates there too rather than at 2^63. Past `MAX_EXACT_INT`
#: neither bound is representable one integer at a time, and a bound that lies
#: is worse than a bound that is merely narrow.
pub fn dtype_max(dtype: int) -> float {
    if dtype == BOOL { return 1.0 }
    return 2147483520.0
}

#: The dtype an operation on `a` and `b` produces.
#:
#: A plain lattice — `BOOL < INT32 < INT64 < FLOAT32 < FLOAT64 < COMPLEX64` —
#: because the constants are ordered to make it one. numpy's value-based
#: promotion is not reproduced: with float32 storage underneath, the only
#: distinctions this can honour are integer-versus-float and real-versus-
#: complex, and pretending otherwise would promise widths that do not exist.
pub fn promote(a: int, b: int) -> int { return max(a, b) }
