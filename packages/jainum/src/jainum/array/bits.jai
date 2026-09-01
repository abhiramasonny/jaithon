#: A float through an integer and back, for the kernels that take one as a
#: `constant uint&`.

#: Reinterpret a float's bits as an integer, so a kernel can take it as a
#: `constant uint&` and read it back with `as_type<float>`.
#:
#: A float scalar that carries no layout — a fill value, a divisor — travels
#: this way. Shapes and strides cannot: they are signed and there are many of
#: them, which is what `layout_buffer` is for.
pub fn f32_bits(value: float) -> int {
    if value != value { return 0x7FC00000 }
    var magnitude = value
    var sign = 0
    if magnitude < 0.0 {
        sign = 1
        magnitude = -magnitude
    }
    #: A negative zero does not compare less than zero, so `copysign` is what
    #: reads its sign. Worth the call: a fill value travels through here, and
    #: -0.0 is a value a caller can mean.
    if magnitude == 0.0 {
        return if __prim__.f64_copysign(1.0, value) < 0.0 { 1 << 31 } else { 0 }
    }
    if magnitude > 3.4028234663852886e38 { return sign << 31 | 0x7F800000 }
    if magnitude <= 1.401298464324817e-45 { return sign << 31 }
    let split = __prim__.f64_frexp(magnitude)
    let biased = split[1] + 126
    if biased <= 0 {
        let fraction = int(__prim__.f64_ldexp(magnitude, 149))
        return sign << 31 | fraction & 0x7FFFFF
    }
    if biased >= 255 { return sign << 31 | 0x7F800000 }
    var fraction = int((split[0] * 2.0 - 1.0) * 8388608.0 + 0.5)
    var exponent = biased
    #: Rounding the mantissa can carry out of it -- 2147483647.0 rounds to
    #: 2^31 -- and masking the carry away instead of adding it to the exponent
    #: turns the largest int32 into 2^30. That is not academic: it is the
    #: saturation bound every integer cast is dispatched with.
    if fraction >= 0x800000 {
        fraction = 0
        exponent += 1
        if exponent >= 255 { return sign << 31 | 0x7F800000 }
    }
    return sign << 31 | exponent << 23 | fraction
}

#: Read a float back out of the bits `f32_bits` made.
pub fn f32_from_bits(bits: int) -> float {
    let sign = if bits & 0x80000000 != 0 { -1.0 } else { 1.0 }
    let biased = bits >> 23 & 0xFF
    let fraction = bits & 0x7FFFFF
    if biased == 0 { return sign * __prim__.f64_ldexp(float(fraction), -149) }
    if biased == 255 {
        if fraction != 0 { return 0.0 / 0.0 }
        return sign * 1.0e308 * 10.0
    }
    return sign * __prim__.f64_ldexp(float(fraction | 0x800000), biased - 150)
}
