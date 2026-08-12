#: Fixed-width integers, rational numbers and complex numbers.
#:
#: ```
#: (I32(2) ** 3).value                  # 8
#: (U8(200, WRAP) + U8(100)).value      # 44
#: (Rational(1, 6) + Rational(1, 10)).to_str()    # "4/15"
#: Complex(0.0, 1.0).sqrt().re          # 0.7071067811865476
#: ```
import std.math

#: Raise `OverflowError` when a result leaves the type's range. The default.
pub const CHECKED: int = 0

#: Discard the bits above the type's width, as machine arithmetic does.
pub const WRAP: int = 1

#: Clamp a result that leaves the type's range to the nearest endpoint.
pub const SATURATE: int = 2

const MASK63: int = 0x7FFFFFFFFFFFFFFF

#: The behaviour shared by every fixed-width integer type.
#:
#: A `Fixed` holds one value and one overflow mode. Binary ops require the same
#: type (or an in-range `int`); the *left* operand's mode decides the result:
#: `CHECKED` raises `OverflowError`, the other two produce a value.
#:
#: ```
#: (I8(120, WRAP) + I8(10)).value       # -126
#: (I8(120, SATURATE) + I8(10)).value   # 127
#: (I8(-128, SATURATE) - I8(1)).value   # -128
#: ```
pub class Fixed {
    prot var raw: int
    prot let bits: int
    prot let signed: bool
    prot let name: str
    prot let min_raw: int
    prot let max_raw: int

    #: One of `CHECKED`, `WRAP` or `SATURATE`.
    pub let mode: int

    prot fn init(self, value: int, bits: int, signed: bool, name: str, mode: int) {
        if mode != CHECKED and mode != WRAP and mode != SATURATE {
            throw ValueError(f"mode must be CHECKED, WRAP or SATURATE, got {mode}")
        }
        self.bits = bits
        self.signed = signed
        self.name = name
        self.mode = mode
        if signed {
            self.min_raw = -(1 << bits - 1)
            self.max_raw = (1 << bits - 1) - 1
        } elif bits == 64 {
            # Every bit pattern is a valid u64; 2**64-1 is the all-ones pattern, which reads back as -1.
            self.min_raw = 0
            self.max_raw = -1
        } else {
            self.min_raw = 0
            self.max_raw = (1 << bits) - 1
        }
        if not self._contains(value) {
            let low = self._format(self.min_raw)
            let high = self._format(self.max_raw)
            throw ValueError(f"{value} is out of range for {name} ({low}..={high})")
        }
        self.raw = value
    }

    #: The value held, as an ordinary int.
    pub get value() -> int {
        return self.raw
    }

    #: The width of the type in bits.
    pub get width() -> int {
        return self.bits
    }

    #: Return the same value with a different overflow mode.
    #:
    #: ```
    #: (I8(120).with_mode(WRAP) + I8(10)).value    # -126
    #: ```
    pub fn with_mode(self, mode: int) -> Fixed {
        return self._make(self.raw, mode)
    }

    #: Return the value as an int, which for `U64` may not fit.
    pub fn to_int(self) -> int {
        return self.raw
    }

    #: Render the value in decimal, unsigned for `U64`.
    pub fn to_str(self) -> str {
        return self._format(self.raw)
    }

    #: Compare against another value of the same type, or an int: -1, 0 or 1.
    pub fn compare(self, other: Fixed | int) -> int {
        return self._compare_raw(self.raw, self._operand(other, "compare"))
    }

    # --- hooks: the ~32 bit types use these as written, U64 replaces them ---
    prot fn _make(self, raw: int, mode: int) -> Fixed {
        throw RuntimeError(f"{self.name} does not implement _make")
    }

    prot fn _contains(self, value: int) -> bool {
        return value >= self.min_raw and value <= self.max_raw
    }

    prot fn _format(self, raw: int) -> str {
        return str(raw)
    }

    prot fn _compare_raw(self, a: int, b: int) -> int {
        if a < b { return -1 }
        if a > b { return 1 }
        return 0
    }

    # Each returns (wrapped result, direction): 0 in range, 1 above the max, -1 below the min.
    prot fn _add_parts(self, a: int, b: int) -> tuple[int, int] {
        # Operands are at most 32 bits wide here, so the sum cannot overflow.
        let total = a + b
        return (self._wrap(total), self._direction(total))
    }

    prot fn _sub_parts(self, a: int, b: int) -> tuple[int, int] {
        let total = a - b
        return (self._wrap(total), self._direction(total))
    }

    prot fn _mul_parts(self, a: int, b: int) -> tuple[int, int] {
        if a == 0 or b == 0 { return (0, 0) }
        let negative = (a < 0) != (b < 0)
        # U32 products reach 2**64; test the magnitude by division before multiplying.
        let limit = if negative { -self.min_raw } else { self.max_raw }
        if abs(a) <= limit // abs(b) { return (a * b, 0) }
        let direction = if negative { -1 } else { 1 }
        return (self._wrap(a *% b), direction)
    }

    prot fn _div_parts(self, a: int, b: int) -> tuple[int, int] {
        # The only overflowing division is min / -1, whose true value is max + 1.
        if a == self.min_raw and b == -1 { return (self._wrap(-a), 1) }
        var quotient = a // b
        # `//` floors and fixed-width division truncates towards zero.
        if quotient < 0 and quotient * b != a { quotient += 1 }
        return (quotient, 0)
    }

    fn _wrap(self, value: int) -> int {
        let masked = value & (1 << self.bits) - 1
        if self.signed and masked > self.max_raw { return masked - (1 << self.bits) }
        return masked
    }

    fn _direction(self, value: int) -> int {
        if value > self.max_raw { return 1 }
        if value < self.min_raw { return -1 }
        return 0
    }

    fn _operand(self, other: any, operation: str) -> int {
        if isinstance(other, Fixed) {
            if other.name != self.name {
                throw TypeError(f"cannot {operation} {self.name} and {other.name}")
            }
            return other.raw
        }
        if isinstance(other, int) {
            if not self._contains(other) {
                throw ValueError(f"{other} is out of range for {self.name}")
            }
            return other
        }
        throw TypeError(f"cannot {operation} {self.name} and {type_of(other)}")
    }

    fn _resolve(self, result: int, direction: int, operation: str) -> Fixed {
        if direction == 0 or self.mode == WRAP { return self._make(result, self.mode) }
        if self.mode == SATURATE {
            let clamped = if direction > 0 { self.max_raw } else { self.min_raw }
            return self._make(clamped, self.mode)
        }
        let edge = if direction > 0 { "above the maximum" } else { "below the minimum" }
        let limit = if direction > 0 { self.max_raw } else { self.min_raw }
        throw OverflowError(f"{self.name} {operation} landed {edge} {self._format(limit)}")
    }

    fn __add__(self, other: Fixed | int) -> Fixed {
        let (result, direction) = self._add_parts(self.raw, self._operand(other, "add"))
        return self._resolve(result, direction, "add")
    }

    fn __sub__(self, other: Fixed | int) -> Fixed {
        let (result, direction) = self._sub_parts(self.raw, self._operand(other, "subtract"))
        return self._resolve(result, direction, "subtract")
    }

    fn __mul__(self, other: Fixed | int) -> Fixed {
        let (result, direction) = self._mul_parts(self.raw, self._operand(other, "multiply"))
        return self._resolve(result, direction, "multiply")
    }

    #: Truncating division, so `I8(-7) / I8(2)` is -3 rather than -4.
    fn __div__(self, other: Fixed | int) -> Fixed {
        let divisor = self._operand(other, "divide")
        if divisor == 0 { throw DivisionByZeroError(f"{self.name} division by zero") }
        let (result, direction) = self._div_parts(self.raw, divisor)
        return self._resolve(result, direction, "divide")
    }

    fn __pow__(self, exponent: int) -> Fixed {
        if exponent < 0 {
            throw ValueError(f"exponent must be non-negative, got {exponent}")
        }
        var result = self._make(1, self.mode)
        var square = self
        var remaining = exponent
        while remaining > 0 {
            if remaining % 2 == 1 { result = result * square }
            remaining = remaining // 2
            if remaining > 0 { square = square * square }
        }
        return result
    }

    fn __neg__(self) -> Fixed {
        let (result, direction) = self._sub_parts(0, self.raw)
        return self._resolve(result, direction, "negate")
    }

    fn __eq__(self, other: any) -> bool {
        if isinstance(other, Fixed) {
            return other.name == self.name and other.raw == self.raw
        }
        if isinstance(other, int) { return self.raw == other and self._contains(other) }
        return false
    }

    fn __lt__(self, other: Fixed | int) -> bool {
        return self.compare(other) < 0
    }

    fn __le__(self, other: Fixed | int) -> bool {
        return self.compare(other) <= 0
    }

    fn __hash__(self) -> int {
        return hash(self.raw)
    }

    fn __str__(self) -> str {
        return self._format(self.raw)
    }

    fn __repr__(self) -> str {
        return f"{self.name}({self._format(self.raw)})"
    }
}

#: A signed 8-bit integer, -128 through 127.
pub class I8 extends Fixed {
    #: Create an `I8`, raising `ValueError` when `value` does not fit.
    pub fn init(self, value: int, mode: int = CHECKED) {
        super(value, 8, true, "I8", mode)
    }

    prot fn _make(self, raw: int, mode: int) -> Fixed {
        return I8(raw, mode)
    }
}

#: A signed 16-bit integer, -32768 through 32767.
pub class I16 extends Fixed {
    #: Create an `I16`, raising `ValueError` when `value` does not fit.
    pub fn init(self, value: int, mode: int = CHECKED) {
        super(value, 16, true, "I16", mode)
    }

    prot fn _make(self, raw: int, mode: int) -> Fixed {
        return I16(raw, mode)
    }
}

#: A signed 32-bit integer, -2147483648 through 2147483647.
pub class I32 extends Fixed {
    #: Create an `I32`, raising `ValueError` when `value` does not fit.
    pub fn init(self, value: int, mode: int = CHECKED) {
        super(value, 32, true, "I32", mode)
    }

    prot fn _make(self, raw: int, mode: int) -> Fixed {
        return I32(raw, mode)
    }
}

#: An unsigned 8-bit integer, 0 through 255.
pub class U8 extends Fixed {
    #: Create a `U8`, raising `ValueError` when `value` does not fit.
    pub fn init(self, value: int, mode: int = CHECKED) {
        super(value, 8, false, "U8", mode)
    }

    prot fn _make(self, raw: int, mode: int) -> Fixed {
        return U8(raw, mode)
    }
}

#: An unsigned 16-bit integer, 0 through 65535.
pub class U16 extends Fixed {
    #: Create a `U16`, raising `ValueError` when `value` does not fit.
    pub fn init(self, value: int, mode: int = CHECKED) {
        super(value, 16, false, "U16", mode)
    }

    prot fn _make(self, raw: int, mode: int) -> Fixed {
        return U16(raw, mode)
    }
}

#: An unsigned 32-bit integer, 0 through 4294967295.
pub class U32 extends Fixed {
    #: Create a `U32`, raising `ValueError` when `value` does not fit.
    pub fn init(self, value: int, mode: int = CHECKED) {
        super(value, 32, false, "U32", mode)
    }

    prot fn _make(self, raw: int, mode: int) -> Fixed {
        return U32(raw, mode)
    }
}

#: An unsigned 64-bit integer, 0 through 18446744073709551615.
#:
#: The upper half of the range doesn't fit in an int, so `U64` stores the raw
#: two's-complement bits: `value` reads back negative from 2**63 up, while
#: `to_str` and every operator treat it as unsigned. Build with `from_str` or
#: `from_bits`; `to_int` raises `OverflowError` rather than return the negative pattern.
#:
#: ```
#: U64.from_bits(-1).to_str()                        # "18446744073709551615"
#: (U64.from_bits(-1) / U64(3)).to_str()             # "6148914691236517205"
#: (U64.from_bits(-1, WRAP) + U64(1)).value          # 0
#: ```
pub class U64 extends Fixed {
    #: Create a `U64` from a non-negative int.
    pub fn init(self, value: int, mode: int = CHECKED) {
        super(value, 64, false, "U64", mode)
    }

    #: Reinterpret a 64-bit pattern as unsigned, so -1 becomes 2**64 - 1.
    pub static fn from_bits(pattern: int, mode: int = CHECKED) -> U64 {
        let result = U64(0, mode)
        result.raw = pattern
        return result
    }

    #: Parse a decimal string, raising `ValueError` above 2**64 - 1.
    #:
    #: ```
    #: U64.from_str("12345").value    # 12345
    #: ```
    pub static fn from_str(text: str, mode: int = CHECKED) -> U64 {
        return U64.from_bits(_u64_parse(text), mode)
    }

    #: Return the value as an int, raising `OverflowError` from 2**63 up.
    pub fn to_int(self) -> int {
        if self.raw < 0 {
            throw OverflowError(f"{_u64_str(self.raw)} does not fit in a signed int")
        }
        return self.raw
    }

    prot fn _make(self, raw: int, mode: int) -> Fixed {
        return U64.from_bits(raw, mode)
    }

    prot fn _contains(self, value: int) -> bool {
        # Only the constructor (which takes an int) consults this; from_bits covers the rest.
        return value >= 0
    }

    prot fn _format(self, raw: int) -> str {
        return _u64_str(raw)
    }

    prot fn _compare_raw(self, a: int, b: int) -> int {
        return _u64_compare(a, b)
    }

    prot fn _add_parts(self, a: int, b: int) -> tuple[int, int] {
        let total = a +% b
        # A sum that wrapped lands below the operand it started from.
        let direction = if _u64_compare(total, a) < 0 { 1 } else { 0 }
        return (total, direction)
    }

    prot fn _sub_parts(self, a: int, b: int) -> tuple[int, int] {
        let direction = if _u64_compare(a, b) < 0 { -1 } else { 0 }
        return (a -% b, direction)
    }

    prot fn _mul_parts(self, a: int, b: int) -> tuple[int, int] {
        if a == 0 or b == 0 { return (0, 0) }
        # The product fits exactly when the other factor is at most (2**64-1)/a.
        let (limit, _remainder) = _u64_divmod(-1, a)
        let direction = if _u64_compare(b, limit) > 0 { 1 } else { 0 }
        return (a *% b, direction)
    }

    prot fn _div_parts(self, a: int, b: int) -> tuple[int, int] {
        let (quotient, _remainder) = _u64_divmod(a, b)
        return (quotient, 0)
    }
}

#: An exact fraction, always in lowest terms with a positive denominator.
#:
#: ```
#: Rational(6, -8).to_str()             # "-3/4"
#: (Rational(1, 3) * Rational(3, 5)).to_str()     # "1/5"
#: Rational.from_float(0.1).to_str()    # "1/10"
#: ```
pub class Rational {
    #: The numerator, carrying the sign of the value.
    pub let numerator: int

    #: The denominator, always positive.
    pub let denominator: int

    #: Create `numerator / denominator`, reduced by their greatest common divisor.
    #:
    #: Raises `DivisionByZeroError` when `denominator` is zero.
    pub fn init(self, numerator: int, denominator: int = 1) {
        if denominator == 0 {
            throw DivisionByZeroError(f"denominator must be non-zero, got {numerator}/0")
        }
        var top = numerator
        var bottom = denominator
        if bottom < 0 {
            top = -top
            bottom = -bottom
        }
        let common = math.gcd(top, bottom)
        self.numerator = top // common
        self.denominator = bottom // common
    }

    #: Return the closest fraction to `value` with a denominator at most
    #: `max_denominator`.
    #:
    #: Expands `value` as a continued fraction, keeping the best convergent or
    #: semiconvergent that fits — exact integer arithmetic throughout, except
    #: that below `abs(value) < 2**-10` low mantissa bits are dropped, which can
    #: shift the last digit when `max_denominator` exceeds about 10**9.
    #:
    #: Raises `ValueError` for NaN or an infinity, and `OverflowError` for a
    #: value too large for an int numerator.
    #:
    #: ```
    #: Rational.from_float(0.25).to_str()               # "1/4"
    #: Rational.from_float(3.141592653589793).to_str()  # "3126535/995207"
    #: Rational.from_float(3.141592653589793, 1000).to_str()    # "355/113"
    #: Rational.from_float(-0.75, 10).to_str()          # "-3/4"
    #: ```
    pub static fn from_float(value: float, max_denominator: int = 1000000) -> Rational {
        if max_denominator < 1 {
            throw ValueError(f"max_denominator must be positive, got {max_denominator}")
        }
        if math.is_nan(value) or math.is_inf(value) {
            throw ValueError(f"cannot convert {value} to a Rational")
        }
        let negative = value < 0.0
        let magnitude = math.fabs(value)
        if magnitude == 0.0 { return Rational(0, 1) }
        let (mantissa, exponent) = math.frexp(magnitude)
        # mantissa is in [0.5, 1), so mantissa * 2**53 is exact, and magnitude is that integer over 2**(53-exponent).
        var top = int(mantissa * 9007199254740992.0)
        if exponent >= 53 {
            if exponent - 53 > 10 {
                throw OverflowError(f"{value} is too large for a Rational")
            }
            let scaled = top << exponent - 53
            let whole = if negative { -scaled } else { scaled }
            return Rational(whole, 1)
        }
        var bits = 53 - exponent
        if bits > 62 {
            # Below 2**-10 the exact denominator no longer fits in an int; drop bits that can't affect a denominator this size.
            top = top >> bits - 62
            bits = 62
        }
        if top == 0 { return Rational(0, 1) }
        let bottom = 1 << bits
        let common = math.gcd(top, bottom)
        let exact = Rational(top // common, bottom // common)
        if exact.denominator <= max_denominator {
            return if negative { -exact } else { exact }
        }
        let limited = _limit_denominator(exact.numerator, exact.denominator, max_denominator)
        return if negative { -limited } else { limited }
    }

    #: Return the value as the nearest float.
    pub fn to_float(self) -> float {
        return float(self.numerator) / float(self.denominator)
    }

    #: Return `1 / self`, raising `DivisionByZeroError` for zero.
    pub fn reciprocal(self) -> Rational {
        if self.numerator == 0 { throw DivisionByZeroError("reciprocal of 0/1") }
        return Rational(self.denominator, self.numerator)
    }

    #: Return the magnitude.
    pub fn abs(self) -> Rational {
        if self.numerator >= 0 { return self }
        return Rational(-self.numerator, self.denominator)
    }

    #: Return the largest int not greater than the value.
    #:
    #: ```
    #: Rational(-7, 2).floor()   # -4
    #: ```
    pub fn floor(self) -> int {
        return self.numerator // self.denominator
    }

    #: Return `self ** exponent`; a negative exponent inverts first.
    #:
    #: ```
    #: Rational(2, 3).pow(-2).to_str()   # "9/4"
    #: ```
    pub fn pow(self, exponent: int) -> Rational {
        if exponent < 0 { return self.reciprocal().pow(-exponent) }
        return Rational(self.numerator ** exponent, self.denominator ** exponent)
    }

    #: Compare against another `Rational` or an int: -1, 0 or 1.
    #:
    #: Cross-multiplication would overflow for large terms, so the two products
    #: are compared without ever being formed.
    pub fn compare(self, other: Rational | int) -> int {
        let that = _rational(other, "compare")
        let negative = self.numerator < 0
        if negative != (that.numerator < 0) {
            return if negative { -1 } else { 1 }
        }
        let order = _compare_products(
            abs(self.numerator),
            that.denominator,
            abs(that.numerator),
            self.denominator
        )
        return if negative { -order } else { order }
    }

    #: Render as "numerator/denominator", or just the numerator when it is whole.
    pub fn to_str(self) -> str {
        if self.denominator == 1 { return str(self.numerator) }
        return f"{self.numerator}/{self.denominator}"
    }

    fn __add__(self, other: Rational | int) -> Rational {
        let that = _rational(other, "add")
        # Knuth 4.5.1: reducing against gcd(denominators) first keeps the
        # intermediate products as small as the result allows.
        let common = math.gcd(self.denominator, that.denominator)
        if common == 1 {
            let top = self.numerator * that.denominator + that.numerator * self.denominator
            return Rational(top, self.denominator * that.denominator)
        }
        let left = self.denominator // common
        let top = self.numerator * (that.denominator // common) + that.numerator * left
        let second = math.gcd(top, common)
        return Rational(top // second, left * (that.denominator // second))
    }

    fn __sub__(self, other: Rational | int) -> Rational {
        return self + -_rational(other, "subtract")
    }

    fn __mul__(self, other: Rational | int) -> Rational {
        let that = _rational(other, "multiply")
        # Cancel crosswise before multiplying, for the same reason.
        let left = math.gcd(self.numerator, that.denominator)
        let right = math.gcd(that.numerator, self.denominator)
        return Rational(
            self.numerator // left * (that.numerator // right),
            self.denominator // right * (that.denominator // left)
        )
    }

    fn __div__(self, other: Rational | int) -> Rational {
        let that = _rational(other, "divide")
        if that.numerator == 0 { throw DivisionByZeroError("Rational division by zero") }
        return self * that.reciprocal()
    }

    fn __pow__(self, exponent: int) -> Rational {
        return self.pow(exponent)
    }

    fn __neg__(self) -> Rational {
        return Rational(-self.numerator, self.denominator)
    }

    fn __eq__(self, other: any) -> bool {
        if isinstance(other, Rational) or isinstance(other, int) {
            return self.compare(other) == 0
        }
        return false
    }

    fn __lt__(self, other: Rational | int) -> bool {
        return self.compare(other) < 0
    }

    fn __le__(self, other: Rational | int) -> bool {
        return self.compare(other) <= 0
    }

    fn __hash__(self) -> int {
        # Rational(3, 1) == 3, so whole values have to hash like their int.
        if self.denominator == 1 { return hash(self.numerator) }
        return hash((self.numerator, self.denominator))
    }

    fn __str__(self) -> str {
        return self.to_str()
    }

    fn __repr__(self) -> str {
        return f"Rational({self.numerator}, {self.denominator})"
    }
}

#: A complex number in rectangular form.
#:
#: ```
#: (Complex(1.0, 2.0) * Complex(3.0, -1.0)).to_str()   # "5.0 + 5.0i"
#: Complex(3.0, 4.0).abs()                             # 5.0
#: Complex(-1.0, 0.0).sqrt().im                        # 1.0
#: ```
pub class Complex {
    #: The real part.
    pub let re: float

    #: The imaginary part.
    pub let im: float

    #: Create `re + im * i`.
    pub fn init(self, re: float, im: float = 0.0) {
        self.re = re
        self.im = im
    }

    #: Create `radius * (cos(angle) + i * sin(angle))`.
    #:
    #: ```
    #: Complex.from_polar(2.0, 0.0).re    # 2.0
    #: ```
    pub static fn from_polar(radius: float, angle: float) -> Complex {
        return Complex(radius * math.cos(angle), radius * math.sin(angle))
    }

    #: Return the modulus, computed without squaring either part.
    pub fn abs(self) -> float {
        return math.hypot(self.re, self.im)
    }

    #: Return the squared modulus, which needs no square root.
    pub fn norm(self) -> float {
        return self.re * self.re + self.im * self.im
    }

    #: Return the argument in radians, in (-pi, pi].
    pub fn arg(self) -> float {
        return math.atan2(self.im, self.re)
    }

    #: Return `(modulus, argument)`.
    pub fn polar(self) -> tuple[float, float] {
        return (self.abs(), self.arg())
    }

    #: Return the complex conjugate.
    pub fn conj(self) -> Complex {
        return Complex(self.re, -self.im)
    }

    #: Return `e ** self`.
    #:
    #: ```
    #: Complex(0.0, 0.0).exp().re    # 1.0
    #: ```
    pub fn exp(self) -> Complex {
        let magnitude = math.exp(self.re)
        return Complex(magnitude * math.cos(self.im), magnitude * math.sin(self.im))
    }

    #: Return the principal square root, the one with a non-negative real part.
    #:
    #: The two half-angle identities each cancel catastrophically on one side of
    #: the imaginary axis; only the one adding `abs(re)` to the modulus is evaluated, the other recovered by division.
    #:
    #: ```
    #: Complex(-4.0, 0.0).sqrt().to_str()   # "0.0 + 2.0i"
    #: Complex(4.0, 0.0).sqrt().re          # 2.0
    #: ```
    pub fn sqrt(self) -> Complex {
        let modulus = self.abs()
        if modulus == 0.0 { return Complex(0.0, 0.0) }
        let half = math.sqrt((math.fabs(self.re) + modulus) * 0.5)
        if self.re >= 0.0 { return Complex(half, self.im / (2.0 * half)) }
        return Complex(math.fabs(self.im) / (2.0 * half), math.copysign(half, self.im))
    }

    #: Render as "re + imi", with the sign of the imaginary part spelled out.
    pub fn to_str(self) -> str {
        if self.im < 0.0 or self.im == 0.0 and math.copysign(1.0, self.im) < 0.0 {
            return f"{self.re} - {math.fabs(self.im)}i"
        }
        return f"{self.re} + {self.im}i"
    }

    fn __add__(self, other: Complex | float | int) -> Complex {
        let that = _complex(other, "add")
        return Complex(self.re + that.re, self.im + that.im)
    }

    fn __sub__(self, other: Complex | float | int) -> Complex {
        let that = _complex(other, "subtract")
        return Complex(self.re - that.re, self.im - that.im)
    }

    fn __mul__(self, other: Complex | float | int) -> Complex {
        let that = _complex(other, "multiply")
        return Complex(self.re * that.re - self.im * that.im, self.re * that.im + self.im * that.re)
    }

    #: Smith's algorithm: dividing by the larger part first keeps the
    #: intermediate squares from overflowing for wide-ranging inputs.
    fn __div__(self, other: Complex | float | int) -> Complex {
        let that = _complex(other, "divide")
        if that.re == 0.0 and that.im == 0.0 {
            throw DivisionByZeroError("Complex division by zero")
        }
        if math.fabs(that.re) >= math.fabs(that.im) {
            let ratio = that.im / that.re
            let scale = that.re + that.im * ratio
            return Complex((self.re + self.im * ratio) / scale, (self.im - self.re * ratio) / scale)
        }
        let ratio = that.re / that.im
        let scale = that.re * ratio + that.im
        return Complex((self.re * ratio + self.im) / scale, (self.im * ratio - self.re) / scale)
    }

    fn __neg__(self) -> Complex {
        return Complex(-self.re, -self.im)
    }

    fn __eq__(self, other: any) -> bool {
        if isinstance(other, Complex) {
            return self.re == other.re and self.im == other.im
        }
        if isinstance(other, float) { return self.im == 0.0 and self.re == other }
        if isinstance(other, int) { return self.im == 0.0 and self.re == float(other) }
        return false
    }

    fn __hash__(self) -> int {
        return hash((self.re, self.im))
    }

    fn __str__(self) -> str {
        return self.to_str()
    }

    fn __repr__(self) -> str {
        return f"Complex({self.re}, {self.im})"
    }
}

fn _rational(value: any, operation: str) -> Rational {
    if isinstance(value, Rational) { return value }
    if isinstance(value, int) { return Rational(value, 1) }
    throw TypeError(f"cannot {operation} a Rational and {type_of(value)}")
}

fn _complex(value: any, operation: str) -> Complex {
    if isinstance(value, Complex) { return value }
    if isinstance(value, float) { return Complex(value, 0.0) }
    if isinstance(value, int) { return Complex(float(value), 0.0) }
    throw TypeError(f"cannot {operation} a Complex and {type_of(value)}")
}

# Best approximation to `top / bottom` with denominator <= `limit`; assumes an
# already-reduced fraction whose denominator exceeds `limit`.
fn _limit_denominator(top: int, bottom: int, limit: int) -> Rational {
    var previous_top = 0
    var previous_bottom = 1
    var current_top = 1
    var current_bottom = 0
    var remainder_top = top
    var remainder_bottom = bottom
    loop {
        let term = remainder_top // remainder_bottom
        if current_bottom > 0 and term > (limit - previous_bottom) // current_bottom { break }
        let next_bottom = previous_bottom + term * current_bottom
        if next_bottom > limit { break }
        if current_top > 0 and term > (math.INT_MAX - previous_top) // current_top {
            throw OverflowError(f"no fraction near {top}/{bottom} fits in an int")
        }
        let next_top = previous_top + term * current_top
        previous_top = current_top
        previous_bottom = current_bottom
        current_top = next_top
        current_bottom = next_bottom
        let rest = remainder_top - term * remainder_bottom
        remainder_top = remainder_bottom
        remainder_bottom = rest
    }
    # `current` and the best semiconvergent below it are the only candidates;
    # the semiconvergent wins when t*current_bottom < previous_bottom + 2*steps*current_bottom.
    let steps = (limit - previous_bottom) // current_bottom
    let threshold = previous_bottom + 2 * steps * current_bottom
    if _compare_products(remainder_top, current_bottom, remainder_bottom, threshold) < 0 {
        return Rational(
            previous_top + steps * current_top,
            previous_bottom + steps * current_bottom
        )
    }
    return Rational(current_top, current_bottom)
}

# Compares a*b against c*d without forming a product, dividing the pair down
# like Euclid's algorithm (O(log n) steps, never overflows an int).
fn _compare_products(a: int, b: int, c: int, d: int) -> int {
    var left_high = a
    var left_low = b
    var right_high = c
    var right_low = d
    var sense = 1
    loop {
        if left_high == 0 or left_low == 0 {
            if right_high == 0 or right_low == 0 { return 0 }
            return -sense
        }
        if right_high == 0 or right_low == 0 { return sense }
        let left_term = left_high // right_high
        let right_term = right_low // left_low
        if left_term != right_term {
            return if left_term > right_term { sense } else { -sense }
        }
        # a*b - c*d == (a % c)*b - c*(d % b), so compare those and swap sides.
        let next_high = left_high % right_high
        let next_low = right_low % left_low
        left_high = right_high
        right_high = next_high
        let previous_left_low = left_low
        left_low = next_low
        right_low = previous_left_low
        sense = -sense
    }
}

fn _u64_compare(a: int, b: int) -> int {
    if (a < 0) != (b < 0) {
        # Exactly one has its top bit set, and that one is the larger unsigned.
        return if a < 0 { 1 } else { -1 }
    }
    if a < b { return -1 }
    if a > b { return 1 }
    return 0
}

fn _u64_divmod(a: int, b: int) -> tuple[int, int] {
    if b == 0 { throw DivisionByZeroError("U64 division by zero") }
    if b < 0 {
        # The divisor is at least 2**63, so the quotient is 0 or 1.
        if _u64_compare(a, b) < 0 { return (0, a) }
        return (1, a -% b)
    }
    if a >= 0 { return (a // b, a % b) }
    # Halving first keeps the dividend positive; `& MASK63` makes the shift logical regardless of whether `>>` sign-extends.
    let half = a >> 1 & MASK63
    var quotient = half // b *% 2
    var remainder = a -% quotient *% b
    while _u64_compare(remainder, b) >= 0 {
        remainder = remainder -% b
        quotient = quotient +% 1
    }
    return (quotient, remainder)
}

fn _u64_str(value: int) -> str {
    if value == 0 { return "0" }
    if value > 0 { return str(value) }
    var digits: list[str] = []
    var rest = value
    while rest != 0 {
        let (quotient, remainder) = _u64_divmod(rest, 10)
        digits.push(str(remainder))
        rest = quotient
    }
    return "".join(digits.reversed())
}

fn _u64_parse(text: str) -> int {
    if text.len() == 0 { throw ValueError("cannot parse an empty string as a U64") }
    let (limit, _remainder) = _u64_divmod(-1, 10)
    var result = 0
    for character in text {
        let digit = ord(character) - 48
        if digit < 0 or digit > 9 {
            throw ValueError(f"invalid digit '{character}' in U64 literal {text}")
        }
        if _u64_compare(result, limit) > 0 {
            throw ValueError(f"{text} is above the U64 maximum 18446744073709551615")
        }
        let scaled = result *% 10
        let total = scaled +% digit
        if _u64_compare(total, scaled) < 0 {
            throw ValueError(f"{text} is above the U64 maximum 18446744073709551615")
        }
        result = total
    }
    return result
}
