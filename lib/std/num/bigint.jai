#: Arbitrary-precision signed integers.
#:
#: A `BigInt` is a sign and a little-endian list of base-2**31 limbs. Values are
#: immutable: every operation returns a new `BigInt`, so a value may be shared
#: without copying. All the ordinary operators work, and `/` is floor division
#: (there is no fractional big number, so `/` and `//` would mean the same
#: thing; see `divmod` for the exact convention).
#:
#: ```
#: (BigInt(2).pow(64) - BigInt(1)).to_str()   # "18446744073709551615"
#: BigInt.from_str("0xdead_beef").to_int()    # 3735928559
#: BigInt(-17).divmod(BigInt(5))[1].to_int()  # 3
#: ```
import std.math

# Limbs are little-endian base-2**31 digits, each stored in a plain `int`.
#
# 2**32 is the traditional limb base, but it cannot be used here: Jaithon's
# `int` is *signed* 64-bit and raises `OverflowError` rather than wrapping, and
# the schoolbook inner step `out[i+j] + a[i]*b[j] + carry` reaches
# (2**32-1)**2 + 2**33 - 1, which is just above 2**64 and so needs an unsigned
# accumulator that this language does not have. With 2**31 the same step peaks
# below 2**62.01, which leaves two whole bits of headroom inside a signed 64-bit
# int. Knuth's algorithm D needs exactly the same property for its trial
# quotient `(u[j+n]*BASE + u[j+n-1]) // v[n-1]`, whose numerator is below 2**62,
# and for the `qhat * v[n-2]` correction test, which is below 2**62 as well.
const SHIFT: int = 31
const BASE: int = 2147483648
const MASK: int = 2147483647

# Below this many limbs the recursion overhead of Karatsuba (three recursive
# products plus six linear passes) costs more than the n**2 inner loop it
# replaces. 32 limbs is ~1000 bits, which is where the crossover sits for an
# interpreted inner loop; a C implementation would set it lower.
const KARATSUBA_THRESHOLD: int = 32

const DIGITS: str = "0123456789abcdefghijklmnopqrstuvwxyz"

#: A signed integer with no width limit.
#:
#: An `int` raises `OverflowError` once a result passes 64 bits; a `BigInt`
#: grows another limb instead, so factorials, Fibonacci numbers, checksums and
#: modular exponentiation stay exact for as long as memory holds out. Values
#: are immutable: every operator returns a new `BigInt` and none of them
#: change an operand.
#:
#: A plain `int` may appear on the right of any operator or method and is
#: converted for you, but there is no conversion on the left, so `total * 3`
#: compiles and `3 * total` does not. Comparisons, `==` and `hash` are all
#: defined, so a `BigInt` sorts and works as a dict key. `/` floors and `%`
#: takes the sign of the divisor, the same convention `int` uses.
#:
#: Raises `OverflowError` from `to_int` when the value is too wide for an
#: `int`; `to_float` returns an infinity rather than raising.
#:
#: ```
#: var total = BigInt(1)
#: for n in 1..=25 { total = total * n }
#: total.to_str()                      # "15511210043330985984000000"
#: total.bit_length()                  # 84
#: (BigInt(-7) / BigInt(2)).to_int()   # -4
#: (BigInt(-7) % BigInt(2)).to_int()   # 1
#: ```
pub class BigInt {
    var signum: int
    var limbs: list[int]

    #: Create a `BigInt` from an ordinary int, zero by default.
    #:
    #: ```
    #: BigInt(255).to_str(16)   # "ff"
    #: BigInt().is_zero()       # true
    #: ```
    pub fn init(self, value: int = 0) {
        if value == math.INT_MIN {
            # abs(INT_MIN) is not representable, so spell 2**63 out directly.
            self.signum = -1
            self.limbs = [0, 0, 2]
            return
        }
        if value == 0 {
            self.signum = 0
            self.limbs = []
            return
        }
        self.signum = if value < 0 { -1 } else { 1 }
        self.limbs = []
        var magnitude = abs(value)
        while magnitude > 0 {
            self.limbs.push(magnitude & MASK)
            magnitude = magnitude >> SHIFT
        }
    }

    #: Parse `text` as an integer in `base`, or in the base its prefix names.
    #:
    #: `base` 0 (the default) infers the base from a `0x`, `0o` or `0b` prefix
    #: and falls back to decimal. An explicit `base` in 2..=36 is used as given,
    #: and a prefix that contradicts it is a `ParseError`. A leading `+` or `-`
    #: is accepted, `_` may be used as a digit separator, and surrounding
    #: whitespace is not allowed.
    #:
    #: ```
    #: BigInt.from_str("1_000_000").to_int()    # 1000000
    #: BigInt.from_str("-0b1011").to_int()      # -11
    #: BigInt.from_str("zz", 36).to_int()       # 1295
    #: ```
    pub static fn from_str(text: str, base: int = 0) -> BigInt {
        if base != 0 and (base < 2 or base > 36) {
            throw ValueError(f"base must be 0 or in 2..=36, got {base}")
        }
        if text.len() == 0 { throw ParseError("cannot parse an empty string as a BigInt") }
        var position = 0
        var negative = false
        if text[0] == "-" {
            negative = true
            position = 1
        } elif text[0] == "+" {
            position = 1
        }
        var actual_base = if base == 0 { 10 } else { base }
        if text.len() >= position + 2 and text[position] == "0" {
            let prefixed = _prefix_base(text[position + 1])
            if prefixed != 0 {
                if base != 0 and base != prefixed {
                    throw ParseError(f"prefix '0{text[position + 1]}' contradicts base {base}")
                }
                actual_base = prefixed
                position += 2
            }
        }
        let (chunk_digits, chunk_base) = _chunk_for(actual_base)
        var limbs: list[int] = []
        var accumulator = 0
        var pending = 0
        var digits_seen = 0
        while position < text.len() {
            let character = text[position]
            position += 1
            if character == "_" { continue }
            let value = _digit_value(character)
            if value < 0 or value >= actual_base {
                throw ParseError(f"invalid digit '{character}' for base {actual_base}")
            }
            accumulator = accumulator * actual_base + value
            pending += 1
            digits_seen += 1
            if pending == chunk_digits {
                limbs = _add_small(_mul_small(limbs, chunk_base), accumulator)
                accumulator = 0
                pending = 0
            }
        }
        if digits_seen == 0 { throw ParseError(f"no digits in {text}") }
        if pending > 0 {
            limbs = _add_small(_mul_small(limbs, actual_base ** pending), accumulator)
        }
        let signum = if negative { -1 } else { 1 }
        return BigInt._make(signum, limbs)
    }

    #: -1 when negative, 0 for zero, 1 when positive.
    pub fn sign(self) -> int {
        return self.signum
    }

    #: True when `self` is zero.
    pub fn is_zero(self) -> bool {
        return self.signum == 0
    }

    #: Return `-self`.
    pub fn negate(self) -> BigInt {
        return BigInt._make(-self.signum, [limb for limb in self.limbs])
    }

    #: Return the magnitude of `self`.
    pub fn abs(self) -> BigInt {
        if self.signum >= 0 { return self }
        return self.negate()
    }

    #: Number of bits in the magnitude, so 0 for zero and 8 for 255 and -255.
    #:
    #: ```
    #: BigInt(255).bit_length()          # 8
    #: BigInt(2).pow(100).bit_length()   # 101
    #: ```
    pub fn bit_length(self) -> int {
        if self.signum == 0 { return 0 }
        let top = self.limbs.len() - 1
        return top * SHIFT + math.bit_length(self.limbs[top])
    }

    #: Return `self * 2**count`.
    pub fn shl(self, count: int) -> BigInt {
        if count < 0 { throw ValueError(f"shift count must be non-negative, got {count}") }
        if self.signum == 0 or count == 0 { return self }
        let whole = count // SHIFT
        let bits = count % SHIFT
        var out: list[int] = []
        for _step in 0..whole { out.push(0) }
        if bits == 0 {
            for limb in self.limbs { out.push(limb) }
        } else {
            var carry = 0
            for limb in self.limbs {
                let shifted = limb << bits | carry
                out.push(shifted & MASK)
                carry = shifted >> SHIFT
            }
            if carry > 0 { out.push(carry) }
        }
        return BigInt._make(self.signum, out)
    }

    #: Return `self` divided by `2**count`, rounded towards negative infinity.
    #:
    #: This matches `divmod`, so `BigInt(-1).shr(1)` is -1 rather than 0.
    pub fn shr(self, count: int) -> BigInt {
        if count < 0 { throw ValueError(f"shift count must be non-negative, got {count}") }
        if self.signum == 0 or count == 0 { return self }
        let whole = count // SHIFT
        let bits = count % SHIFT
        let length = self.limbs.len()
        if whole >= length {
            return if self.signum < 0 { BigInt(-1) } else { BigInt() }
        }
        var discarded = false
        for index in 0..whole {
            if self.limbs[index] != 0 { discarded = true }
        }
        if bits > 0 and self.limbs[whole] & (1 << bits) - 1 != 0 { discarded = true }
        var out: list[int] = []
        var index = whole
        while index < length {
            var value = self.limbs[index] >> bits
            if bits > 0 and index + 1 < length {
                value = value | self.limbs[index + 1] << SHIFT - bits & MASK
            }
            out.push(value)
            index += 1
        }
        let shifted = BigInt._make(self.signum, out)
        if self.signum < 0 and discarded { return shifted - BigInt(1) }
        return shifted
    }

    #: Return `(quotient, remainder)` with the quotient rounded towards negative
    #: infinity, so the remainder always has the sign of the divisor.
    #:
    #: Uses Knuth's algorithm D (TAOCP 4.3.1) on the magnitudes and then adjusts
    #: the truncated pair to the floored one. Raises `DivisionByZeroError` when
    #: `other` is zero.
    #:
    #: ```
    #: BigInt(17).divmod(5)[0].to_int()    # 3
    #: BigInt(-17).divmod(5)[0].to_int()   # -4
    #: BigInt(-17).divmod(5)[1].to_int()   # 3
    #: ```
    pub fn divmod(self, other: BigInt | int) -> tuple[BigInt, BigInt] {
        let that = _coerce(other, "divide")
        if that.signum == 0 { throw DivisionByZeroError("BigInt division by zero") }
        if self.signum == 0 { return (BigInt(), BigInt()) }
        let (quotient_limbs, remainder_limbs) = _divmod_limbs(self.limbs, that.limbs)
        var quotient = BigInt._make(self.signum * that.signum, quotient_limbs)
        var remainder = BigInt._make(self.signum, remainder_limbs)
        if remainder.signum != 0 and self.signum != that.signum {
            quotient = quotient - BigInt(1)
            remainder = remainder + that
        }
        return (quotient, remainder)
    }

    #: Return `self ** exponent` by binary exponentiation.
    #:
    #: Raises `ValueError` for a negative exponent, which is not an integer.
    #:
    #: ```
    #: BigInt(3).pow(5).to_int()     # 243
    #: BigInt(-2).pow(3).to_int()    # -8
    #: ```
    pub fn pow(self, exponent: int) -> BigInt {
        if exponent < 0 {
            throw ValueError(f"exponent must be non-negative, got {exponent}")
        }
        var result = BigInt(1)
        var square = self
        var remaining = exponent
        while remaining > 0 {
            if remaining % 2 == 1 { result = result * square }
            remaining = remaining // 2
            if remaining > 0 { square = square * square }
        }
        return result
    }

    #: Return `self ** exponent % modulus`, reducing after every step.
    #:
    #: The intermediate values never exceed the square of the modulus, so this
    #: stays cheap for exponents that `pow` could never finish. `modulus` must be
    #: positive and `exponent` non-negative.
    #:
    #: ```
    #: BigInt(4).pow_mod(13, 497).to_int()                      # 445
    #: BigInt(2).pow_mod(BigInt(1000), 1000000007).to_int()     # 688423210
    #: ```
    pub fn pow_mod(self, exponent: BigInt | int, modulus: BigInt | int) -> BigInt {
        let power = _coerce(exponent, "exponentiate")
        let divisor = _coerce(modulus, "reduce")
        if divisor.signum <= 0 {
            throw ValueError(f"modulus must be positive, got {divisor.to_str()}")
        }
        if power.signum < 0 {
            throw ValueError(f"exponent must be non-negative, got {power.to_str()}")
        }
        var result = BigInt(1) % divisor
        let residue = self % divisor
        var bit = power.bit_length() - 1
        while bit >= 0 {
            result = result * result % divisor
            if power._bit(bit) { result = result * residue % divisor }
            bit -= 1
        }
        return result
    }

    #: Render `self` in `base`, using lowercase letters for digits above nine.
    #:
    #: ```
    #: BigInt(255).to_str()      # "255"
    #: BigInt(255).to_str(2)     # "11111111"
    #: BigInt(-255).to_str(16)   # "-ff"
    #: ```
    pub fn to_str(self, base: int = 10) -> str {
        if base < 2 or base > 36 { throw ValueError(f"base must be in 2..=36, got {base}") }
        if self.signum == 0 { return "0" }
        # Peel off as many digits per division as fit in one limb, so a value of
        # n limbs costs n**2 / chunk_digits limb divisions rather than n**2.
        let (chunk_digits, chunk_base) = _chunk_for(base)
        var parts: list[str] = []
        var rest = [limb for limb in self.limbs]
        while rest.len() > 0 {
            let (quotient, remainder) = _divmod_small(rest, chunk_base)
            rest = quotient
            var text = _small_to_str(remainder, base)
            if rest.len() > 0 {
                while text.len() < chunk_digits { text = "0" + text }
            }
            parts.push(text)
        }
        let body = "".join(parts.reversed())
        if self.signum < 0 { return "-" + body }
        return body
    }

    #: Return `self` as an int, raising `OverflowError` when it does not fit.
    #:
    #: ```
    #: BigInt.from_str("123456789012").to_int()   # 123456789012
    #: ```
    pub fn to_int(self) -> int {
        if self.signum == 0 { return 0 }
        if self.signum < 0 and self._is_two_to_the_63() { return math.INT_MIN }
        var result = 0
        var index = self.limbs.len() - 1
        while index >= 0 {
            let limb = self.limbs[index]
            if result > (math.INT_MAX - limb) // BASE {
                throw OverflowError(f"{self.to_str()} does not fit in an int")
            }
            result = result * BASE + limb
            index -= 1
        }
        return result * self.signum
    }

    #: Return the nearest float, or an infinity when the value is too large.
    #:
    #: The low bits below the 53 a float carries are rounded away.
    pub fn to_float(self) -> float {
        if self.signum == 0 { return 0.0 }
        let length = self.limbs.len()
        # Three limbs are 93 bits, comfortably more than a float's 53, so the
        # result is off by at most one rounding of the discarded tail.
        var mantissa = 0.0
        var taken = 0
        var index = length - 1
        while index >= 0 and taken < 3 {
            mantissa = mantissa * 2147483648.0 + self.limbs[index]
            taken += 1
            index -= 1
        }
        let scaled = math.ldexp(mantissa, (length - taken) * SHIFT)
        return if self.signum < 0 { -scaled } else { scaled }
    }

    #: Compare against another `BigInt` or int: -1, 0 or 1.
    pub fn compare(self, other: BigInt | int) -> int {
        let that = _coerce(other, "compare")
        if self.signum != that.signum {
            return if self.signum < that.signum { -1 } else { 1 }
        }
        if self.signum == 0 { return 0 }
        return _cmp_limbs(self.limbs, that.limbs) * self.signum
    }

    # -2**63 is the one int that abs() cannot express, so to_int() spots its
    # magnitude, 2 * 2**62, as a limb pattern instead of computing it.
    fn _is_two_to_the_63(self) -> bool {
        if self.limbs.len() != 3 { return false }
        return self.limbs[0] == 0 and self.limbs[1] == 0 and self.limbs[2] == 2
    }

    # True when bit `index` of the magnitude is set.
    fn _bit(self, index: int) -> bool {
        let limb = index // SHIFT
        if limb >= self.limbs.len() { return false }
        return self.limbs[limb] >> index % SHIFT & 1 == 1
    }

    static fn _make(signum: int, limbs: list[int]) -> BigInt {
        _trim(limbs)
        let result = BigInt()
        result.signum = if limbs.len() == 0 { 0 } else { signum }
        result.limbs = limbs
        return result
    }

    fn __add__(self, other: BigInt | int) -> BigInt {
        let that = _coerce(other, "add")
        if self.signum == 0 { return that }
        if that.signum == 0 { return self }
        if self.signum == that.signum {
            return BigInt._make(self.signum, _add_limbs(self.limbs, that.limbs))
        }
        let order = _cmp_limbs(self.limbs, that.limbs)
        if order == 0 { return BigInt() }
        if order > 0 {
            return BigInt._make(self.signum, _sub_limbs(self.limbs, that.limbs))
        }
        return BigInt._make(that.signum, _sub_limbs(that.limbs, self.limbs))
    }

    fn __sub__(self, other: BigInt | int) -> BigInt {
        return self + _coerce(other, "subtract").negate()
    }

    fn __mul__(self, other: BigInt | int) -> BigInt {
        let that = _coerce(other, "multiply")
        if self.signum == 0 or that.signum == 0 { return BigInt() }
        return BigInt._make(self.signum * that.signum, _mul_limbs(self.limbs, that.limbs))
    }

    fn __div__(self, other: BigInt | int) -> BigInt {
        return self.divmod(other)[0]
    }

    fn __mod__(self, other: BigInt | int) -> BigInt {
        return self.divmod(other)[1]
    }

    fn __pow__(self, exponent: int) -> BigInt {
        return self.pow(exponent)
    }

    fn __neg__(self) -> BigInt {
        return self.negate()
    }

    fn __eq__(self, other: any) -> bool {
        if isinstance(other, BigInt) or isinstance(other, int) {
            return self.compare(other) == 0
        }
        return false
    }

    fn __lt__(self, other: BigInt | int) -> bool {
        return self.compare(other) < 0
    }

    fn __le__(self, other: BigInt | int) -> bool {
        return self.compare(other) <= 0
    }

    fn __hash__(self) -> int {
        # `BigInt(n) == n` holds, so every value that fits in an int has to hash
        # like one. Two limbs are 62 bits, and -2**63 is the one wider case.
        if self.limbs.len() <= 2 { return hash(self.to_int()) }
        if self.signum < 0 and self._is_two_to_the_63() { return hash(math.INT_MIN) }
        return hash(self.to_str(36))
    }

    fn __str__(self) -> str {
        return self.to_str()
    }

    fn __repr__(self) -> str {
        return f"BigInt({self.to_str()})"
    }
}

#: Return the factorial of `count` as an exact `BigInt`.
#:
#: `math.factorial` overflows an int at 21!; this is exact for any `count` you
#: are willing to wait for. 100! is 158 digits long and ends in 24 zeros, one
#: for every factor of five below 100 plus the extra from 25, 50, 75 and 100.
#:
#: ```
#: factorial(10).to_int()               # 3628800
#: factorial(100).to_str().len()        # 158
#: factorial(100).to_str()[0:12]        # "933262154439"
#: factorial(100) == factorial(99) * 100    # true
#: factorial(100) % BigInt(10).pow(24) == BigInt(0)    # true
#: ```
pub fn factorial(count: int) -> BigInt {
    if count < 0 { throw ValueError(f"factorial is undefined for {count}") }
    var result = BigInt(1)
    for value in 2..=count { result = result * BigInt(value) }
    return result
}

fn _coerce(value: any, operation: str) -> BigInt {
    if isinstance(value, BigInt) { return value }
    if isinstance(value, int) { return BigInt(value) }
    throw TypeError(f"cannot {operation} a BigInt and {type_of(value)}")
}

fn _prefix_base(marker: str) -> int {
    if marker == "x" or marker == "X" { return 16 }
    if marker == "o" or marker == "O" { return 8 }
    if marker == "b" or marker == "B" { return 2 }
    return 0
}

fn _digit_value(character: str) -> int {
    let code = ord(character)
    if code >= 48 and code <= 57 { return code - 48 }
    if code >= 97 and code <= 122 { return code - 87 }
    if code >= 65 and code <= 90 { return code - 55 }
    return -1
}

# Largest power of `base` below BASE, as (digit count, power).
fn _chunk_for(base: int) -> tuple[int, int] {
    var count = 0
    var power = 1
    while power * base < BASE {
        power = power * base
        count += 1
    }
    return (count, power)
}

fn _small_to_str(value: int, base: int) -> str {
    if value == 0 { return "0" }
    var digits: list[str] = []
    var rest = value
    while rest > 0 {
        digits.push(DIGITS[rest % base])
        rest = rest // base
    }
    return "".join(digits.reversed())
}

fn _trim(limbs: list[int]) -> void {
    while limbs.len() > 0 and limbs[limbs.len() - 1] == 0 { limbs.pop() }
}

fn _cmp_limbs(a: list[int], b: list[int]) -> int {
    if a.len() != b.len() {
        return if a.len() < b.len() { -1 } else { 1 }
    }
    var index = a.len() - 1
    while index >= 0 {
        if a[index] != b[index] {
            return if a[index] < b[index] { -1 } else { 1 }
        }
        index -= 1
    }
    return 0
}

fn _add_limbs(a: list[int], b: list[int]) -> list[int] {
    var out: list[int] = []
    var carry = 0
    let length = max(a.len(), b.len())
    for index in 0..length {
        var total = carry
        if index < a.len() { total += a[index] }
        if index < b.len() { total += b[index] }
        out.push(total & MASK)
        carry = total >> SHIFT
    }
    if carry > 0 { out.push(carry) }
    return out
}

# Requires magnitude(a) >= magnitude(b).
fn _sub_limbs(a: list[int], b: list[int]) -> list[int] {
    var out: list[int] = []
    var borrow = 0
    for index in 0..a.len() {
        var difference = a[index] - borrow
        if index < b.len() { difference -= b[index] }
        if difference < 0 {
            difference += BASE
            borrow = 1
        } else {
            borrow = 0
        }
        out.push(difference)
    }
    _trim(out)
    return out
}

fn _shift_limbs(a: list[int], places: int) -> list[int] {
    if a.len() == 0 { return [] }
    var out: list[int] = []
    for _step in 0..places { out.push(0) }
    for limb in a { out.push(limb) }
    return out
}

fn _slice_trim(a: list[int], start: int, stop: int) -> list[int] {
    if start >= a.len() { return [] }
    var out = a[start:stop]
    _trim(out)
    return out
}

fn _mul_small(a: list[int], multiplier: int) -> list[int] {
    if a.len() == 0 or multiplier == 0 { return [] }
    var out: list[int] = []
    var carry = 0
    for limb in a {
        let accumulator = limb * multiplier + carry
        out.push(accumulator & MASK)
        carry = accumulator >> SHIFT
    }
    while carry > 0 {
        out.push(carry & MASK)
        carry = carry >> SHIFT
    }
    return out
}

fn _add_small(a: list[int], addend: int) -> list[int] {
    var out = [limb for limb in a]
    var carry = addend
    var index = 0
    while carry > 0 {
        if index == out.len() { out.push(0) }
        let total = out[index] + carry
        out[index] = total & MASK
        carry = total >> SHIFT
        index += 1
    }
    return out
}

fn _divmod_small(a: list[int], divisor: int) -> tuple[list[int], int] {
    var quotient: list[int] = []
    for _step in 0..a.len() { quotient.push(0) }
    var remainder = 0
    var index = a.len() - 1
    while index >= 0 {
        # remainder < divisor <= MASK, so this stays below 2**62.
        let accumulator = remainder * BASE + a[index]
        quotient[index] = accumulator // divisor
        remainder = accumulator % divisor
        index -= 1
    }
    _trim(quotient)
    return (quotient, remainder)
}

fn _mul_limbs(a: list[int], b: list[int]) -> list[int] {
    if a.len() < KARATSUBA_THRESHOLD or b.len() < KARATSUBA_THRESHOLD {
        return _mul_school(a, b)
    }
    let half = max(a.len(), b.len()) // 2
    let a_low = _slice_trim(a, 0, half)
    let a_high = _slice_trim(a, half, a.len())
    let b_low = _slice_trim(b, 0, half)
    let b_high = _slice_trim(b, half, b.len())
    let low = _mul_limbs(a_low, b_low)
    let high = _mul_limbs(a_high, b_high)
    # (a_low + a_high)(b_low + b_high) - low - high is never negative, so the
    # middle term needs no sign tracking.
    let middle = _mul_limbs(_add_limbs(a_low, a_high), _add_limbs(b_low, b_high))
    let cross = _sub_limbs(_sub_limbs(middle, low), high)
    return _add_limbs(_add_limbs(low, _shift_limbs(cross, half)), _shift_limbs(high, 2 * half))
}

fn _mul_school(a: list[int], b: list[int]) -> list[int] {
    if a.len() == 0 or b.len() == 0 { return [] }
    var out: list[int] = []
    for _step in 0..a.len() + b.len() { out.push(0) }
    for i in 0..a.len() {
        let left = a[i]
        if left == 0 { continue }
        var carry = 0
        for j in 0..b.len() {
            # Peaks at (2**31-1)**2 + 2*(2**31-1) < 2**62: see the note on SHIFT.
            let accumulator = out[i + j] + left * b[j] + carry
            out[i + j] = accumulator & MASK
            carry = accumulator >> SHIFT
        }
        var index = i + b.len()
        while carry > 0 {
            let accumulator = out[index] + carry
            out[index] = accumulator & MASK
            carry = accumulator >> SHIFT
            index += 1
        }
    }
    _trim(out)
    return out
}

# Knuth, TAOCP volume 2, algorithm 4.3.1D. Both arguments are trimmed
# magnitudes and the divisor is non-zero; the result is truncated, not floored.
fn _divmod_limbs(u: list[int], v: list[int]) -> tuple[list[int], list[int]] {
    let n = v.len()
    if n == 1 {
        let (quotient, remainder) = _divmod_small(u, v[0])
        var remainder_limbs: list[int] = []
        if remainder != 0 { remainder_limbs.push(remainder) }
        return (quotient, remainder_limbs)
    }
    if _cmp_limbs(u, v) < 0 { return ([], [limb for limb in u]) }
    # D1: scale so the leading divisor limb is at least BASE/2, which is what
    # bounds the trial quotient below to at most two too large.
    let scale = BASE // (v[n - 1] + 1)
    var un = if scale > 1 { _mul_small(u, scale) } else { [limb for limb in u] }
    let vn = if scale > 1 { _mul_small(v, scale) } else { [limb for limb in v] }
    if un.len() == u.len() { un.push(0) }
    let m = un.len() - 1 - n
    var quotient: list[int] = []
    for _step in 0..m + 1 { quotient.push(0) }
    var j = m
    while j >= 0 {
        # D3: estimate the next quotient limb from the top two dividend limbs.
        let numerator = un[j + n] * BASE + un[j + n - 1]
        var qhat = numerator // vn[n - 1]
        var rhat = numerator % vn[n - 1]
        var correcting = true
        while correcting {
            if qhat >= BASE or qhat * vn[n - 2] > rhat * BASE + un[j + n - 2] {
                qhat -= 1
                rhat += vn[n - 1]
                correcting = rhat < BASE
            } else {
                correcting = false
            }
        }
        # D4: subtract qhat * divisor from the current window.
        var borrow = 0
        var carry = 0
        for i in 0..n {
            let product = qhat * vn[i] + carry
            carry = product >> SHIFT
            var difference = un[i + j] - (product & MASK) - borrow
            if difference < 0 {
                difference += BASE
                borrow = 1
            } else {
                borrow = 0
            }
            un[i + j] = difference
        }
        let top_difference = un[j + n] - carry - borrow
        if top_difference < 0 {
            # D6: qhat was one too large (this happens for about 2 in 2**31
            # random divisors), so add the divisor back and fix the estimate.
            un[j + n] = top_difference + BASE
            qhat -= 1
            var addend_carry = 0
            for i in 0..n {
                let total = un[i + j] + vn[i] + addend_carry
                un[i + j] = total & MASK
                addend_carry = total >> SHIFT
            }
            un[j + n] = un[j + n] + addend_carry & MASK
        } else {
            un[j + n] = top_difference
        }
        quotient[j] = qhat
        j -= 1
    }
    _trim(quotient)
    var remainder = un[0:n]
    _trim(remainder)
    if scale > 1 {
        let (unscaled, _zero) = _divmod_small(remainder, scale)
        remainder = unscaled
    }
    return (quotient, remainder)
}
