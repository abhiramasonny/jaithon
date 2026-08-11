#: Divisors, moduli and shift counts the compiled tier reads off the literal
#: that precedes them.
#:
#: Two claims are being checked. The first is arithmetic: `x // 2**s` is an
#: arithmetic shift right and `x %% 2**s` is the low s bits, both of which are
#: exact only because jaithon floors toward negative infinity and arm64's `asr`
#: and two's complement do the same -- `sdiv` does not, which is why the long
#: sequence exists at all. So every case here is run over NEGATIVE dividends as
#: well as positive ones, and over INT64_MIN, where the correction the shift
#: replaces has its own special case.
#:
#: The second is about control flow: "the instruction before this one pushed a
#: literal" is not the same as "this instruction always sees that literal". An
#: `if` in divisor position puts a literal immediately before the operator and
#: a jump from the other arm onto it, and folding the adjacent one is then a
#: wrong answer on half the calls. The `branchy` group below is that shape.
fn divs(x: int) -> int {
    var acc = 0
    acc = acc +% x // 2  # power of two: one asr
    acc = acc +% x // 8
    acc = acc +% x // 1024
    acc = acc +% x // 1   # shift of zero
    acc = acc +% x // 3   # not a power of two: guards fold, sdiv stays
    acc = acc +% x // -2  # negative: no shift is exact for this
    return acc
}

fn mods(x: int) -> int {
    var acc = 0
    acc = acc +% x % 2
    acc = acc +% x % 8
    acc = acc +% x % 1           # always zero
    acc = acc +% x % 5           # not a power of two
    acc = acc +% x % -4          # negative modulus floors the other way
    acc = acc +% x % 2147483648  # 2**31 -- an OP_CONST, not an OP_INT
    return acc
}

fn shifts(x: int) -> int {
    var acc = 0
    acc = acc +% (x << 0)
    acc = acc +% (x << 3)
    acc = acc +% (x << 63)  # the top of the range
    acc = acc +% (x >> 0)
    acc = acc +% (x >> 11)
    acc = acc +% (x >> 63)  # leaves the sign
    return acc
}

fn branchy(x: int, c: bool) -> int { return x // (if c { 2 } else { 3 }) }

fn branchyMod(x: int, c: bool) -> int { return x % (if c { 8 } else { 5 }) }

fn branchyShift(x: int, c: bool) -> int { return x >> (if c { 2 } else { 60 }) }

fn drive(n: int) -> int {
    var total = 0
    var i = 0
    while i < n {
        let x = i - n // 2
        total = total +% divs(x)
        total = total +% mods(x)
        total = total +% shifts(x)
        total = total +% branchy(x, i % 2 == 0)
        total = total +% branchyMod(x, i % 3 == 0)
        total = total +% branchyShift(x, i % 5 == 0)
        i += 1
    }
    return total
}

print(drive(20000))

# A zero divisor still has to reach the interpreter's error, and a literal one
# is exactly the case where the guard is no longer emitted -- so the fold must
# not be reached for it at all.
var caught = 0
try { print(1 // 0) } catch _e { caught = 1 }
print(caught)

# The same arithmetic at module scope, which is the OSR tier rather than the
# whole-function one.
var osr = 0
var j = 0
while j < 60000 {
    let y = j - 30000
    osr = osr +% y // 8 +% y % 16 +% (y >> 3) +% (y << 2)
    j += 1
}
print(osr)

# The ends of the range, where flooring and truncation disagree the most.
let intMin = -9223372036854775807 - 1
print(divs(intMin))
print(mods(intMin))
print(shifts(intMin))
print(divs(9223372036854775807))
print(mods(9223372036854775807))
print(-7 // 2)
print(-7 % 8)
print(-1 >> 63)
