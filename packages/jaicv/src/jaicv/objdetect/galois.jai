#: GF(256) and the Reed-Solomon decoder the QR reader corrects with.
#:
#: The field is the one every QR code uses: bytes, multiplied modulo the
#: primitive polynomial x^8 + x^4 + x^3 + x^2 + 1 (0x11D), which is what makes a
#: codeword an element and a block of codewords a polynomial over it.
#:
#: Correction is syndromes, Berlekamp-Massey, Chien search and Forney -- the
#: standard chain, written out rather than folded together because each step is
#: separately checkable and a QR block that fails is meant to be reported as
#: unreadable rather than guessed at.
#:
#: Exponent and logarithm tables are built once at import: a multiply is then
#: two lookups and an add, which matters because correcting one large code
#: multiplies on the order of a hundred thousand times.
fn build_exp() -> list[int] {
    var table: list[int] = []
    var value = 1
    for _index in 0..512 {
        table.push(value)
        value = value << 1
        if value >= 256 { value = value ^ 0x11D }
    }
    return table
}

fn build_log(exp: list[int]) -> list[int] {
    var table: list[int] = []
    for _index in 0..256 { table.push(0) }
    for index in 0..255 { table[exp[index]] = index }
    return table
}

let EXP = build_exp()
let LOG = build_log(EXP)

#: Product of two field elements.
pub fn gf_mul(a: int, b: int) -> int {
    if a == 0 or b == 0 { return 0 }
    return EXP[LOG[a] + LOG[b]]
}

#: Quotient of two field elements. Dividing by zero has no meaning here and
#: raises rather than returning something that would silently corrupt a block.
pub fn gf_div(a: int, b: int) -> int {
    if b == 0 { throw ValueError("division by zero in GF(256)") }
    if a == 0 { return 0 }
    return EXP[LOG[a] + 255 - LOG[b]]
}

#: `2` raised to a power, which is how the generator's roots are named.
pub fn gf_exp(power: int) -> int { return EXP[(power % 255 + 255) % 255] }

#: The multiplicative inverse.
pub fn gf_inv(a: int) -> int { return gf_div(1, a) }

fn poly_mul(a: list[int], b: list[int]) -> list[int] {
    var out: list[int] = []
    for _index in 0..a.len() + b.len() - 1 { out.push(0) }
    for i in 0..a.len() {
        if a[i] == 0 { continue }
        for j in 0..b.len() { out[i + j] = out[i + j] ^ gf_mul(a[i], b[j]) }
    }
    return out
}

#: Evaluate a polynomial whose first coefficient is its HIGHEST power, which is
#: how a codeword is laid out: the first byte is the leading term.
fn poly_eval(poly: list[int], x: int) -> int {
    var total = 0
    for coefficient in poly { total = gf_mul(total, x) ^ coefficient }
    return total
}

#: Evaluate a polynomial whose first coefficient is its CONSTANT term, which is
#: how the error locator and evaluator are built. Mixing the two conventions is
#: the mistake this pair of functions exists to prevent: the locator evaluated
#: the codeword's way finds roots that are the reverse of the real ones, and
#: the block is then "corrected" at the wrong positions.
fn poly_eval_low(poly: list[int], x: int) -> int {
    var total = 0
    var power = 1
    for coefficient in poly {
        total = total ^ gf_mul(coefficient, power)
        power = gf_mul(power, x)
    }
    return total
}

#: Correct `data` in place, given that its last `ec_count` codewords are parity.
#:
#: Returns whether the block was recoverable. A block with more errors than the
#: parity can locate comes back false rather than half-corrected, because a QR
#: reader that returns a plausible wrong string is worse than one that returns
#: nothing.
pub fn rs_correct(data: list[int], ec_count: int) -> bool {
    if ec_count <= 0 or data.len() <= ec_count { return false }

    #: Syndrome i is the block evaluated at the generator's i'th root. All zero
    #: means the block is already correct, which is the ordinary case and worth
    #: leaving before anything is allocated.
    var syndromes: list[int] = []
    var clean = true
    for index in 0..ec_count {
        let value = poly_eval(data, gf_exp(index))
        if value != 0 { clean = false }
        syndromes.push(value)
    }
    if clean { return true }

    #: Berlekamp-Massey: the shortest recurrence the syndromes satisfy, whose
    #: roots name where the errors are. Written with the register length `span`
    #: tracked explicitly rather than inferred from the polynomial's degree --
    #: the two part company as soon as a leading coefficient cancels, and
    #: inferring it corrects the wrong positions on blocks that have any.
    var locator = [1]
    var previous = [1]
    var span = 0
    var shift = 1
    var last_delta = 1
    for step in 0..ec_count {
        var delta = syndromes[step]
        for index in 1..span + 1 {
            if index >= locator.len() or index > step { break }
            delta = delta ^ gf_mul(locator[index], syndromes[step - index])
        }
        if delta == 0 {
            shift += 1
            continue
        }
        var scaled: list[int] = []
        for _index in 0..shift { scaled.push(0) }
        let factor = gf_div(delta, last_delta)
        for coefficient in previous { scaled.push(gf_mul(coefficient, factor)) }

        let saved = [value for value in locator]
        while locator.len() < scaled.len() { locator.push(0) }
        for index in 0..scaled.len() { locator[index] = locator[index] ^ scaled[index] }

        if 2 * span <= step {
            span = step + 1 - span
            previous = saved
            last_delta = delta
            shift = 1
        } else {
            shift += 1
        }
    }

    var degree = locator.len() - 1
    while degree > 0 and locator[degree] == 0 { degree -= 1 }
    if degree != span or span <= 0 or span > ec_count // 2 { return false }
    let errors = span

    #: Chien search. The locator's roots are the inverses of the error
    #: positions, counted as powers of the generator from the END of the block:
    #: exponent `power` is codeword index `len - 1 - power`.
    var powers: list[int] = []
    for power in 0..data.len() {
        if poly_eval_low(locator, gf_exp(255 - power)) == 0 { powers.push(power) }
    }
    if powers.len() != errors { return false }

    #: Forney. The evaluator is the syndromes times the locator, truncated to
    #: the parity length; the locator's derivative over this field keeps only
    #: its odd-power terms, since a term added to itself is zero.
    let product = poly_mul(syndromes, locator)
    var evaluator: list[int] = []
    for index in 0..ec_count {
        evaluator.push(if index < product.len() { product[index] } else { 0 })
    }

    var derivative: list[int] = []
    var index = 1
    while index < locator.len() {
        derivative.push(locator[index])
        derivative.push(0)
        index += 2
    }

    for power in powers {
        let inverse = gf_exp(255 - power)
        let bottom = poly_eval_low(derivative, inverse)
        if bottom == 0 { return false }
        let top = poly_eval_low(evaluator, inverse)
        #: The generator's roots start at alpha^0, so the magnitude carries a
        #: factor of the error position itself; with roots from alpha^1 it
        #: would not, and leaving it out corrects every block by the wrong
        #: byte while locating every error correctly.
        let magnitude = gf_div(gf_mul(gf_exp(power), top), bottom)
        data[data.len() - 1 - power] = data[data.len() - 1 - power] ^ magnitude
    }

    for index in 0..ec_count {
        if poly_eval(data, gf_exp(index)) != 0 { return false }
    }
    return true
}
