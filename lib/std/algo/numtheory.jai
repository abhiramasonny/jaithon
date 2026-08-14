#: Number theory over the 64-bit integers: primality, factorisation, modular
#: arithmetic and continued fractions.
#:
#: Modular multiplication falls back to Russian-peasant doubling past the
#: square root of the largest int, so every routine here stays exact up to the
#: largest int rather than stopping at its square root, unlike a naive `a * b % m`.
from std.math import gcd, isqrt

# Products stay inside an int as long as both factors are below the square root
# of the largest int, which is floor(sqrt(2**63 - 1)).
const MULMOD_FAST_LIMIT = 3037000499

# The twelve bases that make Miller-Rabin deterministic for every int; see
# `is_prime`. They double as the trial-division wheel used by `factorize`.
let SMALL_PRIMES: list[int] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

#: Return true when `n` is prime.
#:
#: Deterministic for every int: no composite below 3.3 * 10**24 is a strong
#: pseudoprime to all twelve witness bases (Sorenson and Webster, 2015), and the
#: largest int is under 10**19, so a pass here is a proof, not a probability.
#:
#: ```
#: is_prime(2147483647)    # true
#: is_prime(561)           # false
#: is_prime(1)             # false
#: ```
pub fn is_prime(n: int) -> bool {
    if n < 2 { return false }
    for prime in SMALL_PRIMES {
        if n == prime { return true }
        if n % prime == 0 { return false }
    }
    var odd = n - 1
    var power = 0
    while odd % 2 == 0 {
        odd //= 2
        power += 1
    }
    for base in SMALL_PRIMES {
        var residue = mod_pow(base, odd, n)
        if residue == 1 or residue == n - 1 { continue }
        var witnessed = true
        for _ in 0..power - 1 {
            residue = _mulmod(residue, residue, n)
            if residue == n - 1 {
                witnessed = false
                break
            }
        }
        if witnessed { return false }
    }
    return true
}

#: Return every prime up to and including `limit`, ascending.
#:
#: O(limit log log limit) time, O(limit) space. Crossing out starts at `p * p`
#: since every smaller multiple of `p` already carries a smaller prime factor.
#:
#: ```
#: sieve_of_eratosthenes(30)    # [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
#: sieve_of_eratosthenes(1)     # []
#: ```
pub fn sieve_of_eratosthenes(limit: int) -> list[int] {
    if limit < 0 { throw ValueError(f"limit must be non-negative, got {limit}") }
    if limit < 2 { return [] }
    let composite = [false for _ in 0..=limit]
    var p = 2
    while p * p <= limit {
        if not composite[p] {
            var multiple = p * p
            while multiple <= limit {
                composite[multiple] = true
                multiple += p
            }
        }
        p += 1
    }
    let primes: list[int] = []
    for candidate in 2..=limit {
        if not composite[candidate] { primes.push(candidate) }
    }
    return primes
}

#: Return every prime in the closed interval `low..=high`, ascending.
#:
#: Only primes up to `sqrt(high)` are sieved in full, then the interval is
#: crossed out in one pass: O(sqrt(high) + width) space, so a window high up the
#: number line needs no memory proportional to where it sits.
#:
#: ```
#: segmented_sieve(100, 120)    # [101, 103, 107, 109, 113]
#: segmented_sieve(0, 5)        # [2, 3, 5]
#: ```
pub fn segmented_sieve(low: int, high: int) -> list[int] {
    if low < 0 { throw ValueError(f"low must be non-negative, got {low}") }
    if high < low { throw ValueError(f"high must be at least low, got {high} < {low}") }
    let start = max(low, 2)
    if high < start { return [] }
    let width = high - start + 1
    let composite = [false for _ in 0..width]
    for p in sieve_of_eratosthenes(isqrt(high)) {
        var multiple = max(p * p, (start + p - 1) // p * p)
        while multiple <= high {
            composite[multiple - start] = true
            multiple += p
        }
    }
    let primes: list[int] = []
    for offset in 0..width {
        if not composite[offset] { primes.push(start + offset) }
    }
    return primes
}

#: Return the prime factors of `n` with multiplicity, ascending.
#:
#: Small factors are stripped by trial division; the rest is split by Pollard's
#: rho with Brent's cycle detection, O(m**0.25) expected multiplications per
#: composite `m`. `n` must be positive.
#:
#: ```
#: factorize(360)             # [2, 2, 2, 3, 3, 5]
#: factorize(600851475143)    # [71, 839, 1471, 6857]
#: factorize(1)               # []
#: ```
pub fn factorize(n: int) -> list[int] {
    if n < 1 { throw ValueError(f"n must be positive, got {n}") }
    let factors: list[int] = []
    var remaining = n
    for prime in SMALL_PRIMES {
        while remaining % prime == 0 {
            factors.push(prime)
            remaining //= prime
        }
    }
    _split_into(remaining, factors)
    return sorted(factors)
}

#: Return the prime factorisation of `n` as (prime, exponent) pairs, ascending.
#:
#: Same cost as `factorize`, which it groups.
#:
#: ```
#: prime_factors(360)    # [(2, 3), (3, 2), (5, 1)]
#: prime_factors(97)     # [(97, 1)]
#: ```
pub fn prime_factors(n: int) -> list[tuple[int, int]] {
    let flat = factorize(n)
    let grouped: list[tuple[int, int]] = []
    var i = 0
    while i < flat.len() {
        var j = i
        while j < flat.len() and flat[j] == flat[i] { j += 1 }
        grouped.push((flat[i], j - i))
        i = j
    }
    return grouped
}

#: Return every positive divisor of `n`, ascending.
#:
#: Built from the factorisation, so the cost is that of `factorize` plus
#: O(d log d) to sort the d divisors.
#:
#: ```
#: divisors(28)    # [1, 2, 4, 7, 14, 28]
#: divisors(1)     # [1]
#: ```
pub fn divisors(n: int) -> list[int] {
    var found = [1]
    for (prime, exponent) in prime_factors(n) {
        let grown: list[int] = []
        for divisor in found {
            var power = 1
            for step in 0..=exponent {
                grown.push(divisor * power)
                if step < exponent { power *= prime }
            }
        }
        found = grown
    }
    return sorted(found)
}

#: Count the positive divisors of `n`.
#:
#: The product of `exponent + 1` over the factorisation; costs no more than
#: `factorize` and never builds the divisor list.
#:
#: ```
#: divisor_count(28)    # 6
#: divisor_count(1)     # 1
#: ```
pub fn divisor_count(n: int) -> int {
    var count = 1
    for (_, exponent) in prime_factors(n) { count *= exponent + 1 }
    return count
}

#: Sum the positive divisors of `n`.
#:
#: Uses the multiplicative form, summing `1 + p + ... + p**e` per prime; costs
#: no more than `factorize`. Raises `OverflowError` if the sum overflows an int.
#:
#: ```
#: divisor_sum(28)    # 56
#: divisor_sum(12)    # 28
#: ```
pub fn divisor_sum(n: int) -> int {
    var total = 1
    for (prime, exponent) in prime_factors(n) {
        var term = 1
        var power = 1
        for _ in 0..exponent {
            power *= prime
            term += power
        }
        total *= term
    }
    return total
}

#: Count the integers in `1..=n` that are coprime to `n`.
#:
#: Euler's totient as `n * product(1 - 1/p)` over the distinct primes of `n`.
#: Division is done before multiplication, so nothing overflows or rounds.
#:
#: ```
#: euler_phi(36)    # 12
#: euler_phi(97)    # 96
#: euler_phi(1)     # 1
#: ```
pub fn euler_phi(n: int) -> int {
    var total = n
    for (prime, _) in prime_factors(n) { total = total // prime * (prime - 1) }
    return total
}

#: Return the Möbius function of `n`: 0, 1 or -1.
#:
#: Zero when `n` is divisible by a square, otherwise the parity of the number of
#: distinct prime factors. Costs no more than `factorize`.
#:
#: ```
#: mobius(30)    # -1
#: mobius(6)     # 1
#: mobius(12)    # 0
#: mobius(1)     # 1
#: ```
pub fn mobius(n: int) -> int {
    let factors = prime_factors(n)
    for (_, exponent) in factors {
        if exponent > 1 { return 0 }
    }
    return factors.len() % 2 == 0 ? 1 : -1
}

#: Return `(g, x, y)` with `a*x + b*y == g` and `g == gcd(a, b)`.
#:
#: Iterative extended Euclidean algorithm, O(log min(|a|, |b|)) divisions. `g`
#: is never negative; the (x, y) returned is the recurrence's own least-magnitude pair.
#:
#: ```
#: extended_gcd(240, 46)    # (2, -9, 47)
#: extended_gcd(7, 0)       # (7, 1, 0)
#: ```
pub fn extended_gcd(a: int, b: int) -> tuple[int, int, int] {
    var old_remainder = a
    var remainder = b
    var old_x = 1
    var x = 0
    var old_y = 0
    var y = 1
    while remainder != 0 {
        let quotient = old_remainder // remainder
        let next_remainder = old_remainder - quotient * remainder
        let next_x = old_x - quotient * x
        let next_y = old_y - quotient * y
        old_remainder = remainder
        remainder = next_remainder
        old_x = x
        x = next_x
        old_y = y
        y = next_y
    }
    if old_remainder < 0 { return (-old_remainder, -old_x, -old_y) }
    return (old_remainder, old_x, old_y)
}

#: Return the inverse of `a` modulo `modulus`, in `0..modulus`.
#:
#: O(log modulus) divisions. Raises `ValueError` when `a` and `modulus` are not
#: coprime, naming the common factor. Modulo 1 every value is 0.
#:
#: ```
#: mod_inverse(3, 11)    # 4
#: mod_inverse(-1, 11)   # 10
#: ```
pub fn mod_inverse(a: int, modulus: int) -> int {
    if modulus < 1 { throw ValueError(f"modulus must be positive, got {modulus}") }
    if modulus == 1 { return 0 }
    let (divisor, x, _) = extended_gcd(_normalize(a, modulus), modulus)
    if divisor != 1 {
        throw ValueError(f"{a} has no inverse modulo {modulus}: both are divisible by {divisor}")
    }
    return _normalize(x, modulus)
}

#: Return `base ** exponent` modulo `modulus`, in `0..modulus`.
#:
#: Square-and-multiply, O(log exponent) modular multiplications. A negative
#: exponent inverts `base` first, so `base` must be coprime to `modulus`.
#:
#: ```
#: mod_pow(2, 100, 1000000007)    # 976371285
#: mod_pow(3, -1, 11)             # 4
#: ```
pub fn mod_pow(base: int, exponent: int, modulus: int) -> int {
    if modulus < 1 { throw ValueError(f"modulus must be positive, got {modulus}") }
    if modulus == 1 { return 0 }
    if exponent < 0 { return mod_pow(mod_inverse(base, modulus), -exponent, modulus) }
    var result = 1
    var factor = _normalize(base, modulus)
    var remaining = exponent
    while remaining > 0 {
        if remaining & 1 == 1 { result = _mulmod(result, factor, modulus) }
        factor = _mulmod(factor, factor, modulus)
        remaining >>= 1
    }
    return result
}

#: Solve a system of congruences `x == residues[i] (mod moduli[i])`.
#:
#: `x` is the unique solution in `0..m` (m = lcm of the moduli), or null if
#: inconsistent. Moduli need not be coprime; raises `OverflowError` if the lcm overflows.
#:
#: ```
#: chinese_remainder([2, 3, 2], [3, 5, 7])    # (23, 105)
#: chinese_remainder([1, 4], [6, 9])          # (13, 18)
#: chinese_remainder([1, 2], [4, 6])          # null
#: ```
pub fn chinese_remainder(residues: list[int], moduli: list[int]) -> tuple[int, int]? {
    if residues.len() != moduli.len() {
        throw ValueError(f"got {residues.len()} residues but {moduli.len()} moduli")
    }
    var solution = 0
    var combined = 1
    for i in 0..moduli.len() {
        let modulus = moduli[i]
        if modulus < 1 {
            throw ValueError(f"moduli must be positive, got {modulus} at index {i}")
        }
        let residue = _normalize(residues[i], modulus)
        let divisor = gcd(combined, modulus)
        if (residue - solution) % divisor != 0 { return null }
        let step = modulus // divisor
        let multiple = combined // divisor * modulus
        let shift = _normalize((residue - solution) // divisor, step)
        let ticks = _mulmod(shift, mod_inverse(combined // divisor, step), step)
        let advance = _mulmod(combined % multiple, ticks, multiple)
        solution = _addmod(solution % multiple, advance, multiple)
        combined = multiple
    }
    return (solution, combined)
}

#: Return the least `x` with `base ** x == target (mod modulus)`, or null.
#:
#: Baby-step giant-step, O(sqrt(modulus)) time and memory. `base` must be
#: coprime to `modulus` — that's what makes the giant steps invertible.
#:
#: ```
#: discrete_log(3, 13, 17)    # 4
#: discrete_log(2, 3, 5)      # 3
#: discrete_log(2, 5, 7)      # null
#: ```
pub fn discrete_log(base: int, target: int, modulus: int) -> int? {
    if modulus < 1 { throw ValueError(f"modulus must be positive, got {modulus}") }
    if modulus == 1 { return 0 }
    let generator = _normalize(base, modulus)
    let wanted = _normalize(target, modulus)
    if gcd(generator, modulus) != 1 {
        throw ValueError(f"base {base} must be coprime to modulus {modulus}")
    }
    if wanted == 1 { return 0 }
    let stride = isqrt(modulus - 1) + 1
    let baby: dict[int, int] = {}
    var value = wanted
    for j in 0..stride {
        baby[value] = j
        value = _mulmod(value, generator, modulus)
    }
    let giant = mod_pow(generator, stride, modulus)
    var reach = 1
    for i in 1..=stride {
        reach = _mulmod(reach, giant, modulus)
        if reach in baby { return i * stride - baby[reach] }
    }
    return null
}

#: Return the least primitive root modulo `modulus`, or null when none exists.
#:
#: A primitive root exists only for moduli 1, 2, 4, `p**k` and `2 * p**k` (`p`
#: an odd prime); null otherwise. Checked via `phi / q` for each prime `q` of `phi`.
#:
#: ```
#: primitive_root(7)     # 3
#: primitive_root(14)    # 3
#: primitive_root(8)     # null
#: ```
pub fn primitive_root(modulus: int) -> int? {
    if modulus < 1 { throw ValueError(f"modulus must be positive, got {modulus}") }
    if modulus == 1 { return 0 }
    let order = euler_phi(modulus)
    let factors = prime_factors(order)
    for candidate in 1..modulus {
        if gcd(candidate, modulus) != 1 { continue }
        var generates = true
        for (prime, _) in factors {
            if mod_pow(candidate, order // prime, modulus) == 1 {
                generates = false
                break
            }
        }
        if generates { return candidate }
    }
    return null
}

#: Return the Legendre symbol of `a` over the odd prime `p`: 0, 1 or -1.
#:
#: 1 when `a` is a non-zero quadratic residue mod `p`, -1 when it isn't, 0 when
#: `p` divides `a` — via Euler's criterion, O(log p) modular multiplications.
#:
#: ```
#: legendre_symbol(2, 7)    # 1
#: legendre_symbol(3, 7)    # -1
#: legendre_symbol(7, 7)    # 0
#: ```
pub fn legendre_symbol(a: int, p: int) -> int {
    if p < 3 or p % 2 == 0 or not is_prime(p) {
        throw ValueError(f"p must be an odd prime, got {p}")
    }
    let residue = mod_pow(_normalize(a, p), (p - 1) // 2, p)
    if residue == 0 { return 0 }
    return residue == 1 ? 1 : -1
}

#: Return the Jacobi symbol of `a` over the odd positive `n`: 0, 1 or -1.
#:
#: Legendre symbol generalised to composite `n`, via reciprocity in O(log n)
#: steps without factorising. Note: 1 doesn't imply `a` is a residue unless `n` is prime.
#:
#: ```
#: jacobi_symbol(1001, 9907)    # -1
#: jacobi_symbol(2, 15)         # 1
#: ```
pub fn jacobi_symbol(a: int, n: int) -> int {
    if n < 1 or n % 2 == 0 { throw ValueError(f"n must be odd and positive, got {n}") }
    var top = _normalize(a, n)
    var bottom = n
    var symbol = 1
    while top != 0 {
        while top % 2 == 0 {
            top //= 2
            # Two is a residue exactly for moduli 1 and 7 mod 8.
            let residue = bottom % 8
            if residue == 3 or residue == 5 { symbol = -symbol }
        }
        let swap = top
        top = bottom
        bottom = swap
        if top % 4 == 3 and bottom % 4 == 3 { symbol = -symbol }
        top = top % bottom
    }
    return bottom == 1 ? symbol : 0
}

#: Return a square root of `a` modulo the prime `p`, or null when none exists.
#:
#: Tonelli-Shanks; one exponentiation when `p == 3 (mod 4)`, else O(log(p)**2)
#: multiplications worst case. The other root is `p - result`.
#:
#: ```
#: tonelli_shanks(2, 7)     # 4
#: tonelli_shanks(5, 7)     # null
#: tonelli_shanks(0, 7)     # 0
#: ```
pub fn tonelli_shanks(a: int, p: int) -> int? {
    if p < 2 or not is_prime(p) { throw ValueError(f"p must be prime, got {p}") }
    let value = _normalize(a, p)
    if p == 2 { return value }
    if value == 0 { return 0 }
    if mod_pow(value, (p - 1) // 2, p) != 1 { return null }
    if p % 4 == 3 { return mod_pow(value, (p + 1) // 4, p) }
    var odd = p - 1
    var power = 0
    while odd % 2 == 0 {
        odd //= 2
        power += 1
    }
    var non_residue = 2
    while mod_pow(non_residue, (p - 1) // 2, p) != p - 1 { non_residue += 1 }
    var exponent = power
    var shifter = mod_pow(non_residue, odd, p)
    var remainder = mod_pow(value, odd, p)
    var root = mod_pow(value, (odd + 1) // 2, p)
    while remainder != 1 {
        var order = 0
        var square = remainder
        while square != 1 {
            square = _mulmod(square, square, p)
            order += 1
        }
        var factor = shifter
        for _ in 0..exponent - order - 1 { factor = _mulmod(factor, factor, p) }
        exponent = order
        shifter = _mulmod(factor, factor, p)
        remainder = _mulmod(remainder, shifter, p)
        root = _mulmod(root, factor, p)
    }
    return root
}

#: Return the continued fraction expansion of `numerator / denominator`.
#:
#: Terms are the Euclidean algorithm's quotients: exact, finite, O(log
#: min(|numerator|, denominator)) long. Every term after the first is positive, even for negative fractions.
#:
#: ```
#: continued_fraction(415, 93)    # [4, 2, 6, 7]
#: continued_fraction(-7, 3)      # [-3, 1, 2]
#: ```
pub fn continued_fraction(numerator: int, denominator: int) -> list[int] {
    if denominator == 0 { throw ValueError("denominator must be non-zero, got 0") }
    var top = numerator
    var bottom = denominator
    if bottom < 0 {
        top = -top
        bottom = -bottom
    }
    let terms: list[int] = []
    while bottom != 0 {
        let quotient = top // bottom
        terms.push(quotient)
        let remainder = top - quotient * bottom
        top = bottom
        bottom = remainder
    }
    return terms
}

#: Return the convergents of a continued fraction, as (numerator, denominator).
#:
#: Each is the best rational approximation to the whole fraction among all
#: rationals with a denominator no larger than its own. Runs in O(len(terms))
#: using the standard `h = a*h1 + h2` recurrence.
#:
#: ```
#: convergents([4, 2, 6, 7])           # [(4, 1), (9, 2), (58, 13), (415, 93)]
#: convergents([3, 7, 15, 1])          # [(3, 1), (22, 7), (333, 106), (355, 113)]
#: ```
pub fn convergents(terms: list[int]) -> list[tuple[int, int]] {
    var previous_numerator = 0
    var previous_denominator = 1
    var numerator = 1
    var denominator = 0
    let out: list[tuple[int, int]] = []
    for term in terms {
        let next_numerator = term * numerator + previous_numerator
        let next_denominator = term * denominator + previous_denominator
        previous_numerator = numerator
        previous_denominator = denominator
        numerator = next_numerator
        denominator = next_denominator
        out.push((numerator, denominator))
    }
    return out
}

#: Return the closest fraction to `value` with a denominator at most the bound.
#:
#: Walks the float's exact continued fraction until the denominator would
#: exceed the bound, then takes the better of the last convergent and the
#: furthest semiconvergent — why 311/99 beats 22/7 for pi below 100. Built in
#: exact integer arithmetic; only the final comparison is done in float, and can
#: differ from the exact answer only when the two candidates are exactly
#: equidistant (then the smaller denominator wins).
#:
#: ```
#: rational_approximation(3.141592653589793, 100)     # (311, 99)
#: rational_approximation(3.141592653589793, 1000)    # (355, 113)
#: rational_approximation(0.25, 100)                  # (1, 4)
#: ```
pub fn rational_approximation(value: float, max_denominator: int) -> tuple[int, int] {
    if max_denominator < 1 {
        throw ValueError(f"max_denominator must be positive, got {max_denominator}")
    }
    if value == 0.0 { return (0, 1) }
    let magnitude = abs(value)
    if magnitude >= 9.223372036854776e18 {
        throw ValueError(f"value must fit in an int, got {value}")
    }
    var scaled = magnitude
    var denominator = 1
    while scaled != float(int(scaled)) and
        scaled < 4.611686018427388e18 and
        denominator <= 2305843009213693952 {
        scaled *= 2.0
        denominator *= 2
    }
    let (top, bottom) = _limit_denominator(int(scaled), denominator, max_denominator)
    return (value < 0.0 ? -top : top, bottom)
}

#: Return the smallest prime strictly greater than `n`.
#:
#: Scans odd candidates with `is_prime`. No prime gap below 4 * 10**18 exceeds
#: 1476, so this is a short walk however large `n` is.
#:
#: ```
#: next_prime(100)    # 101
#: next_prime(0)      # 2
#: ```
pub fn next_prime(n: int) -> int {
    if n < 2 { return 2 }
    var candidate = n % 2 == 0 ? n + 1 : n + 2
    while not is_prime(candidate) { candidate += 2 }
    return candidate
}

#: Return the largest prime strictly less than `n`, or null when there is none.
#:
#: The mirror of `next_prime`; null for `n <= 2`, since 2 is the least prime.
#:
#: ```
#: prev_prime(100)    # 97
#: prev_prime(3)      # 2
#: prev_prime(2)      # null
#: ```
pub fn prev_prime(n: int) -> int? {
    if n <= 2 { return null }
    if n == 3 { return 2 }
    var candidate = n % 2 == 0 ? n - 1 : n - 2
    while candidate >= 3 and not is_prime(candidate) { candidate -= 2 }
    return candidate >= 3 ? candidate : 2
}

#: Return the `k`-th prime, counting from `nth_prime(1) == 2`.
#:
#: Sieves up to `16 * k` (holds at least k primes for k up to about a million;
#: doubles the bound if not). O(p log log p) time, O(p) space for the answer p.
#:
#: ```
#: nth_prime(1)       # 2
#: nth_prime(1000)    # 7919
#: ```
pub fn nth_prime(k: int) -> int {
    if k < 1 { throw ValueError(f"k must be positive, got {k}") }
    var bound = max(32, k * 16)
    loop {
        let primes = sieve_of_eratosthenes(bound)
        if primes.len() >= k { return primes[k - 1] }
        bound *= 2
    }
}

#: Count the primes up to and including `limit`.
#:
#: Lucy_Hedgehog's method, not a sieve: keeps only the O(sqrt(limit)) distinct
#: values of `limit // i`, removing each prime's multiples from the counts, in
#: O(limit**0.75) time — a sieve would need a billion flags to count primes below 10**9.
#:
#: ```
#: prime_count(100)        # 25
#: prime_count(1000000)    # 78498
#: prime_count(1)          # 0
#: ```
pub fn prime_count(limit: int) -> int {
    if limit < 2 { return 0 }
    let root = isqrt(limit)
    let keys: list[int] = []
    for i in 1..=root { keys.push(limit // i) }
    var smaller = keys[keys.len() - 1] - 1
    while smaller >= 1 {
        keys.push(smaller)
        smaller -= 1
    }
    let counts: dict[int, int] = {}
    for key in keys { counts[key] = key - 1 }
    for p in 2..=root {
        if counts[p] > counts[p - 1] {
            let below = counts[p - 1]
            let square = p * p
            for key in keys {
                if key < square { break }
                counts[key] -= counts[key // p] - below
            }
        }
    }
    return counts[limit]
}

fn _normalize(value: int, modulus: int) -> int {
    let remainder = value % modulus
    if remainder < 0 { return remainder + modulus }
    return remainder
}

fn _addmod(a: int, b: int, modulus: int) -> int {
    if a >= modulus - b { return a - (modulus - b) }
    return a + b
}

fn _mulmod(a: int, b: int, modulus: int) -> int {
    var addend = _normalize(a, modulus)
    var multiplier = _normalize(b, modulus)
    if modulus <= MULMOD_FAST_LIMIT { return addend * multiplier % modulus }
    var result = 0
    while multiplier > 0 {
        if multiplier & 1 == 1 { result = _addmod(result, addend, modulus) }
        addend = _addmod(addend, addend, modulus)
        multiplier >>= 1
    }
    return result
}

fn _split_into(n: int, factors: list[int]) -> void {
    if n == 1 { return }
    if is_prime(n) {
        factors.push(n)
        return
    }
    let divisor = _pollard_rho(n)
    _split_into(divisor, factors)
    _split_into(n // divisor, factors)
}

fn _pollard_rho(n: int) -> int {
    if n % 2 == 0 { return 2 }
    var constant = 1
    loop {
        let divisor = _brent_cycle(n, constant)
        if divisor > 1 and divisor < n { return divisor }
        constant += 1
    }
}

fn _brent_cycle(n: int, constant: int) -> int {
    let shift = constant % n
    let batch = 128
    var y = shift
    var x = 0
    var trail = 0
    var divisor = 1
    var range_size = 1
    var product = 1
    while divisor == 1 {
        x = y
        for _ in 0..range_size { y = _addmod(_mulmod(y, y, n), shift, n) }
        var done = 0
        while done < range_size and divisor == 1 {
            trail = y
            for _ in 0..min(batch, range_size - done) {
                y = _addmod(_mulmod(y, y, n), shift, n)
                product = _mulmod(product, _abs_difference(x, y), n)
            }
            divisor = gcd(product, n)
            done += batch
        }
        range_size *= 2
    }
    if divisor == n {
        divisor = 1
        while divisor == 1 {
            trail = _addmod(_mulmod(trail, trail, n), shift, n)
            divisor = gcd(_abs_difference(x, trail), n)
        }
    }
    return divisor
}

fn _abs_difference(a: int, b: int) -> int {
    return a > b ? a - b : b - a
}

fn _limit_denominator(numerator: int, denominator: int, max_denominator: int) -> tuple[int, int] {
    var top = numerator
    var bottom = denominator
    var previous_numerator = 0
    var previous_denominator = 1
    var best_numerator = 1
    var best_denominator = 0
    loop {
        let term = top // bottom
        let next_denominator = previous_denominator + term * best_denominator
        if next_denominator > max_denominator { break }
        let next_numerator = previous_numerator + term * best_numerator
        previous_numerator = best_numerator
        previous_denominator = best_denominator
        best_numerator = next_numerator
        best_denominator = next_denominator
        let remainder = top - term * bottom
        top = bottom
        bottom = remainder
        if bottom == 0 { return (best_numerator, best_denominator) }
    }
    let reach = (max_denominator - previous_denominator) // best_denominator
    let far_denominator = previous_denominator + reach * best_denominator
    if 2.0 * float(bottom) * float(far_denominator) <= float(denominator) {
        return (best_numerator, best_denominator)
    }
    return (previous_numerator + reach * best_numerator, far_denominator)
}
