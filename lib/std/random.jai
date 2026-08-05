#: Random numbers: the xoshiro256** generator, the distributions built on it,
#: and module-level shorthands that share one default instance.
#:
#: `Random` is xoshiro256** — 256 bits of state, a period of 2**256 - 1, and one
#: multiply, one rotate and five xors per draw. It passes BigCrush. It is *not*
#: cryptographically secure: 256 bits of output reveal the state, so never use
#: it for keys, tokens or password salts.
#:
#: Every instance owns its state, so two generators never interfere. A generator
#: made with `Random.with_seed(n)` produces exactly the same stream on every run
#: and on every platform — that is what makes a failing randomised test
#: reproducible. A generator made with `Random()`, and the default instance
#: behind the module-level functions, is seeded from system entropy instead and
#: produces a different stream every run. **A reproducible run needs an explicit
#: seed**: call `Random.with_seed(n)`, or `seed(n)` to pin the default instance.
#:
#: ```
#: let generator = Random.with_seed(2026)
#: let roll = generator.int_range(1, 7)
#: roll >= 1 and roll <= 6                                      # true
#: Random.with_seed(1).next_int() == Random.with_seed(1).next_int()   # true
#: ```
from std.math import PI, exp, fabs, floor, is_finite, ln, sqrt
from std.str import StringBuilder

#: The characters `random_string` draws from when given no alphabet.
pub const ALPHANUMERIC: str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

#: The lowercase hexadecimal digits, for `random_string` and `uuid4`.
pub const HEX_DIGITS: str = "0123456789abcdef"

const _SIGN_BIT = -9223372036854775807 - 1

# 2 ** -53: one unit in the last place of a float mantissa, the scale that turns
# 53 random bits into a uniform value in [0, 1).
const _TWO_POW_53_INVERSE = 1.1102230246251565e-16

# Below this many trials the binomial is drawn by counting coin flips, which is
# exact and needs no distribution machinery. Above it the cost of one flip per
# trial stops being acceptable and the recursive beta split takes over.
const _BINOMIAL_FLIP_LIMIT = 40

# Below this mean, Knuth's product method draws a Poisson variate in `mean` + 1
# uniforms on average. Above it that is too many and PTRS takes over.
const _POISSON_KNUTH_LIMIT = 30.0

# The three SplitMix64 constants and the four xoshiro jump constants, written
# as 32-bit halves because the full values are above INT_MAX and so have no
# signed literal. The shift discards nothing: it moves `high` into bits 32-63,
# sign bit included, which is exactly the pattern wanted.
let _SPLIT_MIX_GAMMA = 0x9E3779B9 << 32 | 0x7F4A7C15
let _SPLIT_MIX_A = 0xBF58476D << 32 | 0x1CE4E5B9
let _SPLIT_MIX_B = 0x94D049BB << 32 | 0x133111EB

let _JUMP: list[int] = [
    0x180EC6D3 << 32 | 0x3CFD0ABA,
    0xD5A61266 << 32 | 0xF0C9392C,
    0xA9582618 << 32 | 0xE03FC9AA,
    0x39ABDC45 << 32 | 0x29B1661C,
]

# Lanczos g=7, n=9. Good to about 15 digits over the half-plane this uses.
let _LANCZOS: list[float] = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
]

#: A seedable, reproducible pseudo-random generator.
#:
#: Every method draws from the same 256-bit state, so a fixed seed fixes the
#: whole sequence of calls, whatever mixture of methods makes it up.
pub class Random {
    var state0: int
    var state1: int
    var state2: int
    var state3: int
    # The Marsaglia polar method makes two normal deviates at a time; the second
    # is kept here so that half of all `gauss` calls cost nothing.
    var spare: float?

    #: Create a generator seeded from system entropy.
    #:
    #: Its stream differs on every run. Use `with_seed` when the run has to be
    #: reproducible.
    pub fn init(self) {
        self.state0 = 0
        self.state1 = 0
        self.state2 = 0
        self.state3 = 0
        self.spare = null
        self.reseed(__prim__.random_u64())
    }

    #: Create a generator whose stream is fixed by `seed`.
    #:
    #: ```
    #: let a = Random.with_seed(11)
    #: let b = Random.with_seed(11)
    #: a.next_int() == b.next_int()    # true
    #: ```
    pub static fn with_seed(seed: int) -> Random {
        let generator = Random()
        generator.reseed(seed)
        return generator
    }

    #: Restart the stream from `seed`, discarding the current position.
    #:
    #: The 64-bit seed is expanded into the 256-bit state with SplitMix64, as
    #: the xoshiro authors specify: seeding the state directly from a small
    #: integer leaves it almost all zeros, and xoshiro needs thousands of draws
    #: to escape that.
    pub fn reseed(self, seed: int) -> void {
        let (after_first, first) = _split_mix(seed)
        let (after_second, second) = _split_mix(after_first)
        let (after_third, third) = _split_mix(after_second)
        let (_after_fourth, fourth) = _split_mix(after_third)
        self.state0 = first
        self.state1 = second
        self.state2 = third
        self.state3 = fourth
        # An all-zero state is the one fixed point of the generator, and it
        # emits nothing but zeros forever.
        if first == 0 and second == 0 and third == 0 and fourth == 0 { self.state0 = 1 }
        self.spare = null
    }

    #: Advance the state by 2**128 draws, in the time of 128 draws.
    #:
    #: This is how independent streams are made: give each worker a copy of the
    #: generator and jump the copy once more for each one. Their streams are
    #: then 2**128 draws apart and will not overlap.
    pub fn jump(self) -> void {
        var t0 = 0
        var t1 = 0
        var t2 = 0
        var t3 = 0
        for word in _JUMP {
            for bit in 0..64 {
                if word & 1 << bit != 0 {
                    t0 ^= self.state0
                    t1 ^= self.state1
                    t2 ^= self.state2
                    t3 ^= self.state3
                }
                self.next_int()
            }
        }
        self.state0 = t0
        self.state1 = t1
        self.state2 = t2
        self.state3 = t3
        self.spare = null
    }

    # --- Primitive draws -----------------------------------------------------

    #: Return a uniformly distributed 64-bit int, negative values included.
    #:
    #: Every one of the 2**64 bit patterns is equally likely, so the result
    #: spans the whole int range rather than only the non-negative half.
    pub fn next_int(self) -> int {
        let result = _rotate_left(self.state1 *% 5, 7) *% 9
        let shifted = self.state1 << 17
        self.state2 ^= self.state0
        self.state3 ^= self.state1
        self.state1 ^= self.state2
        self.state0 ^= self.state3
        self.state2 ^= shifted
        self.state3 = _rotate_left(self.state3, 45)
        return result
    }

    #: Return a uniform float in `[0.0, 1.0)`.
    #:
    #: Built from the top 53 bits of a draw, which is exactly the number of bits
    #: a float mantissa holds: every representable value in the interval that is
    #: a multiple of 2**-53 comes out with equal probability, and the result is
    #: never 1.0.
    #:
    #: ```
    #: let value = Random.with_seed(4).next_float()
    #: value >= 0.0 and value < 1.0    # true
    #: ```
    pub fn next_float(self) -> float {
        return _shift_right(self.next_int(), 11) * _TWO_POW_53_INVERSE
    }

    #: Return a uniform int in `[low, high)`, without modulo bias.
    #:
    #: The half-open interval matches `low..high`, so `int_range(0, xs.len())`
    #: is always a valid index. Raises `ValueError` when the interval is empty.
    #:
    #: A plain `draw % span` is not uniform: the residues below
    #: `2**64 % span` occur once more often than the rest, an error of one part
    #: in 2**64/span that is glaring for a span near 2**63 and invisible — but
    #: still there — for a span of 6. This masks the draw down to the next power
    #: of two and redraws when the result overshoots, so every value is reached
    #: through exactly the same number of bit patterns. The expected number of
    #: draws is below 2.
    #:
    #: ```
    #: let value = Random.with_seed(8).int_range(-5, 5)
    #: value >= -5 and value < 5    # true
    #: ```
    pub fn int_range(self, low: int, high: int) -> int {
        if low >= high {
            throw ValueError(f"int_range expects low < high, got low={low}, high={high}")
        }
        # Computed with wrapping arithmetic so that a span wider than an int,
        # such as int_range(INT_MIN, INT_MAX), stays exact as a bit pattern.
        let span = high -% low -% 1
        let mask = _smear(span)
        loop {
            let draw = self.next_int() & mask
            if _unsigned_le(draw, span) { return low +% draw }
        }
    }

    #: Return a uniform float in `[low, high)`.
    #:
    #: Raises `ValueError` unless `low < high` and both are finite. Compare
    #: `uniform`, which is the distribution-family spelling: it takes its bounds
    #: in either order and its interval is closed.
    pub fn float_range(self, low: float, high: float) -> float {
        if not low < high {
            throw ValueError(f"float_range expects low < high, got low={low}, high={high}")
        }
        if not is_finite(low) or not is_finite(high) {
            throw ValueError(f"float_range expects finite bounds, got [{low}, {high})")
        }
        loop {
            # `low + span * u` can round up onto `high` even though `u < 1.0`,
            # once the span is large enough that one ulp of the product exceeds
            # the gap. Redrawing keeps the interval genuinely half-open.
            let value = low + (high - low) * self.next_float()
            if value < high { return value }
        }
    }

    #: Return true with the given `probability`.
    #:
    #: Raises `ValueError` outside `[0.0, 1.0]`. The two endpoints are exact:
    #: probability 0.0 never draws, and 1.0 always does.
    #:
    #: ```
    #: Random.with_seed(6).bool(1.0)    # true
    #: ```
    pub fn bool(self, probability: float = 0.5) -> bool {
        if not (probability >= 0.0 and probability <= 1.0) {
            throw ValueError(f"probability must be in [0.0, 1.0], got {probability}")
        }
        if probability == 0.0 { return false }
        if probability == 1.0 { return true }
        return self.next_float() < probability
    }

    # --- Sequences -----------------------------------------------------------

    #: Return one element of `items`, uniformly.
    #:
    #: Raises `ValueError` for an empty list, which has nothing to return.
    #:
    #: ```
    #: Random.with_seed(12).choice([7])    # 7
    #: ```
    pub fn choice[T](self, items: list[T]) -> T {
        if items.len() == 0 { throw ValueError("choice from an empty list") }
        return items[self.int_range(0, items.len())]
    }

    #: Draw `count` elements of `items` with replacement.
    #:
    #: With `weights`, element `i` is drawn with probability proportional to
    #: `weights[i]`. Weights must be non-negative, finite, as many as there are
    #: elements, and not all zero. Selection is by binary search over the
    #: running totals, so a draw costs O(log n) after an O(n) setup.
    #:
    #: ```
    #: Random.with_seed(13).choices(["a", "b"], [1.0, 0.0], 3)    # ["a", "a", "a"]
    #: ```
    pub fn choices[T](
        self,
        items: list[T],
        weights: list[float]? = null,
        count: int = 1
    ) -> list[T] {
        if count < 0 { throw ValueError(f"count must be non-negative, got {count}") }
        if items.len() == 0 { throw ValueError("choices from an empty list") }
        let drawn: list[T] = []
        if weights is null {
            for _draw in 0..count { drawn.push(items[self.int_range(0, items.len())]) }
            return drawn
        }
        let totals = _running_totals(weights, items.len())
        let total = totals[totals.len() - 1]
        for _draw in 0..count {
            drawn.push(items[_first_above(totals, self.next_float() * total)])
        }
        return drawn
    }

    #: Draw `count` distinct elements of `items`, without replacement.
    #:
    #: The result is in random order, and an element repeated in `items` can be
    #: drawn twice — distinctness is by position, not by value. Runs in O(n)
    #: with one copy of the list, so it stays correct and fast even when `count`
    #: is the whole length. Raises `ValueError` unless `0 <= count <= len`.
    #:
    #: ```
    #: sorted(Random.with_seed(14).sample([1, 2, 3], 3))    # [1, 2, 3]
    #: ```
    pub fn sample[T](self, items: list[T], count: int) -> list[T] {
        if count < 0 or count > items.len() {
            throw ValueError(f"cannot sample {count} of {items.len()} elements")
        }
        let pool = items[0:items.len()]
        let drawn: list[T] = []
        for position in 0..count {
            let pick = position + self.int_range(0, pool.len() - position)
            let held = pool[position]
            pool[position] = pool[pick]
            pool[pick] = held
            drawn.push(pool[position])
        }
        return drawn
    }

    #: Shuffle `items` in place, uniformly over all its permutations.
    #:
    #: Fisher-Yates, drawing each position from the elements not yet placed.
    #: The variant that draws from the whole list every time is the classic bug:
    #: it produces n**n equally likely runs over n! permutations, and n! never
    #: divides n**n for n > 2, so some orderings come out more often.
    pub fn shuffle[T](self, items: list[T]) -> void {
        var position = items.len() - 1
        while position > 0 {
            let pick = self.int_range(0, position + 1)
            let held = items[position]
            items[position] = items[pick]
            items[pick] = held
            position -= 1
        }
    }

    #: Return a shuffled copy of `items`, leaving `items` untouched.
    #:
    #: ```
    #: sorted(Random.with_seed(15).shuffled([3, 1, 2]))    # [1, 2, 3]
    #: ```
    pub fn shuffled[T](self, items: list[T]) -> list[T] {
        let copy = items[0:items.len()]
        self.shuffle(copy)
        return copy
    }

    # --- Continuous distributions --------------------------------------------

    #: Return a uniform float between `low` and `high` inclusive.
    #:
    #: The bounds may be given in either order. Rounding can place the result
    #: exactly on `high`, which is why this is documented as closed; when the
    #: interval must exclude its upper end, use `float_range`.
    pub fn uniform(self, low: float, high: float) -> float {
        if not is_finite(low) or not is_finite(high) {
            throw ValueError(f"uniform expects finite bounds, got [{low}, {high}]")
        }
        return low + (high - low) * self.next_float()
    }

    #: Return a normal deviate with the given mean and standard deviation.
    #:
    #: The Marsaglia polar method: draw points in the unit square until one
    #: falls inside the unit circle, then map it to two independent standard
    #: normals. It needs no trigonometry, and the second deviate is cached for
    #: the next call, so the amortised cost is about 1.3 uniforms.
    #:
    #: Raises `ValueError` for a negative `deviation`. Compare `normal`, which
    #: takes a variance.
    #:
    #: ```
    #: Random.with_seed(16).gauss(5.0, 0.0)    # 5.0
    #: ```
    pub fn gauss(self, mean: float = 0.0, deviation: float = 1.0) -> float {
        if deviation < 0.0 or not is_finite(deviation) {
            throw ValueError(f"deviation must be finite and non-negative, got {deviation}")
        }
        let held = self.spare
        if held is not null {
            self.spare = null
            return mean + deviation * held
        }
        loop {
            let x = 2.0 * self.next_float() - 1.0
            let y = 2.0 * self.next_float() - 1.0
            let radius = x * x + y * y
            if radius >= 1.0 or radius == 0.0 { continue }
            let factor = sqrt(-2.0 * ln(radius) / radius)
            self.spare = y * factor
            return mean + deviation * x * factor
        }
    }

    #: Return a normal deviate parameterised as `N(mean, variance)`.
    #:
    #: The second argument is the **variance** sigma squared, matching the
    #: `N(mu, sigma**2)` notation; `gauss` takes the standard deviation sigma.
    #: Which one an API means is the most common source of silently
    #: wrong-scaled noise, so the two spellings are named apart rather than
    #: aliased together.
    #:
    #: Raises `ValueError` for a negative `variance`.
    pub fn normal(self, mean: float = 0.0, variance: float = 1.0) -> float {
        if variance < 0.0 or not is_finite(variance) {
            throw ValueError(f"variance must be finite and non-negative, got {variance}")
        }
        return self.gauss(mean, sqrt(variance))
    }

    #: Return an exponential deviate with the given `rate`.
    #:
    #: The mean is `1.0 / rate`. Drawn by inverting the cumulative distribution:
    #: `-ln(1 - u) / rate`. The `1 - u` is not idle — `next_float` can return
    #: 0.0 but never 1.0, so `ln(u)` would occasionally be asked for `ln(0.0)`
    #: while `ln(1 - u)` never is.
    #:
    #: Raises `ValueError` for a rate that is not positive.
    pub fn exponential(self, rate: float = 1.0) -> float {
        if rate <= 0.0 or not is_finite(rate) {
            throw ValueError(f"rate must be finite and positive, got {rate}")
        }
        return -ln(1.0 - self.next_float()) / rate
    }

    #: Return a gamma deviate with the given `shape` and `scale`.
    #:
    #: Marsaglia and Tsang's squeeze method, which accepts about 99% of its
    #: candidates on the first test and needs one normal and one uniform per
    #: attempt. A shape below 1.0 is drawn at `shape + 1.0` and scaled down by
    #: `u ** (1 / shape)`, since the density is unbounded at zero there.
    #:
    #: Raises `ValueError` unless both arguments are positive.
    pub fn gamma(self, shape: float, scale: float = 1.0) -> float {
        if shape <= 0.0 or not is_finite(shape) {
            throw ValueError(f"shape must be finite and positive, got {shape}")
        }
        if scale <= 0.0 or not is_finite(scale) {
            throw ValueError(f"scale must be finite and positive, got {scale}")
        }
        if shape < 1.0 {
            let boosted = self.gamma(shape + 1.0, scale)
            var u = self.next_float()
            while u <= 0.0 { u = self.next_float() }
            return boosted * u ** (1.0 / shape)
        }
        let d = shape - 1.0 / 3.0
        let c = 1.0 / sqrt(9.0 * d)
        loop {
            var x = 0.0
            var v = 0.0
            loop {
                x = self.gauss()
                v = 1.0 + c * x
                if v > 0.0 { break }
            }
            v = v * v * v
            let u = self.next_float()
            if u <= 0.0 { continue }
            let square = x * x
            if u < 1.0 - 0.0331 * square * square { return d * v * scale }
            if ln(u) < 0.5 * square + d * (1.0 - v + ln(v)) { return d * v * scale }
        }
    }

    #: Return a beta deviate with shape parameters `alpha` and `beta`.
    #:
    #: The result lies in `[0.0, 1.0]`. Normally taken as the ratio of two gamma
    #: deviates. When both shapes are below 1.0 that ratio is numerically
    #: hopeless — both gammas underflow towards zero and their quotient loses
    #: every significant bit — so Johnk's method is used there instead.
    #:
    #: Raises `ValueError` unless both shapes are positive.
    #:
    #: ```
    #: let value = Random.with_seed(17).beta(2.0, 3.0)
    #: value >= 0.0 and value <= 1.0    # true
    #: ```
    pub fn beta(self, alpha: float, beta: float) -> float {
        if alpha <= 0.0 or not is_finite(alpha) {
            throw ValueError(f"alpha must be finite and positive, got {alpha}")
        }
        if beta <= 0.0 or not is_finite(beta) {
            throw ValueError(f"beta must be finite and positive, got {beta}")
        }
        if alpha < 1.0 and beta < 1.0 {
            loop {
                let u = self.next_float() ** (1.0 / alpha)
                let v = self.next_float() ** (1.0 / beta)
                let total = u + v
                if total > 0.0 and total <= 1.0 { return u / total }
            }
        }
        let first = self.gamma(alpha)
        let second = self.gamma(beta)
        let total = first + second
        if total == 0.0 { return 0.5 }
        return first / total
    }

    #: Return a triangular deviate on `[low, high]` peaking at `mode`.
    #:
    #: `mode` defaults to the midpoint. This is the distribution to reach for
    #: when all that is known about a quantity is a floor, a ceiling and a most
    #: likely value — a task estimate, a latency budget — and it is drawn by
    #: inverting the cumulative distribution, so it needs one uniform.
    #:
    #: Raises `ValueError` unless `low < high` and `low <= mode <= high`.
    #:
    #: ```
    #: let value = Random.with_seed(18).triangular(1.0, 4.0, 3.0)
    #: value >= 1.0 and value <= 4.0    # true
    #: ```
    pub fn triangular(self, low: float = 0.0, high: float = 1.0, mode: float? = null) -> float {
        if not low < high {
            throw ValueError(f"triangular expects low < high, got low={low}, high={high}")
        }
        let peak = mode ?? (low + high) / 2.0
        if peak < low or peak > high {
            throw ValueError(f"mode must lie in [{low}, {high}], got {peak}")
        }
        let span = high - low
        let u = self.next_float()
        if u < (peak - low) / span { return low + sqrt(u * span * (peak - low)) }
        return high - sqrt((1.0 - u) * span * (high - peak))
    }

    # --- Discrete distributions ----------------------------------------------

    #: Return a Poisson deviate with the given `mean`.
    #:
    #: Below a mean of 30, Knuth's method multiplies uniforms until the product
    #: falls below `exp(-mean)`, which costs `mean + 1` draws on average. Above
    #: it, that is too many, and Hormann's PTRS transformed-rejection algorithm
    #: takes over at a cost independent of the mean.
    #:
    #: Raises `ValueError` for a negative or non-finite mean.
    #:
    #: ```
    #: Random.with_seed(19).poisson(0.0)    # 0
    #: ```
    pub fn poisson(self, mean: float) -> int {
        if mean < 0.0 or not is_finite(mean) {
            throw ValueError(f"mean must be finite and non-negative, got {mean}")
        }
        if mean < _POISSON_KNUTH_LIMIT {
            let limit = exp(-mean)
            var count = 0
            var product = 1.0
            loop {
                product *= self.next_float()
                if product <= limit { return count }
                count += 1
            }
        }
        let b = 0.931 + 2.53 * sqrt(mean)
        let a = -0.059 + 0.02483 * b
        let inverse_alpha = 1.1239 + 1.1328 / (b - 3.4)
        let boundary = 0.9277 - 3.6224 / (b - 2.0)
        let log_mean = ln(mean)
        loop {
            let u = self.next_float() - 0.5
            let v = self.next_float()
            let us = 0.5 - fabs(u)
            if us <= 0.0 { continue }
            let candidate = floor((2.0 * a / us + b) * u + mean + 0.43)
            if us >= 0.07 and v <= boundary { return int(candidate) }
            if candidate < 0.0 { continue }
            if us < 0.013 and v > us { continue }
            # The acceptance test takes ln(v), which has no value at zero; the
            # limit there is minus infinity, so a zero always accepts.
            if v <= 0.0 { return int(candidate) }
            let accept = candidate * log_mean - mean - _log_gamma(candidate + 1.0)
            if ln(v * inverse_alpha / (a / (us * us) + b)) <= accept { return int(candidate) }
        }
    }

    #: Return the number of successes in `trials` independent trials.
    #:
    #: Up to 40 trials the coins are simply flipped. Beyond that, Devroye's
    #: identity splits the problem instead: the `a`-th smallest of `n` uniforms
    #: is Beta(a, n + 1 - a), and knowing where it falls relative to
    #: `probability` fixes the outcome of half the trials and leaves a smaller
    #: binomial for the rest. Each step halves `trials`, so a billion trials
    #: cost about two dozen beta deviates rather than a billion flips, and the
    #: result is exact — not a normal approximation.
    #:
    #: Raises `ValueError` for negative trials or a probability outside
    #: `[0.0, 1.0]`.
    #:
    #: ```
    #: Random.with_seed(20).binomial(10, 1.0)    # 10
    #: ```
    pub fn binomial(self, trials: int, probability: float) -> int {
        if trials < 0 { throw ValueError(f"trials must be non-negative, got {trials}") }
        if not (probability >= 0.0 and probability <= 1.0) {
            throw ValueError(f"probability must be in [0.0, 1.0], got {probability}")
        }
        return self._binomial(trials, probability)
    }

    # --- Identifiers and text ------------------------------------------------

    #: Return a random version-4 UUID in the canonical 8-4-4-4-12 form.
    #:
    #: 122 of the 128 bits are random; the other six are the version and variant
    #: fields that RFC 4122 fixes. Since the source is this generator, a seeded
    #: instance produces the same identifiers every run — which is what a test
    #: wants and what a production key must never rely on.
    #:
    #: ```
    #: let identifier = Random.with_seed(21).uuid4()
    #: identifier.len()    # 36
    #: identifier[14]      # "4"
    #: ```
    pub fn uuid4(self) -> str {
        let octets: list[int] = []
        for _half in 0..2 {
            var word = self.next_int()
            for _step in 0..8 {
                octets.push(word & 0xFF)
                word = _shift_right(word, 8)
            }
        }
        octets[6] = octets[6] & 0x0F | 0x40
        octets[8] = octets[8] & 0x3F | 0x80
        let out = StringBuilder()
        for (position, value) in octets.enumerate() {
            if position == 4 or position == 6 or position == 8 or position == 10 {
                out.push("-")
            }
            out.push(HEX_DIGITS[value >> 4 & 0xF])
            out.push(HEX_DIGITS[value & 0xF])
        }
        return out.build()
    }

    #: Return a string of `length` characters drawn from `alphabet`.
    #:
    #: Each character is an independent uniform draw, so the alphabet may repeat
    #: characters to weight them. Raises `ValueError` for a negative length or
    #: an empty alphabet.
    #:
    #: ```
    #: Random.with_seed(22).random_string(8).len()          # 8
    #: Random.with_seed(23).random_string(4, "aaaa")        # "aaaa"
    #: ```
    pub fn random_string(self, length: int, alphabet: str = ALPHANUMERIC) -> str {
        if length < 0 { throw ValueError(f"length must be non-negative, got {length}") }
        if alphabet.len() == 0 { throw ValueError("alphabet must not be empty") }
        let out = StringBuilder()
        for _position in 0..length { out.push(alphabet[self.int_range(0, alphabet.len())]) }
        return out.build()
    }

    fn _binomial(self, trials: int, probability: float) -> int {
        if trials <= 0 or probability <= 0.0 { return 0 }
        if probability >= 1.0 { return trials }
        if trials <= _BINOMIAL_FLIP_LIMIT {
            var successes = 0
            for _trial in 0..trials {
                if self.next_float() < probability { successes += 1 }
            }
            return successes
        }
        let pivot = 1 + trials // 2
        let split = self.beta(float(pivot), float(trials + 1 - pivot))
        # `split` is where the pivot-th smallest of `trials` uniforms landed.
        # Everything below it is one smaller binomial; everything above it is
        # another. Only one of the two needs to be drawn.
        if split >= probability { return self._binomial(pivot - 1, probability / split) }
        return pivot + self._binomial(trials - pivot, (probability - split) / (1.0 - split))
    }
}

# The default instance behind the module-level functions. Seeded from system
# entropy at import, so a program that wants a reproducible run must call
# `seed`.
let _DEFAULT = Random()

#: Return the shared generator the module-level functions draw from.
#:
#: Reach for it to pass the default stream somewhere that wants a `Random`,
#: rather than making a second entropy-seeded instance.
pub fn default_random() -> Random { return _DEFAULT }

#: Reseed the shared generator, making every later module-level draw
#: reproducible.
#:
#: ```
#: seed(99)
#: let first = next_int()
#: seed(99)
#: next_int() == first    # true
#: ```
pub fn seed(value: int) -> void { _DEFAULT.reseed(value) }

#: Return a uniformly distributed 64-bit int from the shared generator.
pub fn next_int() -> int { return _DEFAULT.next_int() }

#: Return a uniform float in `[0.0, 1.0)` from the shared generator.
pub fn next_float() -> float { return _DEFAULT.next_float() }

#: Return a uniform int in `[low, high)` from the shared generator.
pub fn int_range(low: int, high: int) -> int { return _DEFAULT.int_range(low, high) }

#: Return a uniform float in `[low, high)` from the shared generator.
pub fn float_range(low: float, high: float) -> float { return _DEFAULT.float_range(low, high) }

#: Return a uniform float between `low` and `high` from the shared generator.
#:
#: The bounds may come in either order and the interval is closed; see
#: `Random.uniform`.
pub fn uniform(low: float, high: float) -> float { return _DEFAULT.uniform(low, high) }

#: Return true with the given `probability`, from the shared generator.
#:
#: Named apart from `Random.bool` so that the module-level name does not read
#: like a conversion to `bool`.
pub fn chance(probability: float = 0.5) -> bool { return _DEFAULT.bool(probability) }

#: Return one element of `items`, uniformly, from the shared generator.
pub fn choice[T](items: list[T]) -> T { return _DEFAULT.choice(items) }

#: Draw `count` elements of `items` with replacement from the shared generator.
pub fn choices[T](items: list[T], weights: list[float]? = null, count: int = 1) -> list[T] {
    return _DEFAULT.choices(items, weights, count)
}

#: Draw `count` distinct elements of `items` from the shared generator.
pub fn sample[T](items: list[T], count: int) -> list[T] {
    return _DEFAULT.sample(items, count)
}

#: Shuffle `items` in place, using the shared generator.
pub fn shuffle[T](items: list[T]) -> void { _DEFAULT.shuffle(items) }

#: Return a shuffled copy of `items`, using the shared generator.
pub fn shuffled[T](items: list[T]) -> list[T] { return _DEFAULT.shuffled(items) }

#: Return a normal deviate from the shared generator.
pub fn gauss(mean: float = 0.0, deviation: float = 1.0) -> float {
    return _DEFAULT.gauss(mean, deviation)
}

#: Return a random version-4 UUID from the shared generator.
#:
#: ```
#: uuid4().len()    # 36
#: ```
pub fn uuid4() -> str { return _DEFAULT.uuid4() }

#: Return a string of `length` characters drawn from `alphabet`.
#:
#: ```
#: random_string(12).len()    # 12
#: ```
pub fn random_string(length: int, alphabet: str = ALPHANUMERIC) -> str {
    return _DEFAULT.random_string(length, alphabet)
}

#: Advance a SplitMix64 state and return the new state with its output.
#:
#: SplitMix64 passes the statistical tests on its own and, crucially here, has
#: no bad states: every 64-bit seed expands into 256 well-mixed bits.
fn _split_mix(state: int) -> tuple[int, int] {
    let next_state = state +% _SPLIT_MIX_GAMMA
    var mixed = next_state
    mixed = (mixed ^ _shift_right(mixed, 30)) *% _SPLIT_MIX_A
    mixed = (mixed ^ _shift_right(mixed, 27)) *% _SPLIT_MIX_B
    return (next_state, mixed ^ _shift_right(mixed, 31))
}

#: Rotate the 64 bits of `value` left by `count`, which must be 1 to 63.
fn _rotate_left(value: int, count: int) -> int {
    return value << count | _shift_right(value, 64 - count)
}

#: Shift `value` right by `count` bits without sign extension, 1 <= count <= 63.
#:
#: `>>` on an int carries the sign bit down, which is right for arithmetic and
#: wrong for a bit pattern: the mask clears the copies it made.
fn _shift_right(value: int, count: int) -> int {
    return value >> count & ~(-1 << 64 - count)
}

#: Return the smallest `2**k - 1` that is at least `span`, read as unsigned.
fn _smear(span: int) -> int {
    var mask = span
    mask |= _shift_right(mask, 1)
    mask |= _shift_right(mask, 2)
    mask |= _shift_right(mask, 4)
    mask |= _shift_right(mask, 8)
    mask |= _shift_right(mask, 16)
    mask |= _shift_right(mask, 32)
    return mask
}

#: Compare two bit patterns as unsigned 64-bit values.
#:
#: Flipping the sign bit of both maps the unsigned order onto the signed one.
fn _unsigned_le(a: int, b: int) -> bool { return a ^ _SIGN_BIT <= b ^ _SIGN_BIT }

#: Return the running totals of `weights`, validating them on the way.
fn _running_totals(weights: list[float], expected: int) -> list[float] {
    if weights.len() != expected {
        throw ValueError(f"expected {expected} weights, got {weights.len()}")
    }
    let totals: list[float] = []
    var running = 0.0
    for weight in weights {
        if weight < 0.0 or not is_finite(weight) {
            throw ValueError(f"weights must be finite and non-negative, got {weight}")
        }
        running += weight
        totals.push(running)
    }
    if running <= 0.0 { throw ValueError("weights must not all be zero") }
    return totals
}

#: Return the index of the first running total strictly above `target`.
fn _first_above(totals: list[float], target: float) -> int {
    var low = 0
    var high = totals.len() - 1
    while low < high {
        let middle = low + (high - low) // 2
        if totals[middle] <= target { low = middle + 1 } else { high = middle }
    }
    return low
}

#: Return the natural logarithm of the gamma function, for `x >= 1.0`.
#:
#: Lanczos' approximation. The Poisson rejection test needs `ln(k!)` for values
#: of `k` far past where `k!` itself is representable, so the factorial can
#: never be formed and cancelled.
fn _log_gamma(x: float) -> float {
    let z = x - 1.0
    var series = _LANCZOS[0]
    for index in 1..9 { series += _LANCZOS[index] / (z + index) }
    let t = z + 7.5
    return 0.5 * ln(2.0 * PI) + (z + 0.5) * ln(t) - t + ln(series)
}
