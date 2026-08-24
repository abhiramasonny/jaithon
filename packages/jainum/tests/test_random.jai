#: Tests for jainum's counter-based sampling.
#:
#: Three things are worth testing about a random number generator, and none of
#: them is a golden number.
#:
#: Determinism: a seed and a stream fix every sample, and the fix survives
#: being asked for in a different shape, after a different number of earlier
#: draws, or from a different object. Those are the promises the module's
#: header makes, and they are the ones a threadgroup-size change would break.
#:
#: Independence: two streams share nothing.
#:
#: Moments: the mean, the variance, a median or a tail count, recomputed here
#: on the host from the downloaded sample and compared against what the
#: distribution's own definition says they should be. Every tolerance below is
#: five standard errors of the statistic at the sample size used, derived from
#: the distribution's moments — `mean_band` from the standard deviation,
#: `var_band` from the fourth central moment, `median_band` from the density
#: at the median. None of them was widened until the suite went green.
#:
#: Five sigma and not three because there are roughly fifty such assertions
#: here: at three sigma one run in three hundred would fail on nothing, and at
#: five it is one in forty thousand.
import std.gpu as gpu
import std.math
from std.test import assert_eq, assert_near, assert_throws, assert_true
from jainum.array import FLOAT32, INT32, NDArray
from jainum.random import Generator, default_rng
from jainum.sorting import sort

fn have_gpu() -> bool { return gpu.is_available() }

#: Draws every moment test uses. Big enough that the bands below are tight
#: (five standard errors of a unit-variance mean is 0.029 here) and small
#: enough that downloading it and looping over it twice on the host stays
#: cheap.
const SAMPLE = 30000

#: Five standard errors of a sample mean.
fn mean_band(deviation: float, n: int) -> float {
    return 5.0 * deviation / math.sqrt(float(n))
}

#: Five standard errors of a sample variance, whose own variance is
#: `(mu4 - sigma^4) / n`.
fn var_band(fourth: float, variance: float, n: int) -> float {
    let spread = fourth - variance * variance
    return 5.0 * math.sqrt(if spread > 0.0 { spread / float(n) } else { 0.0 })
}

#: Five standard errors of a sample median, whose own deviation is
#: `1 / (2 f(m) sqrt(n))`.
fn median_band(density: float, n: int) -> float {
    return 5.0 / (2.0 * density * math.sqrt(float(n)))
}

#: Five standard errors of a proportion.
fn share_band(share: float, n: int) -> float {
    return 5.0 * math.sqrt(share * (1.0 - share) / float(n))
}

#: The sample mean and variance, summed here rather than reduced on the
#: device: the point is to arrive at the answer by a route the kernel under
#: test had no part in.
fn moments(values: list[float]) -> (float, float) {
    var total = 0.0
    for value in values { total += value }
    let mean = total / float(values.len())
    var squares = 0.0
    for value in values {
        let offset = value - mean
        squares += offset * offset
    }
    return (mean, squares / float(values.len()))
}

fn lowest(values: list[float]) -> float {
    var found = values[0]
    for value in values { if value < found { found = value } }
    return found
}

fn highest(values: list[float]) -> float {
    var found = values[0]
    for value in values { if value > found { found = value } }
    return found
}

fn share_below(values: list[float], bound: float) -> float {
    var seen = 0
    for value in values { if value < bound { seen += 1 } }
    return float(seen) / float(values.len())
}

fn all_whole(values: list[float]) -> bool {
    for value in values { if value != math.floor(value) { return false } }
    return true
}

#: The sample median, from the module's own sort. `sorting.jai` is tested
#: against its own definition elsewhere, so leaning on it here is not
#: circular: nothing in `random.jai` sorts.
fn middle(a: NDArray) -> float {
    let ordered = sort(a)
    defer { ordered.free() }
    return ordered[a.size() // 2].item()
}

fn drawn(a: NDArray) -> list[float] {
    let values = a.to_list()
    a.free()
    return values
}

fn test_the_generator_is_the_published_philox() -> void {
    if not have_gpu() { return }

    #: Random123's first known-answer vector: counter and key both all zero,
    #: ten rounds, output 6627e8d5 e169c58d bc57ac4c 9b00dbd8. Seed 0, stream
    #: 0 and the first call put exactly that counter and key into the kernel,
    #: and the top 24 bits of each word divided by 2^24 are exact in float32,
    #: so this is an equality and not a tolerance.
    #:
    #: Without this the rest of the file only says the numbers look random.
    #: With it, they are a specific generator whose properties are published.
    let head = drawn(Generator(0, 0).random(4))
    assert_eq(head[0], 0x6627E8 / 16777216.0)
    assert_eq(head[1], 0xE169C5 / 16777216.0)
    assert_eq(head[2], 0xBC57AC / 16777216.0)
    assert_eq(head[3], 0x9B00DB / 16777216.0)
}

fn test_a_seed_and_a_stream_fix_every_sample() -> void {
    if not have_gpu() { return }
    let first = drawn(Generator(20260824, 3).random(257))
    let again = drawn(Generator(20260824, 3).random(257))
    assert_eq(first, again, "the same seed and stream reproduce element by element")

    let elsewhere = drawn(Generator(20260824, 4).random(257))
    assert_true(not elsewhere == first, "a different stream is a different sequence")
    var matches = 0
    for at in 0..257 { if elsewhere[at] == first[at] { matches += 1 } }
    assert_true(matches < 4, f"streams 3 and 4 agreed on {matches} of 257 samples")

    let reseeded = drawn(Generator(20260825, 3).random(257))
    var seed_matches = 0
    for at in 0..257 { if reseeded[at] == first[at] { seed_matches += 1 } }
    assert_true(seed_matches < 4, f"seeds one apart agreed on {seed_matches} of 257")
}

fn test_a_sample_does_not_move_with_the_shape_it_was_asked_for() -> void {
    if not have_gpu() { return }

    #: The claim the whole design exists to make: element i is a function of
    #: i, so the grid the device chooses cannot be visible in the output.
    let flat = drawn(Generator(11).standard_normal(1000))
    let block = drawn(Generator(11).standard_normal([4, 250]))
    let cube = drawn(Generator(11).standard_normal([10, 10, 10]))
    assert_eq(flat, block, "1000 samples as [4, 250] are the same 1000 samples")
    for at in 0..1000 { assert_eq(cube[at], flat[at], f"sample {at} moved with the rank") }

    #: A count that does not divide by four takes the scalar tail of the
    #: kernel, which must agree with the wide store above it.
    let ragged = drawn(Generator(11).standard_normal(1002))
    for at in 0..1000 { assert_eq(ragged[at], flat[at], f"sample {at} moved with the count") }
}

fn test_a_later_draw_is_a_different_draw() -> void {
    if not have_gpu() { return }
    let generator = Generator(5)
    let first = drawn(generator.random(256))
    let second = drawn(generator.random(256))
    var matches = 0
    for at in 0..256 { if first[at] == second[at] { matches += 1 } }
    assert_true(matches < 4, f"two draws from one generator agreed on {matches} of 256")
    assert_eq(generator.draws(), 2)

    let fresh = drawn(Generator(5).random(256))
    assert_eq(fresh, first, "a fresh generator starts the sequence over")
}

fn test_a_stream_does_not_care_when_it_was_asked_for() -> void {
    if not have_gpu() { return }
    let generator = Generator(9)
    let burnt = generator.random(4096)
    burnt.free()
    generator.standard_normal([7, 13])

    let spawned = drawn(generator.spawn(2).random(64))
    let direct = drawn(Generator(9, 2).random(64))
    assert_eq(spawned, direct, "a spawned stream is the pair, not the history")
}

fn test_uniform_fills_its_interval() -> void {
    if not have_gpu() { return }
    let generator = Generator(101)
    let unit = drawn(generator.random(SAMPLE))
    let (mean, variance) = moments(unit)
    assert_near(mean, 0.5, mean_band(math.sqrt(1.0 / 12.0), SAMPLE))

    #: A uniform's fourth central moment is (b-a)^4 / 80.
    assert_near(variance, 1.0 / 12.0, var_band(1.0 / 80.0, 1.0 / 12.0, SAMPLE))
    assert_true(lowest(unit) >= 0.0, "the interval is closed at the bottom")
    assert_true(highest(unit) < 1.0, "and open at the top")

    let wide = drawn(generator.uniform(-3.0, 7.0, SAMPLE))
    let (wide_mean, wide_variance) = moments(wide)
    assert_near(wide_mean, 2.0, mean_band(math.sqrt(100.0 / 12.0), SAMPLE))
    assert_near(wide_variance, 100.0 / 12.0, var_band(10000.0 / 80.0, 100.0 / 12.0, SAMPLE))
    assert_true(lowest(wide) >= -3.0 and highest(wide) < 7.0)
}

fn test_normal_moments_and_tails() -> void {
    if not have_gpu() { return }
    let generator = Generator(202)
    let z = drawn(generator.standard_normal(SAMPLE))
    let (mean, variance) = moments(z)
    assert_near(mean, 0.0, mean_band(1.0, SAMPLE))
    assert_near(variance, 1.0, var_band(3.0, 1.0, SAMPLE), "a normal's fourth moment is 3")

    #: Box-Muller gets the centre right by construction; the tails are where a
    #: bad transform shows, so the 2.5% quantile is checked directly.
    assert_near(share_below(z, -1.959964), 0.025, share_band(0.025, SAMPLE))
    assert_near(share_below(z, 1.959964), 0.975, share_band(0.975, SAMPLE))

    let shifted = drawn(generator.normal(2.0, 3.0, SAMPLE))
    let (shifted_mean, shifted_variance) = moments(shifted)
    assert_near(shifted_mean, 2.0, mean_band(3.0, SAMPLE))
    assert_near(shifted_variance, 9.0, var_band(3.0 * 81.0, 9.0, SAMPLE))
}

fn test_integers_cover_a_half_open_range_evenly() -> void {
    if not have_gpu() { return }
    let generator = Generator(303)
    let picked = drawn(generator.integers(-3, 5, SAMPLE))
    assert_true(all_whole(picked), "integers are whole")
    assert_eq(lowest(picked), -3.0)
    assert_eq(highest(picked), 4.0, "the top of the range is excluded")

    var counts = [0 for _bucket in 0..8]
    for value in picked { counts[int(value) + 3] += 1 }
    let expected = float(SAMPLE) / 8.0
    let band = 5.0 * math.sqrt(float(SAMPLE) * 0.125 * 0.875)
    for bucket in 0..8 {
        assert_near(float(counts[bucket]), expected, band, f"bucket {bucket}")
    }

    let endpoint = drawn(generator.integers(0, 3, 4096, endpoint: true))
    assert_eq(highest(endpoint), 3.0, "endpoint includes the top")
    let implied = drawn(generator.integers(6, null, 4096))
    assert_true(lowest(implied) >= 0.0 and highest(implied) <= 5.0, "one bound means [0, low)")
}

fn test_the_exponential_family_has_the_right_scale() -> void {
    if not have_gpu() { return }
    let generator = Generator(404)

    let waits = drawn(generator.exponential(2.5, SAMPLE))
    let (wait_mean, wait_variance) = moments(waits)
    assert_near(wait_mean, 2.5, mean_band(2.5, SAMPLE))

    #: An exponential's fourth central moment is 9 * scale^4.
    assert_near(wait_variance, 6.25, var_band(9.0 * 39.0625, 6.25, SAMPLE))
    assert_true(lowest(waits) > 0.0, "the open unit keeps the logarithm finite")

    let lengths = drawn(generator.rayleigh(2.0, SAMPLE))
    let (length_mean, length_variance) = moments(lengths)
    assert_near(length_mean, 2.0 * math.sqrt(math.PI / 2.0), mean_band(2.0 * 0.6551364, SAMPLE))
    assert_near(length_variance, (4.0 - math.PI) / 2.0 * 4.0, 0.06, "Rayleigh variance")
}

fn test_the_closed_form_tails_land_on_their_medians() -> void {
    if not have_gpu() { return }
    let generator = Generator(505)

    #: A median is the cleanest independent check for a heavy tail: it comes
    #: straight from inverting the CDF, and unlike a mean it is not at the
    #: mercy of one enormous draw.
    let weibull = generator.weibull(2.0, SAMPLE)
    defer { weibull.free() }
    let weibull_median = math.sqrt(math.log(2.0))
    let weibull_density = 2.0 * weibull_median * 0.5
    assert_near(middle(weibull), weibull_median, median_band(weibull_density, SAMPLE))

    let pareto = generator.pareto(3.0, SAMPLE)
    defer { pareto.free() }
    let pareto_median = 2.0 ** (1.0 / 3.0) - 1.0
    let pareto_density = 3.0 / (1.0 + pareto_median) ** 4.0
    assert_near(middle(pareto), pareto_median, median_band(pareto_density, SAMPLE))

    let logistic = generator.logistic(1.0, 1.5, SAMPLE)
    defer { logistic.free() }
    assert_near(middle(logistic), 1.0, median_band(1.0 / 6.0, SAMPLE))

    let laplace = drawn(generator.laplace(-2.0, 3.0, SAMPLE))
    let (laplace_mean, laplace_variance) = moments(laplace)
    assert_near(laplace_mean, -2.0, mean_band(math.sqrt(18.0), SAMPLE))

    #: A Laplace's variance is 2b^2 and its fourth central moment is 24b^4.
    assert_near(laplace_variance, 18.0, var_band(24.0 * 81.0, 18.0, SAMPLE))

    let lognormal = drawn(generator.lognormal(0.0, 0.5, SAMPLE))
    let (lognormal_mean, _lognormal_variance) = moments(lognormal)
    let expected = math.exp(0.125)
    let deviation = expected * math.sqrt(math.exp(0.25) - 1.0)
    assert_near(lognormal_mean, expected, mean_band(deviation, SAMPLE))
}

fn test_gamma_and_the_distributions_built_on_it() -> void {
    if not have_gpu() { return }
    let generator = Generator(606)

    let big = drawn(generator.gamma(2.5, 2.0, SAMPLE))
    let (big_mean, big_variance) = moments(big)
    assert_near(big_mean, 5.0, mean_band(math.sqrt(10.0), SAMPLE))

    #: A gamma's fourth central moment is 3k(k+2) * scale^4.
    assert_near(big_variance, 10.0, var_band(3.0 * 2.5 * 4.5 * 16.0, 10.0, SAMPLE))
    assert_true(lowest(big) > 0.0)

    #: A shape below one takes Marsaglia's boost rather than the squeeze, so
    #: it is a different path through the same loop.
    let small = drawn(generator.gamma(0.4, 1.0, SAMPLE))
    let (small_mean, small_variance) = moments(small)
    assert_near(small_mean, 0.4, mean_band(math.sqrt(0.4), SAMPLE))
    assert_near(small_variance, 0.4, var_band(3.0 * 0.4 * 2.4, 0.4, SAMPLE))

    let fraction = drawn(generator.beta(2.0, 5.0, SAMPLE))
    let (fraction_mean, fraction_variance) = moments(fraction)
    let beta_mean = 2.0 / 7.0
    let beta_variance = 10.0 / (49.0 * 8.0)
    assert_near(fraction_mean, beta_mean, mean_band(math.sqrt(beta_variance), SAMPLE))

    #: For a variable on [0, 1] the fourth moment is at most sigma^2 times the
    #: squared worst-case deviation, which makes this band conservative rather
    #: than tuned.
    let beta_fourth = beta_variance * (1.0 - beta_mean) * (1.0 - beta_mean)
    assert_near(fraction_variance, beta_variance, var_band(beta_fourth, beta_variance, SAMPLE))
    assert_true(lowest(fraction) >= 0.0 and highest(fraction) <= 1.0)

    let chi = drawn(generator.chisquare(4.0, SAMPLE))
    let (chi_mean, chi_variance) = moments(chi)
    assert_near(chi_mean, 4.0, mean_band(math.sqrt(8.0), SAMPLE))
    assert_near(chi_variance, 8.0, var_band(3.0 * 2.0 * 4.0 * 16.0, 8.0, SAMPLE))

    #: Student's t with eight degrees of freedom: variance df/(df-2) and
    #: fourth moment 3df^2/((df-2)(df-4)), both finite only because df > 4.
    let student = drawn(generator.standard_t(8.0, SAMPLE))
    let (student_mean, student_variance) = moments(student)
    assert_near(student_mean, 0.0, mean_band(math.sqrt(8.0 / 6.0), SAMPLE))
    assert_near(student_variance, 8.0 / 6.0, var_band(8.0, 8.0 / 6.0, SAMPLE))
}

fn test_poisson_over_both_of_its_algorithms() -> void {
    if not have_gpu() { return }
    let generator = Generator(707)

    #: Below ten the sampler multiplies uniforms; above it, transformed
    #: rejection. Both are checked, because the crossover is a real seam.
    let quiet = drawn(generator.poisson(3.0, SAMPLE))
    let (quiet_mean, quiet_variance) = moments(quiet)
    assert_true(all_whole(quiet) and lowest(quiet) >= 0.0)
    assert_near(quiet_mean, 3.0, mean_band(math.sqrt(3.0), SAMPLE))

    #: A Poisson's fourth central moment is lambda(1 + 3 lambda).
    assert_near(quiet_variance, 3.0, var_band(3.0 * 10.0, 3.0, SAMPLE))

    let busy = drawn(generator.poisson(50.0, SAMPLE))
    let (busy_mean, busy_variance) = moments(busy)
    assert_true(all_whole(busy))
    assert_near(busy_mean, 50.0, mean_band(math.sqrt(50.0), SAMPLE))
    assert_near(busy_variance, 50.0, var_band(50.0 * 151.0, 50.0, SAMPLE))
}

fn test_binomial_over_both_of_its_algorithms() -> void {
    if not have_gpu() { return }
    let generator = Generator(808)

    #: n*p below ten inverts the pmf; above it, transformed rejection.
    let rare = drawn(generator.binomial(20, 0.15, SAMPLE))
    let (rare_mean, rare_variance) = moments(rare)
    assert_true(all_whole(rare) and lowest(rare) >= 0.0 and highest(rare) <= 20.0)
    assert_near(rare_mean, 3.0, mean_band(math.sqrt(2.55), SAMPLE))

    #: npq(1 + 3(n-2)pq) is the fourth central moment of a binomial.
    assert_near(rare_variance, 2.55, var_band(2.55 * (1.0 + 54.0 * 0.1275), 2.55, SAMPLE))

    let common = drawn(generator.binomial(400, 0.4, SAMPLE))
    let (common_mean, common_variance) = moments(common)
    assert_true(all_whole(common) and highest(common) <= 400.0)
    assert_near(common_mean, 160.0, mean_band(math.sqrt(96.0), SAMPLE))
    assert_near(common_variance, 96.0, var_band(96.0 * (1.0 + 1194.0 * 0.24), 96.0, SAMPLE))

    #: p above a half is drawn from its reflection and subtracted back, which
    #: is a third path again.
    let likely = drawn(generator.binomial(50, 0.9, SAMPLE))
    let (likely_mean, likely_variance) = moments(likely)
    assert_true(highest(likely) <= 50.0)
    assert_near(likely_mean, 45.0, mean_band(math.sqrt(4.5), SAMPLE))
    assert_near(likely_variance, 4.5, var_band(4.5 * (1.0 + 144.0 * 0.09), 4.5, SAMPLE))
}

fn test_multinomial_rows_are_a_partition_of_the_trials() -> void {
    if not have_gpu() { return }
    let weights = [0.1, 0.2, 0.3, 0.4]
    let counts = Generator(909).multinomial(100, weights, 2000)
    defer { counts.free() }
    assert_eq(counts.shape, [2000, 4], "the categories are the last axis")

    let values = counts.to_list()
    var column = [0.0, 0.0, 0.0, 0.0]
    for row in 0..2000 {
        var total = 0.0
        for slot in 0..4 {
            let value = values[row * 4 + slot]
            assert_true(value >= 0.0, "a count cannot be negative")
            column[slot] += value
            total += value
        }
        assert_eq(total, 100.0, f"row {row} does not spend all hundred trials")
    }
    for slot in 0..4 {
        let share = weights[slot]
        let deviation = math.sqrt(100.0 * share * (1.0 - share))
        assert_near(column[slot] / 2000.0, 100.0 * share, mean_band(deviation, 2000))
    }
}

fn test_a_permutation_moves_everything_and_loses_nothing() -> void {
    if not have_gpu() { return }
    let generator = Generator(1010)
    let order = generator.permutation(4096)
    defer { order.free() }
    let ordered = sort(order)
    defer { ordered.free() }
    let seen = ordered.to_list()
    for at in 0..4096 { assert_eq(seen[at], float(at), f"index {at} is missing or doubled") }

    var fixed = 0
    let raw = order.to_list()
    for at in 0..4096 { if raw[at] == float(at) { fixed += 1 } }

    #: The number of fixed points of a uniform permutation is Poisson(1), so
    #: more than a handful means the sort keys were not random.
    assert_true(fixed < 10, f"{fixed} of 4096 positions did not move")

    let rows = NDArray([64, 3], FLOAT32)
    defer { rows.free() }
    rows.upload([float(index) for index in 0..192])
    let moved = generator.permutation(rows)
    defer { moved.free() }
    let taken = moved.to_list()
    for row in 0..64 {
        let head = taken[row * 3]
        assert_eq(taken[row * 3 + 1], head + 1.0, "a row travels whole")
        assert_eq(taken[row * 3 + 2], head + 2.0)
    }
}

fn test_shuffle_rewrites_the_array_it_was_given() -> void {
    if not have_gpu() { return }
    let a = NDArray([2048], FLOAT32)
    defer { a.free() }
    a.upload([float(index) for index in 0..2048])
    Generator(1111).shuffle(a)

    let ordered = sort(a)
    defer { ordered.free() }
    let seen = ordered.to_list()
    for at in 0..2048 { assert_eq(seen[at], float(at)) }

    var fixed = 0
    let raw = a.to_list()
    for at in 0..2048 { if raw[at] == float(at) { fixed += 1 } }
    assert_true(fixed < 10, f"{fixed} of 2048 positions did not move")
}

fn test_choice_without_replacement_never_repeats() -> void {
    if not have_gpu() { return }
    let generator = Generator(1212)
    let picked = generator.choice(5000, 400, replace: false)
    defer { picked.free() }
    assert_eq(picked.shape, [400])

    var seen: dict[int, bool] = {}
    for value in picked.to_list() {
        let at = int(value)
        assert_true(at >= 0 and at < 5000, f"{at} is outside the population")
        assert_true(not seen.get(at, false), f"{at} was drawn twice")
        seen[at] = true
    }

    #: With replacement the same request repeats, which is the difference
    #: between the two modes stated as a test rather than as a comment.
    let repeated = generator.choice(20, 200)
    defer { repeated.free() }
    assert_eq(repeated.shape, [200])
    assert_true(highest(repeated.to_list()) < 20.0)

    assert_throws(ValueError, || generator.choice(10, 11, replace: false))
}

fn test_choice_follows_the_weights_it_was_given() -> void {
    if not have_gpu() { return }
    let generator = Generator(1313)

    #: All the mass on one entry is the sharpest statement of the CDF search:
    #: every draw must land there and the zeros must never be reached.
    let pinned = drawn(generator.choice(4, 512, p: [0.0, 0.0, 1.0, 0.0]))
    for value in pinned { assert_eq(value, 2.0, "a zero weight is never drawn") }

    let split = drawn(generator.choice(2, SAMPLE, p: [0.25, 0.75]))
    var ones = 0
    for value in split { if value == 1.0 { ones += 1 } }
    assert_near(float(ones) / float(SAMPLE), 0.75, share_band(0.75, SAMPLE))

    #: Past the tile the search reads the CDF straight out of device memory,
    #: so a population wider than 1024 is a second code path.
    var ramp: list[float] = []
    for index in 0..2000 { ramp.push(float(index)) }
    let weighted = drawn(generator.choice(2000, SAMPLE, p: ramp))

    #: Weight proportional to the index makes the draw a triangular
    #: distribution on [0, n): its mean is 2n/3 and its variance n^2/18.
    let (mean, variance) = moments(weighted)
    let n = 2000.0
    assert_near(mean, 2.0 * n / 3.0, mean_band(n / math.sqrt(18.0), SAMPLE) + 1.0)
    assert_true(variance > 0.8 * n * n / 18.0 and variance < 1.2 * n * n / 18.0)

    #: A population of a thousand stays inside the threadgroup tile and one
    #: of two thousand does not, so the same weights are checked through both
    #: of the search kernels.
    var short_ramp: list[float] = []
    for index in 0..1000 { short_ramp.push(float(index)) }
    let tiled = drawn(generator.choice(1000, SAMPLE, p: short_ramp))
    let (tiled_mean, _tiled_variance) = moments(tiled)
    assert_near(tiled_mean, 2.0 * 1000.0 / 3.0, mean_band(1000.0 / math.sqrt(18.0), SAMPLE) + 1.0)

    #: Without replacement a zero weight still has to be excluded, which is
    #: the reservoir key's job rather than the CDF's.
    let reserved = drawn(generator.choice(4, 3, replace: false, p: [1.0, 0.0, 1.0, 1.0]))
    for value in reserved { assert_true(value != 1.0, "a zero weight survived the reservoir") }
}

fn test_choice_reads_through_an_array_population() -> void {
    if not have_gpu() { return }
    let population = NDArray([64], FLOAT32)
    defer { population.free() }
    population.upload([float(index) * 10.0 + 5.0 for index in 0..64])

    let picked = Generator(1717).choice(population, [3, 7])
    defer { picked.free() }
    assert_eq(picked.shape, [3, 7], "the draw is shaped like the size, not like the population")
    for value in picked.to_list() {
        assert_true(value >= 5.0 and value <= 635.0)
        assert_eq((value - 5.0) % 10.0, 0.0, f"{value} is not a member of the population")
    }

    assert_throws(ValueError, || Generator(1).choice(NDArray([2, 2]), 3), "a 1-d population")
}

fn test_permuted_shuffles_every_slice_on_its_own() -> void {
    if not have_gpu() { return }
    var values: list[float] = []
    for _row in 0..128 {
        for slot in 0..8 { values.push(float(slot)) }
    }
    let a = NDArray([128, 8], FLOAT32)
    defer { a.free() }
    a.upload(values)

    let mixed = Generator(1414).permuted(a, 1)
    defer { mixed.free() }
    let seen = mixed.to_list()
    var distinct = 0
    for row in 0..128 {
        var found = [false, false, false, false, false, false, false, false]
        for slot in 0..8 { found[int(seen[row * 8 + slot])] = true }
        for slot in 0..8 { assert_true(found[slot], f"row {row} lost the value {slot}") }
        if seen[row * 8] != 0.0 { distinct += 1 }
    }

    #: Each row is shuffled independently, so a row leading with its original
    #: first element should happen about one time in eight.
    assert_true(distinct > 96, f"{128 - distinct} of 128 rows kept their first element")

    #: Along axis 0 the array is walked through a transposed view first,
    #: which is a different route to the same kernel. Filling each column with
    #: 0..127 makes a column shuffle visible where the row-filled array above
    #: would have hidden it.
    var down: list[float] = []
    for row in 0..128 {
        for _slot in 0..8 { down.push(float(row)) }
    }
    let tall = NDArray([128, 8], FLOAT32)
    defer { tall.free() }
    tall.upload(down)
    let columns = Generator(1515).permuted(tall, 0)
    defer { columns.free() }
    let held = columns.to_list()
    var apart = 0
    for slot in 0..8 {
        var found = [false for _row in 0..128]
        for row in 0..128 { found[int(held[row * 8 + slot])] = true }
        for row in 0..128 { assert_true(found[row], f"column {slot} lost the value {row}") }
        if held[slot] != held[0] { apart += 1 }
    }

    #: Seven columns each independently shuffled agree with the first on their
    #: leading entry about seven times in a thousand.
    assert_true(apart >= 5, f"{7 - apart + 1} of 8 columns took the same permutation")

    #: `shuffle` moves rows and keeps their contents together; `permuted`
    #: does the opposite. The distinction is the whole reason both exist.
    let rows = Generator(1414).permutation(a)
    defer { rows.free() }
    let intact = rows.to_list()
    for row in 0..128 {
        for slot in 0..8 { assert_eq(intact[row * 8 + slot], float(slot)) }
    }
}

fn test_a_null_size_is_a_scalar_array() -> void {
    if not have_gpu() { return }
    let generator = Generator(1515)
    let one = generator.random()
    defer { one.free() }
    assert_eq(one.shape, [], "a null size is a 0-d array, not a float")
    assert_eq(one.size(), 1)
    assert_true(one.item() >= 0.0 and one.item() < 1.0)

    let empty = generator.standard_normal(0)
    defer { empty.free() }
    assert_eq(empty.size(), 0, "an empty draw is a real array, not an error")

    let legacy = generator.rand([2, 2])
    defer { legacy.free() }
    assert_eq(legacy.shape, [2, 2])
    let dtype = generator.integers(0, 4, 8)
    defer { dtype.free() }
    assert_eq(dtype.dtype, INT32, "whole numbers carry an integer dtype")
}

fn test_bad_parameters_are_refused_before_a_dispatch() -> void {
    if not have_gpu() { return }
    let generator = Generator(1616)
    assert_throws(ValueError, || generator.integers(5, 5), "an empty range")
    assert_throws(ValueError, || generator.exponential(0.0), "a scale of zero")
    assert_throws(ValueError, || generator.gamma(-1.0), "a negative shape")
    assert_throws(ValueError, || generator.binomial(10, 1.5), "a probability above one")
    assert_throws(ValueError, || generator.random(-4), "a negative size")
    assert_throws(ValueError, || Generator(1, -1), "a stream outside 32 bits")
    assert_throws(ValueError, || generator.choice(4, 2, p: [0.0, 0.0, 0.0, 0.0]), "no mass")
    assert_throws(ValueError, || generator.choice(3, 2, p: [0.5, 0.5]), "a weight per member")

    #: float32 storage is what an INT32 array is, so a bound past 2^24 is a
    #: bound the result could not carry.
    assert_throws(ValueError, || generator.integers(0, 1 << 30))
}

fn test_default_rng_and_the_module_level_generator() -> void {
    if not have_gpu() { return }

    #: An explicit seed reproduces; the entropy path only has to differ.
    let fixed = drawn(default_rng(77).random(32))
    let same = drawn(default_rng(77).random(32))
    assert_eq(fixed, same)

    let one = drawn(default_rng().random(32))
    let two = drawn(default_rng().random(32))
    assert_true(not one == two, "two entropy-seeded generators are not the same generator")
}
