#: Descriptive statistics, correlation, regression, and the four distributions
#: that come up often enough to belong in the standard library.
#:
#: ## Sample or population
#:
#: Every dispersion measure here comes in two forms, and picking the wrong one
#: is the most common statistical bug in a program.
#:
#:   - `variance` and `stdev` are the **sample** estimators. They divide the
#:     sum of squared deviations by `n - 1`.
#:   - `population_variance` and `population_stdev` divide by `n`.
#:
#: Use the population form only when the list in hand *is* the entire
#: population — every pixel of this image, every employee of this company.
#: Whenever the data is a sample drawn from something larger, and you want to
#: say something about that larger thing, use the sample form. That is why it
#: is the unmarked default.
#:
#: The `n - 1` is Bessel's correction, and the reason for it is that the
#: deviations are measured from the sample mean rather than the true mean. The
#: sample mean is by construction the point that *minimises* the sum of squared
#: deviations for this particular sample, so those deviations are systematically
#: too small — on average by exactly the factor `(n - 1) / n`. Dividing by
#: `n - 1` instead of `n` cancels that factor and makes the estimator unbiased:
#: averaged over all samples it lands on the true variance. The correction
#: matters most where samples are small, which is where it is most often
#: forgotten: at `n = 4` it is a 33% adjustment to the variance.
#:
#: Note that an unbiased variance does not give an unbiased standard deviation,
#: because the square root is not a linear function. `stdev` is the square root
#: of the unbiased variance, which is the universal convention and still
#: slightly underestimates. No correction for that is applied here; the exact
#: one depends on the distribution.
#:
#: ## Accuracy and inputs
#:
#: Sums that can cancel are computed in a form that does not: `Summary` uses
#: Welford's recurrence, the means come from the compensated summation in
#: `std.math`, and the regressions centre their data before accumulating.
#:
#: Every function takes `list[float]`. Convert integer data with
#: `[float(n) for n in counts]`. NaN is not filtered out — one NaN in the input
#: makes every result NaN, which is the honest answer for data with a hole in
#: it. Filter first if that is not what you want. Functions that have no answer
#: for their input raise `ValueError` rather than returning NaN.
import std.math
from std.math import mean
from std.core import Printable
from std.algo.sort import sort_by_key

#: `mean` is `std.math.mean`, re-exported: the compensated summation it needs
#: belongs beside the other floating-point primitives, and a statistics module
#: without a `mean` would be absurd.
export { mean }

#: The natural logarithm of the square root of pi, which is `lgamma(0.5)`.
const LN_SQRT_PI: float = 0.5723649429247001

#: The iteration cap for the incomplete gamma series and continued fraction.
const GAMMA_MAX_ITERATIONS: int = 300

#: The relative convergence target for those two loops, near the float limit.
const GAMMA_TOLERANCE: float = 1e-15

#: A number small enough to stand in for zero in the Lentz recurrence.
const TINY: float = 1e-300

#: The cap on automatic bin selection, so that one far outlier cannot ask for
#: millions of empty bins.
const MAX_BINS: int = 10_000

#: A running count, mean and variance, computed in one pass and O(1) memory.
#:
#: The textbook one-pass formula keeps `sum(x)` and `sum(x*x)` and finishes
#: with `(sum_squares - sum * sum / n) / (n - 1)`. It is algebraically correct
#: and numerically indefensible. Take the three values 1e9, 1e9 + 1, 1e9 + 2,
#: whose variance is exactly 1. Their `sum_squares` is about 3e18, and
#: `sum * sum / n` is about 3e18 as well; the answer is the difference between
#: them, which is 2. But a binary64 near 3e18 has a spacing of 512, so each of
#: those two enormous numbers is already rounded by hundreds before they are
#: subtracted. The difference that survives is noise, and every digit of the
#: answer is gone. With slightly larger values the result comes out negative,
#: and a variance cannot be negative.
#:
#: The disease is catastrophic cancellation: the answer is tiny compared to the
#: quantities it is computed from, so the rounding those quantities carry
#: swamps it. Welford's recurrence never forms them. It carries the mean and
#: the sum of squared deviations *from that running mean*, and updates both
#: with one deviation per observation:
#:
#:     delta  = x - mean
#:     mean  += delta / n
#:     m2    += delta * (x - mean)        # the second deviation uses the new mean
#:
#: Every quantity in that loop is on the scale of the *spread* of the data
#: rather than the scale of the data, so there is nothing large to cancel and
#: nothing to lose. It costs one extra division per observation and is exact to
#: within a few ulp regardless of how far from zero the values sit.
#:
#: ```
#: let summary = Summary.of([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
#: summary.count()                                    # 8
#: math.round_to(summary.mean(), 12)                  # 5.0
#: math.round_to(summary.population_variance(), 12)   # 4.0
#: math.round_to(summary.variance(), 9)               # 4.571428571
#: ```
pub class Summary: Printable {
    var observations: int
    var running_mean: float
    var squared_deviations: float
    var lowest: float
    var highest: float

    #: Create an empty summary.
    pub fn init(self) {
        self.observations = 0
        self.running_mean = 0.0
        self.squared_deviations = 0.0
        self.lowest = 0.0
        self.highest = 0.0
    }

    #: Create a summary of `values` in one pass.
    pub static fn of(values: list[float]) -> Summary {
        let summary = Summary()
        for value in values { summary.push(value) }
        return summary
    }

    #: Fold one more observation in. O(1).
    pub fn push(self, value: float) -> void {
        self.observations += 1
        let delta = value - self.running_mean
        self.running_mean += delta / self.observations
        # The second factor is the deviation from the *updated* mean. Their
        # product is exactly the increment the sum of squared deviations needs.
        self.squared_deviations += delta * (value - self.running_mean)
        if self.observations == 1 {
            self.lowest = value
            self.highest = value
        } else {
            # Written out rather than with `min`/`max` so that nothing here
            # depends on how a bare name resolves against this class's own
            # `min` and `max` methods.
            if value < self.lowest { self.lowest = value }
            if value > self.highest { self.highest = value }
        }
    }

    #: Fold in every value of `values`.
    pub fn extend(self, values: list[float]) -> void {
        for value in values { self.push(value) }
    }

    #: Return how many observations have been folded in.
    pub fn count(self) -> int { return self.observations }

    #: Return the arithmetic mean.
    #:
    #: Raises `ValueError` when nothing has been observed.
    pub fn mean(self) -> float {
        if self.observations == 0 { throw ValueError("mean of an empty Summary") }
        return self.running_mean
    }

    #: Return the sample variance, dividing by `n - 1`.
    #:
    #: Raises `ValueError` below two observations, where the estimator is
    #: undefined rather than zero.
    pub fn variance(self) -> float {
        if self.observations < 2 {
            throw ValueError(
                f"sample variance needs at least 2 observations, got {self.observations}"
            )
        }
        return self.squared_deviations / (self.observations - 1)
    }

    #: Return the population variance, dividing by `n`.
    pub fn population_variance(self) -> float {
        if self.observations == 0 { throw ValueError("variance of an empty Summary") }
        return self.squared_deviations / self.observations
    }

    #: Return the sample standard deviation.
    pub fn stdev(self) -> float { return math.sqrt(self.variance()) }

    #: Return the population standard deviation.
    pub fn population_stdev(self) -> float { return math.sqrt(self.population_variance()) }

    #: Return the smallest observation.
    pub fn min(self) -> float {
        if self.observations == 0 { throw ValueError("min of an empty Summary") }
        return self.lowest
    }

    #: Return the largest observation.
    pub fn max(self) -> float {
        if self.observations == 0 { throw ValueError("max of an empty Summary") }
        return self.highest
    }

    #: Return the summary of both sets of observations, without revisiting them.
    #:
    #: Chan, Golub and LeVeque's pairwise combination. It is what lets a large
    #: input be summarised in parallel chunks and the chunks folded together,
    #: and it is as stable as the sequential update: the correction term is
    #: again built from a difference of means rather than of sums.
    #:
    #: ```
    #: let left = Summary.of([1.0, 2.0, 3.0])
    #: let right = Summary.of([4.0, 5.0])
    #: left.merge(right).mean()    # 3.0
    #: ```
    pub fn merge(self, other: Summary) -> Summary {
        let combined = Summary()
        let total = self.observations + other.observations
        if total == 0 { return combined }
        let delta = other.running_mean - self.running_mean
        combined.observations = total
        combined.running_mean = self.running_mean + delta * other.observations / total
        combined.squared_deviations = self.squared_deviations +
            other.squared_deviations +
            delta * delta * self.observations * other.observations / total
        if self.observations == 0 {
            combined.lowest = other.lowest
            combined.highest = other.highest
        } elif other.observations == 0 {
            combined.lowest = self.lowest
            combined.highest = self.highest
        } else {
            combined.lowest = self.lowest
            combined.highest = self.highest
            if other.lowest < self.lowest { combined.lowest = other.lowest }
            if other.highest > self.highest { combined.highest = other.highest }
        }
        return combined
    }

    #: Return the summary rendered as `Summary(n=..., mean=..., variance=...)`.
    pub fn to_str(self) -> str {
        if self.observations < 2 { return f"Summary(n={self.observations})" }
        return f"Summary(n={self.observations}, mean={self.running_mean}, " +
            f"variance={self.variance()})"
    }

    fn __len__(self) -> int { return self.observations }

    fn __str__(self) -> str { return self.to_str() }
}

#: A fitted straight line, with the fraction of variance it explains.
pub class LinearFit: Printable {
    #: The change in y per unit of x.
    pub let slope: float

    #: The value of the line at `x == 0`.
    pub let intercept: float

    #: The coefficient of determination, in `[0, 1]`.
    pub let r_squared: float

    #: Create the fit `y = intercept + slope * x`.
    pub fn init(self, slope: float, intercept: float, r_squared: float) {
        self.slope = slope
        self.intercept = intercept
        self.r_squared = r_squared
    }

    #: Return the value the line predicts at `x`.
    pub fn predict(self, x: float) -> float { return self.intercept + self.slope * x }

    #: Return the fit rendered as `LinearFit(slope=..., intercept=..., r2=...)`.
    pub fn to_str(self) -> str {
        return f"LinearFit(slope={self.slope}, intercept={self.intercept}, " +
            f"r2={self.r_squared})"
    }

    fn __str__(self) -> str { return self.to_str() }
}

#: Return the middle value, averaging the two middle ones for an even count.
#:
#: Exactly the 0.5 quantile of `quantile`, and therefore interpolated the same
#: way. Unlike the mean it is unmoved by an extreme value, which is the whole
#: reason to reach for it.
#:
#: ```
#: median([3.0, 1.0, 2.0])         # 2.0
#: median([1.0, 2.0, 3.0, 4.0])    # 2.5
#: ```
pub fn median(values: list[float]) -> float { return quantile(values, 0.5) }

#: Return the most frequent value, the smallest of them if several tie.
#:
#: Raises `ValueError` for an empty list. Use `multimode` when the ties matter.
#:
#: ```
#: mode([1.0, 2.0, 2.0, 3.0])    # 2.0
#: ```
pub fn mode(values: list[float]) -> float { return multimode(values)[0] }

#: Return every value tied for most frequent, in ascending order.
#:
#: A single-element result means the data has one clear mode; a result as long
#: as the distinct value count means every value occurs equally often and the
#: mode says nothing.
#:
#: ```
#: multimode([1.0, 1.0, 2.0, 2.0, 3.0])    # [1.0, 2.0]
#: ```
pub fn multimode(values: list[float]) -> list[float] {
    if values.len() == 0 { throw ValueError("mode of an empty list is undefined") }
    let counts: dict[float, int] = {}
    for value in values { counts[value] = counts.get(value, 0) + 1 }
    var best = 0
    for count in counts.values() { best = max(best, count) }
    let winners: list[float] = []
    for (value, count) in counts.items() {
        if count == best { winners.push(value) }
    }
    return sorted(winners)
}

#: Return the value below which a fraction `q` of the data falls.
#:
#: The linear interpolation definition, stated exactly: sort the data, take the
#: position `p = q * (n - 1)`, and if `p` falls between two order statistics
#: return the point that fraction of the way between them. So `q = 0.0` is the
#: minimum, `q = 1.0` is the maximum, and `q = 0.5` is the median.
#:
#: This is R's type 7 and the default of NumPy, pandas and most spreadsheets.
#: It is not the only definition — R alone offers nine, and the "nearest rank"
#: convention used in much of the finance and SLA literature returns an actual
#: data point instead of an interpolation. They agree once the data is large
#: and disagree visibly when it is small, so a percentile that must match
#: another tool's output needs that tool's definition, not this one.
#:
#: Raises `ValueError` for an empty list or a `q` outside `[0, 1]`.
#:
#: ```
#: quantile([1.0, 2.0, 3.0, 4.0], 0.25)    # 1.75
#: quantile([1.0, 2.0, 3.0, 4.0], 1.0)     # 4.0
#: ```
pub fn quantile(values: list[float], q: float) -> float {
    if values.len() == 0 { throw ValueError("quantile of an empty list is undefined") }
    if q < 0.0 or q > 1.0 { throw ValueError(f"q must be in [0, 1], got {q}") }
    return _quantile_of_sorted(sorted(values), q)
}

#: Return the sample variance, dividing by `n - 1`.
#:
#: Computed with Welford's recurrence — see `Summary` for why the textbook
#: formula is not used. Raises `ValueError` below two values.
#:
#: ```
#: math.round_to(variance([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]), 9)    # 4.571428571
#: ```
pub fn variance(values: list[float]) -> float { return Summary.of(values).variance() }

#: Return the population variance, dividing by `n`.
#:
#: Correct only when `values` is the whole population; see the module doc.
#:
#: ```
#: let spread = population_variance([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
#: math.round_to(spread, 12)    # 4.0
#: ```
pub fn population_variance(values: list[float]) -> float {
    return Summary.of(values).population_variance()
}

#: Return the sample standard deviation, the square root of `variance`.
#:
#: In the units of the data, unlike the variance, which is in their square.
#:
#: ```
#: math.round_to(stdev([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]), 9)    # 2.138089935
#: ```
pub fn stdev(values: list[float]) -> float { return Summary.of(values).stdev() }

#: Return the population standard deviation, the root of `population_variance`.
#:
#: ```
#: math.round_to(population_stdev([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]), 12)    # 2.0
#: ```
pub fn population_stdev(values: list[float]) -> float {
    return Summary.of(values).population_stdev()
}

#: Return the sample covariance of two paired series, dividing by `n - 1`.
#:
#: Positive when the two rise together, negative when one rises as the other
#: falls, zero when they are linearly unrelated. Its units are the product of
#: the two series' units, which makes the magnitude hard to read — that is what
#: `correlation` fixes.
#:
#: Both deviations are taken from the respective means before multiplying,
#: which is the two-pass form: it costs a second traversal and avoids the
#: cancellation that `sum(x*y) - sum(x)*sum(y)/n` suffers from.
#:
#: Raises `ValueError` for lists of different lengths or fewer than two points.
#:
#: ```
#: covariance([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])    # 2.0
#: ```
pub fn covariance(xs: list[float], ys: list[float]) -> float {
    _require_paired(xs, ys, "covariance")
    if xs.len() < 2 {
        throw ValueError(f"covariance needs at least 2 points, got {xs.len()}")
    }
    let mean_x = mean(xs)
    let mean_y = mean(ys)
    var total = 0.0
    for (index, x) in xs.enumerate() {
        total += (x - mean_x) * (ys[index] - mean_y)
    }
    return total / (xs.len() - 1)
}

#: Return the Pearson correlation of two paired series, in `[-1, 1]`.
#:
#: The covariance divided by both standard deviations, which cancels the units
#: and the scale: 1 means the points lie exactly on a rising line, -1 on a
#: falling one, 0 that no straight line describes them. It measures *linear*
#: association only — the points on a parabola through the origin correlate at
#: zero while being perfectly determined.
#:
#: Raises `ValueError` when either series is constant, which leaves the
#: correlation undefined rather than zero.
#:
#: ```
#: correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])     # 1.0
#: correlation([1.0, 2.0, 3.0], [6.0, 4.0, 2.0])     # -1.0
#: ```
pub fn correlation(xs: list[float], ys: list[float]) -> float {
    _require_paired(xs, ys, "correlation")
    if xs.len() < 2 {
        throw ValueError(f"correlation needs at least 2 points, got {xs.len()}")
    }
    let mean_x = mean(xs)
    let mean_y = mean(ys)
    var sxx = 0.0
    var syy = 0.0
    var sxy = 0.0
    for (index, x) in xs.enumerate() {
        let dx = x - mean_x
        let dy = ys[index] - mean_y
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
    }
    if sxx == 0.0 or syy == 0.0 {
        throw ValueError("correlation is undefined when either series is constant")
    }
    # Rounding can push a perfect correlation a hair past 1.0, and a
    # correlation outside [-1, 1] breaks everything downstream.
    return math.clamp(sxy / math.sqrt(sxx * syy), -1.0, 1.0)
}

#: Return the Spearman rank correlation of two paired series, in `[-1, 1]`.
#:
#: The Pearson correlation of the *ranks* rather than the values. That makes it
#: a measure of monotone association: it reaches 1 whenever y always rises with
#: x, straight line or not, where Pearson would report something less. Because
#: only order survives the ranking it is also unmoved by a single wild outlier,
#: which can dominate a Pearson correlation on its own.
#:
#: Tied values share the average of the ranks they span, the standard
#: correction.
#:
#: ```
#: spearman([1.0, 2.0, 3.0, 4.0], [1.0, 4.0, 9.0, 16.0])    # 1.0
#: ```
pub fn spearman(xs: list[float], ys: list[float]) -> float {
    _require_paired(xs, ys, "spearman")
    return correlation(_ranks(xs), _ranks(ys))
}

#: Return how many sample standard deviations `value` sits from the mean.
#:
#: The unit of "unusual": a z-score of 2 means the value is two standard
#: deviations above the mean, which for normally distributed data puts it
#: outside about 95% of the population.
#:
#: Raises `ValueError` for a constant sample, where every value is zero
#: deviations from the mean and the score is 0/0.
#:
#: ```
#: let sample = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
#: math.round_to(z_score(7.0, sample), 6)    # 0.935414
#: ```
pub fn z_score(value: float, values: list[float]) -> float {
    let summary = Summary.of(values)
    let deviation = summary.stdev()
    if deviation == 0.0 {
        throw ValueError(
            f"z_score is undefined for a constant sample; every value is {summary.mean()}"
        )
    }
    return (value - summary.mean()) / deviation
}

#: Return the values rescaled onto `[0, 1]` by their own range.
#:
#: Min-max scaling. Use it to put unrelated features on one axis for a plot or
#: a distance metric. It keeps the shape of the distribution and is entirely at
#: the mercy of the extremes: one outlier squeezes everything else into a
#: corner. `standardize` is the robust alternative.
#:
#: Raises `ValueError` for a constant list, which has no range to divide by.
#:
#: ```
#: normalize([10.0, 20.0, 30.0])    # [0.0, 0.5, 1.0]
#: ```
pub fn normalize(values: list[float]) -> list[float] {
    let (lowest, highest) = min_max(values)
    if lowest == highest {
        throw ValueError(f"cannot normalize a constant list; every value is {lowest}")
    }
    let span = highest - lowest
    return [(value - lowest) / span for value in values]
}

#: Return the values as z-scores: mean 0, sample standard deviation 1.
#:
#: The other rescaling. Unlike `normalize` it has no fixed output range, so a
#: far-out value stays far out instead of compressing everything else, and the
#: result is comparable across differently sized samples.
#:
#: Raises `ValueError` for a constant list.
#:
#: ```
#: standardize([1.0, 2.0, 3.0])    # [-1.0, 0.0, 1.0]
#: ```
pub fn standardize(values: list[float]) -> list[float] {
    let summary = Summary.of(values)
    let deviation = summary.stdev()
    if deviation == 0.0 {
        throw ValueError("cannot standardize a constant list; its deviation is zero")
    }
    let center = summary.mean()
    return [(value - center) / deviation for value in values]
}

#: Return the smallest and largest values, in one pass.
#:
#: Raises `ValueError` for an empty list.
#:
#: ```
#: min_max([3.0, 1.0, 4.0, 1.0, 5.0])    # (1.0, 5.0)
#: ```
pub fn min_max(values: list[float]) -> tuple[float, float] {
    if values.len() == 0 { throw ValueError("min_max of an empty list") }
    var lowest = values[0]
    var highest = values[0]
    for value in values {
        lowest = min(lowest, value)
        highest = max(highest, value)
    }
    return (lowest, highest)
}

#: Return the statistical range: the largest value minus the smallest.
#:
#: Named `value_range` because `range` is a builtin. The crudest measure of
#: spread there is — it uses two of the n values and ignores the rest — but the
#: right one when what matters is the extremes, as for an axis or a buffer.
#:
#: ```
#: value_range([3.0, 1.0, 4.0, 1.0, 5.0])    # 4.0
#: ```
pub fn value_range(values: list[float]) -> float {
    let (lowest, highest) = min_max(values)
    return highest - lowest
}

#: Return the interquartile range, the width of the middle half of the data.
#:
#: The 0.75 quantile minus the 0.25 quantile, both by the definition in
#: `quantile`. Unlike `value_range` and `stdev` it is a robust measure of
#: spread: up to a quarter of the data at each end can be arbitrarily wrong
#: without moving it.
#:
#: ```
#: iqr([1.0, 2.0, 3.0, 4.0, 5.0])    # 2.0
#: ```
pub fn iqr(values: list[float]) -> float {
    if values.len() == 0 { throw ValueError("iqr of an empty list") }
    let ordered = sorted(values)
    return _quantile_of_sorted(ordered, 0.75) - _quantile_of_sorted(ordered, 0.25)
}

#: Return the fences of the IQR rule, as `(low, high)`.
#:
#: `factor` multiplies the interquartile range: 1.5 gives Tukey's ordinary
#: outlier fences, 3.0 the "far out" fences. Anything beyond the returned pair
#: is an outlier by the rule.
#:
#: ```
#: outlier_bounds([1.0, 2.0, 3.0, 4.0, 5.0])    # (-1.0, 7.0)
#: ```
pub fn outlier_bounds(values: list[float], factor: float = 1.5) -> tuple[float, float] {
    if values.len() == 0 { throw ValueError("outlier_bounds of an empty list") }
    if factor < 0.0 { throw ValueError(f"factor must be non-negative, got {factor}") }
    let ordered = sorted(values)
    let first_quartile = _quantile_of_sorted(ordered, 0.25)
    let third_quartile = _quantile_of_sorted(ordered, 0.75)
    let width = factor * (third_quartile - first_quartile)
    return (first_quartile - width, third_quartile + width)
}

#: Return the values that fall outside the IQR fences, in their original order.
#:
#: Tukey's rule. It is built on quartiles rather than on the mean and standard
#: deviation precisely because outliers corrupt those two: a single absurd
#: value inflates the standard deviation enough to hide itself inside three
#: sigma. The quartiles do not notice it at all.
#:
#: The rule is a convention, not a test. It flags roughly 0.7% of a normal
#: sample by construction, and on a strongly skewed distribution it will flag a
#: long tail that is perfectly ordinary for that distribution.
#:
#: ```
#: outliers([1.0, 2.0, 3.0, 4.0, 100.0])    # [100.0]
#: ```
pub fn outliers(values: list[float], factor: float = 1.5) -> list[float] {
    let (low, high) = outlier_bounds(values, factor)
    return [value for value in values if value < low or value > high]
}

#: Return the mean of every consecutive window of `window` values.
#:
#: The result has `n - window + 1` entries: one per full window, so nothing is
#: invented at the ends. Computed with a running total in O(n) rather than
#: O(n * window); over a very long series that total accumulates a little
#: rounding drift, which is the price of the linear algorithm and is far below
#: the noise in anything a moving average is used on.
#:
#: Raises `ValueError` when `window` is below 1 or longer than the data.
#:
#: ```
#: moving_average([1.0, 2.0, 3.0, 4.0], 2)    # [1.5, 2.5, 3.5]
#: ```
pub fn moving_average(values: list[float], window: int) -> list[float] {
    if window < 1 { throw ValueError(f"window must be at least 1, got {window}") }
    if window > values.len() {
        throw ValueError(f"window {window} exceeds the {values.len()} values available")
    }
    var total = 0.0
    for index in 0..window { total += values[index] }
    let averages = [total / window]
    for index in window..values.len() {
        total += values[index] - values[index - window]
        averages.push(total / window)
    }
    return averages
}

#: Return the exponentially weighted moving average of `values`.
#:
#: Each output is `alpha * value + (1 - alpha) * previous_output`, so the
#: weight of an observation decays geometrically with its age and nothing is
#: ever fully forgotten. A larger `alpha` follows the data more closely; a
#: smaller one smooths harder. Traders quote it as a span: `alpha = 2 / (span + 1)`
#: makes the average behave roughly like a `span`-length simple one.
#:
#: The series is seeded with its first value, so the result is as long as the
#: input and its first entry is exact. Seeding instead with the mean of the
#: first few values trades that exactness for a shorter warm-up.
#:
#: Raises `ValueError` for an empty list or an `alpha` outside `(0, 1]`.
#:
#: ```
#: exponential_moving_average([1.0, 2.0, 3.0], 0.5)    # [1.0, 1.5, 2.25]
#: ```
pub fn exponential_moving_average(values: list[float], alpha: float) -> list[float] {
    if values.len() == 0 { throw ValueError("exponential_moving_average of an empty list") }
    if alpha <= 0.0 or alpha > 1.0 {
        throw ValueError(f"alpha must be in (0, 1], got {alpha}")
    }
    var current = values[0]
    let smoothed = [current]
    for index in 1..values.len() {
        current = alpha * values[index] + (1.0 - alpha) * current
        smoothed.push(current)
    }
    return smoothed
}

#: Fit `y = intercept + slope * x` by least squares.
#:
#: The line that minimises the sum of squared vertical residuals, with
#: `r_squared` reporting the fraction of the variance in y that it accounts
#: for: 1.0 for points exactly on a line, 0.0 for a fit no better than the mean
#: of y. Both sums are accumulated from deviations about the means rather than
#: from the raw values, which is what keeps the slope accurate when the data
#: sits far from the origin — the textbook `n*sum(xy) - sum(x)*sum(y)` form
#: loses most of its digits there.
#:
#: A high `r_squared` is not evidence that the model is right; it says the line
#: fits these points, not that the relationship is linear or causal.
#:
#: Raises `ValueError` for mismatched lengths, fewer than two points, or an x
#: series with no spread, which fixes no slope. A constant y series is fitted
#: with slope zero and reported as `r_squared` 1.0, since the line reproduces
#: the data exactly.
#:
#: ```
#: let fit = linear_regression([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
#: fit.slope         # 2.0
#: fit.intercept     # 0.0
#: fit.r_squared     # 1.0
#: fit.predict(4.0)  # 8.0
#: ```
pub fn linear_regression(xs: list[float], ys: list[float]) -> LinearFit {
    _require_paired(xs, ys, "linear_regression")
    if xs.len() < 2 {
        throw ValueError(f"linear_regression needs at least 2 points, got {xs.len()}")
    }
    let mean_x = mean(xs)
    let mean_y = mean(ys)
    var sxx = 0.0
    var sxy = 0.0
    var syy = 0.0
    for (index, x) in xs.enumerate() {
        let dx = x - mean_x
        let dy = ys[index] - mean_y
        sxx += dx * dx
        sxy += dx * dy
        syy += dy * dy
    }
    if sxx == 0.0 {
        throw ValueError(f"linear_regression needs two distinct x values; every x is {mean_x}")
    }
    let slope = sxy / sxx
    var r_squared = 1.0
    if syy > 0.0 {
        # Residual sum of squares, which algebra reduces to syy - slope * sxy.
        # Rounding can make that a hair negative for a perfect fit.
        r_squared = 1.0 - max(syy - slope * sxy, 0.0) / syy
    }
    return LinearFit(slope, mean_y - slope * mean_x, r_squared)
}

#: Fit a polynomial of degree `degree` by least squares.
#:
#: Returns the coefficients in ascending order, so the result `[c0, c1, c2]`
#: means `c0 + c1*x + c2*x*x`; feed it to `polynomial_value`. The normal
#: equations are formed from the power sums and solved by Gaussian elimination
#: with partial pivoting.
#:
#: Two warnings, both real. First, the normal equations square the condition
#: number of the underlying Vandermonde matrix, and that matrix is already
#: badly conditioned: past degree five or so, or with x values far from zero
#: and clustered together, the coefficients returned will have few correct
#: digits even though the residual looks small. Centre and scale x — subtract
#: the mean, divide by the spread — before fitting anything but a low degree,
#: or use a QR factorisation or orthogonal polynomials, which do not square
#: anything. Second, a high-degree fit through few points interpolates noise;
#: the fit passing exactly through every point is usually the useless one.
#:
#: Raises `ValueError` for mismatched lengths, a negative degree, fewer points
#: than coefficients, or a system that turns out singular because the x values
#: take too few distinct values.
#:
#: ```
#: let coefficients = polynomial_regression([0.0, 1.0, 2.0], [0.0, 1.0, 4.0], 2)
#: coefficients.len()                          # 3
#: math.round_to(coefficients[2], 9)           # 1.0
#: ```
pub fn polynomial_regression(xs: list[float], ys: list[float], degree: int) -> list[float] {
    _require_paired(xs, ys, "polynomial_regression")
    if degree < 0 { throw ValueError(f"degree must be non-negative, got {degree}") }
    let size = degree + 1
    if xs.len() < size {
        throw ValueError(f"a degree {degree} fit needs at least {size} points, got {xs.len()}")
    }
    let power_sums = [0.0 for _ in 0..=2 * degree]
    let targets = [0.0 for _ in 0..size]
    for (index, x) in xs.enumerate() {
        var power = 1.0
        for exponent in 0..=2 * degree {
            power_sums[exponent] += power
            if exponent < size { targets[exponent] += ys[index] * power }
            power *= x
        }
    }
    let matrix = [[power_sums[row + column] for column in 0..size] for row in 0..size]
    return _solve_linear_system(matrix, targets)
}

#: Evaluate a polynomial given by ascending coefficients at `x`.
#:
#: Horner's method: `c0 + x*(c1 + x*(c2 + ...))`. It uses n multiplications
#: instead of the 2n of summing `c[k] * x**k`, and it is the more accurate
#: order to sum them in.
#:
#: ```
#: polynomial_value([1.0, 0.0, 2.0], 3.0)    # 19.0
#: ```
pub fn polynomial_value(coefficients: list[float], x: float) -> float {
    if coefficients.len() == 0 {
        throw ValueError("a polynomial needs at least one coefficient")
    }
    var total = coefficients[coefficients.len() - 1]
    var index = coefficients.len() - 2
    while index >= 0 {
        total = total * x + coefficients[index]
        index -= 1
    }
    return total
}

#: Return the bin edges and the count in each bin.
#:
#: The edges are evenly spaced from the smallest value to the largest, and
#: there is one more edge than there are counts. The last bin is closed at both
#: ends so that the maximum lands inside it rather than in a bin of its own.
#: Passing `bins` as null selects the count with `freedman_diaconis_bins`.
#:
#: A list whose values are all identical has no width to divide; it is given a
#: single bin of width one centred on the value.
#:
#: ```
#: let (edges, counts) = histogram([1.0, 2.0, 3.0, 4.0], 2)
#: edges     # [1.0, 2.5, 4.0]
#: counts    # [2, 2]
#: ```
pub fn histogram(values: list[float], bins: int? = null) -> tuple[list[float], list[int]] {
    if values.len() == 0 { throw ValueError("histogram of an empty list") }
    let count = bins ?? freedman_diaconis_bins(values)
    if count < 1 { throw ValueError(f"bins must be at least 1, got {count}") }
    let (lowest, highest) = min_max(values)
    var start = lowest
    var stop = highest
    if start == stop {
        start = lowest - 0.5
        stop = highest + 0.5
    }
    let width = (stop - start) / count
    let edges = [start + width * index for index in 0..=count]
    let counts = [0 for _ in 0..count]
    for value in values {
        var index = int(math.floor((value - start) / width))
        if index < 0 { index = 0 }
        if index >= count { index = count - 1 }
        counts[index] += 1
    }
    return (edges, counts)
}

#: Return a bin count for `values` by the Freedman-Diaconis rule.
#:
#: The rule sets the bin *width* to `2 * IQR / n**(1/3)` and the count follows
#: from the data's range. Using the interquartile range rather than the
#: standard deviation is what makes it robust: a single outlier widens the
#: range the histogram must cover but does not widen the bins, whereas Scott's
#: rule would let it blur the whole picture. The cube root is the term that
#: narrows the bins as more data arrives, at the rate that minimises the
#: expected error of the histogram as a density estimate.
#:
#: When the middle half of the data is a single repeated value the rule gives a
#: width of zero and Sturges' rule, which looks only at the count, is used
#: instead. The result is capped at 10000 bins.
#:
#: ```
#: freedman_diaconis_bins([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])    # 2
#: ```
pub fn freedman_diaconis_bins(values: list[float]) -> int {
    let count = values.len()
    if count == 0 { throw ValueError("freedman_diaconis_bins of an empty list") }
    let ordered = sorted(values)
    let span = ordered[count - 1] - ordered[0]
    if span <= 0.0 { return 1 }
    let middle_half = _quantile_of_sorted(ordered, 0.75) - _quantile_of_sorted(ordered, 0.25)
    # `cbrt`, not `x ** (1.0 / 3.0)`: one third is not representable, so the
    # power form returns 1.9999999999999998 for the cube root of eight and the
    # bin count comes out several times too large.
    let width = 2.0 * middle_half / math.cbrt(float(count))
    if width <= 0.0 {
        return max(1, int(math.ceil(math.log2(float(count)))) + 1)
    }
    return max(1, min(MAX_BINS, int(math.ceil(span / width))))
}

#: Return the empirical distribution function as `(value, probability)` pairs.
#:
#: One pair per distinct value, ascending, whose probability is the fraction of
#: the data less than or equal to that value. The last probability is always
#: 1.0. This is the step function that a Kolmogorov-Smirnov test compares
#: against a theoretical CDF, and what a quantile plot is drawn from.
#:
#: ```
#: cumulative_distribution([1.0, 2.0, 2.0, 3.0])
#: # [(1.0, 0.25), (2.0, 0.75), (3.0, 1.0)]
#: ```
pub fn cumulative_distribution(values: list[float]) -> list[tuple[float, float]] {
    if values.len() == 0 { throw ValueError("cumulative_distribution of an empty list") }
    let ordered = sorted(values)
    let total = float(ordered.len())
    let steps: list[tuple[float, float]] = []
    var index = 0
    while index < ordered.len() {
        var last = index
        while last + 1 < ordered.len() and ordered[last + 1] == ordered[index] { last += 1 }
        steps.push((ordered[index], float(last + 1) / total))
        index = last + 1
    }
    return steps
}

#: Return the density of the normal distribution at `x`.
#:
#: A density, not a probability: it can exceed 1.0 for a narrow distribution,
#: and only its integral over an interval is a probability. Use `normal_cdf`
#: for that.
#:
#: Raises `ValueError` for a non-positive standard deviation.
#:
#: ```
#: math.round_to(normal_pdf(0.0), 6)    # 0.398942
#: ```
pub fn normal_pdf(x: float, mean_value: float = 0.0, standard_deviation: float = 1.0) -> float {
    if standard_deviation <= 0.0 {
        throw ValueError(f"standard deviation must be positive, got {standard_deviation}")
    }
    let z = (x - mean_value) / standard_deviation
    return math.exp(-0.5 * z * z) / (standard_deviation * math.sqrt(math.TAU))
}

#: Return the probability that a normal variable falls at or below `x`.
#:
#: Written as `0.5 * erfc(-z / sqrt(2))` rather than `0.5 * (1 + erf(z / sqrt(2)))`.
#: The two are algebraically the same and numerically are not: deep in the left
#: tail the second form computes `1 + (something within an ulp of -1)` and
#: returns a result with no significant digits left, while `erfc` delivers the
#: small probability directly.
#:
#: ```
#: normal_cdf(0.0)                        # 0.5
#: math.round_to(normal_cdf(1.96), 4)     # 0.975
#: ```
pub fn normal_cdf(x: float, mean_value: float = 0.0, standard_deviation: float = 1.0) -> float {
    if standard_deviation <= 0.0 {
        throw ValueError(f"standard deviation must be positive, got {standard_deviation}")
    }
    let z = (x - mean_value) / standard_deviation
    return 0.5 * erfc(-z / math.SQRT_2)
}

#: Return the error function of `x`.
#:
#: `erf(x)` is the probability that a standard normal variable lands within
#: `x * sqrt(2)` of its mean, and it is the function every normal probability
#: is expressed through. Evaluated as the regularised incomplete gamma function
#: `P(1/2, x*x)`: a power series below the crossover and a continued fraction
#: above it, each used only where it converges quickly, which gives about
#: fifteen correct digits across the whole range.
#:
#: ```
#: erf(0.0)                                 # 0.0
#: math.round_to(erf(1.0), 9)               # 0.842700793
#: ```
pub fn erf(x: float) -> float {
    if x == 0.0 { return 0.0 }
    if x < 0.0 { return -erf(-x) }
    let squared = x * x
    if squared < 1.5 { return _incomplete_gamma_series(squared) }
    return 1.0 - _incomplete_gamma_fraction(squared)
}

#: Return `1 - erf(x)`, without the cancellation that subtraction would cause.
#:
#: For `x` above about 2 the error function is within an ulp of 1.0, so
#: `1.0 - erf(x)` throws away every significant digit of the answer. This
#: computes the complement directly and stays accurate out to `erfc(26)`, which
#: is around 1e-296.
#:
#: ```
#: erfc(0.0)                                # 1.0
#: math.round_to(erfc(2.0), 9)              # 0.004677735
#: ```
pub fn erfc(x: float) -> float {
    if x == 0.0 { return 1.0 }
    if x < 0.0 { return 2.0 - erfc(-x) }
    let squared = x * x
    if squared < 1.5 { return 1.0 - _incomplete_gamma_series(squared) }
    return _incomplete_gamma_fraction(squared)
}

#: Return the probability of exactly `successes` in `trials` independent tries.
#:
#: Each try succeeds with probability `probability`. Computed in log space —
#: the binomial coefficient alone overflows a float past a few hundred trials,
#: long before the product with the probabilities brings it back into range.
#:
#: Raises `ValueError` for negative trials, a `successes` outside
#: `0..=trials`, or a probability outside `[0, 1]`.
#:
#: ```
#: math.round_to(binomial_pmf(2, 4, 0.5), 9)        # 0.375
#: math.round_to(binomial_pmf(50, 100, 0.5), 6)     # 0.079589
#: ```
pub fn binomial_pmf(successes: int, trials: int, probability: float) -> float {
    if trials < 0 { throw ValueError(f"trials must be non-negative, got {trials}") }
    if successes < 0 or successes > trials {
        throw ValueError(f"successes must be in 0..={trials}, got {successes}")
    }
    if probability < 0.0 or probability > 1.0 {
        throw ValueError(f"probability must be in [0, 1], got {probability}")
    }
    if probability == 0.0 {
        if successes == 0 { return 1.0 }
        return 0.0
    }
    if probability == 1.0 {
        if successes == trials { return 1.0 }
        return 0.0
    }
    let log_probability = _log_binomial(trials, successes) +
        successes * math.ln(probability) +
        (trials - successes) * math.ln(1.0 - probability)
    return math.exp(log_probability)
}

#: Return the probability of exactly `occurrences` events in one interval.
#:
#: `rate` is the mean number of events per interval. The Poisson distribution
#: is the limit of the binomial as the number of tries grows and the chance of
#: each shrinks with their product held at `rate`, which is what makes it the
#: distribution of rare independent events: arrivals at a queue, decays, typos.
#:
#: Computed in log space for the same reason as `binomial_pmf`; the log
#: factorial costs O(occurrences).
#:
#: Raises `ValueError` for a negative count or rate.
#:
#: ```
#: math.approx_eq(poisson_pmf(0, 1.0), math.exp(-1.0))    # true
#: math.round_to(poisson_pmf(3, 2.0), 9)                  # 0.180447044
#: ```
pub fn poisson_pmf(occurrences: int, rate: float) -> float {
    if occurrences < 0 {
        throw ValueError(f"occurrences must be non-negative, got {occurrences}")
    }
    if rate < 0.0 { throw ValueError(f"rate must be non-negative, got {rate}") }
    if rate == 0.0 {
        if occurrences == 0 { return 1.0 }
        return 0.0
    }
    let log_probability = -rate + occurrences * math.ln(rate) - _log_factorial(occurrences)
    return math.exp(log_probability)
}

# --- Private helpers ---------------------------------------------------------
fn _require_paired(xs: list[float], ys: list[float], function_name: str) -> void {
    if xs.len() != ys.len() {
        throw ValueError(f"{function_name} needs equal lengths, got {xs.len()} and {ys.len()}")
    }
    if xs.len() == 0 { throw ValueError(f"{function_name} of empty lists is undefined") }
}

#: Return the `q` quantile of an already sorted list, by linear interpolation.
fn _quantile_of_sorted(ordered: list[float], q: float) -> float {
    let count = ordered.len()
    if count == 1 { return ordered[0] }
    let position = q * (count - 1)
    let lower = int(math.floor(position))
    let fraction = position - lower
    if fraction == 0.0 { return ordered[lower] }
    return math.lerp(ordered[lower], ordered[min(lower + 1, count - 1)], fraction)
}

#: Return the rank of each value, ties sharing the average of their ranks.
fn _ranks(values: list[float]) -> list[float] {
    let count = values.len()
    let order = sort_by_key([index for index in 0..count], |index| values[index])
    let ranks = [0.0 for _ in 0..count]
    var position = 0
    while position < count {
        var last = position
        while last + 1 < count and values[order[last + 1]] == values[order[position]] {
            last += 1
        }
        # Ranks are 1-based, so the average of positions `position..=last` is
        # their midpoint plus one.
        let shared = (position + last) / 2.0 + 1.0
        for tied in position..=last { ranks[order[tied]] = shared }
        position = last + 1
    }
    return ranks
}

#: Solve `matrix * solution = rhs` by Gaussian elimination with partial pivoting.
#:
#: Partial pivoting — taking the largest available pivot at each step — is what
#: keeps the multipliers below one and stops a small pivot from amplifying the
#: rounding of every row beneath it.
fn _solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float] {
    let size = rhs.len()
    let rows = [[matrix[row][column] for column in 0..size] for row in 0..size]
    let right = [rhs[index] for index in 0..size]
    var scale = 0.0
    for row in 0..size {
        for column in 0..size { scale = max(scale, math.fabs(rows[row][column])) }
    }
    if scale == 0.0 { throw ValueError("the system is singular; its matrix is all zeros") }
    for column in 0..size {
        var pivot = column
        for row in column + 1..size {
            if math.fabs(rows[row][column]) > math.fabs(rows[pivot][column]) { pivot = row }
        }
        if math.fabs(rows[pivot][column]) <= scale * 1e-14 {
            throw ValueError("the normal equations are singular; the x values repeat")
        }
        let held_row = rows[column]
        rows[column] = rows[pivot]
        rows[pivot] = held_row
        let held_value = right[column]
        right[column] = right[pivot]
        right[pivot] = held_value
        for row in column + 1..size {
            let factor = rows[row][column] / rows[column][column]
            if factor != 0.0 {
                for k in column..size { rows[row][k] -= factor * rows[column][k] }
                right[row] -= factor * right[column]
            }
        }
    }
    let solution = [0.0 for _ in 0..size]
    var row = size - 1
    while row >= 0 {
        var total = right[row]
        for column in row + 1..size { total -= rows[row][column] * solution[column] }
        solution[row] = total / rows[row][row]
        row -= 1
    }
    return solution
}

#: Return `P(1/2, x)`, the regularised lower incomplete gamma, by its series.
#:
#: Every term is positive, so the sum has nothing to cancel. It converges
#: quickly for `x` below about `a + 1` and slowly above it, which is where
#: `_incomplete_gamma_fraction` takes over.
fn _incomplete_gamma_series(x: float) -> float {
    var term = 2.0
    var total = term
    var denominator = 0.5
    for _step in 0..GAMMA_MAX_ITERATIONS {
        denominator += 1.0
        term *= x / denominator
        total += term
        if math.fabs(term) < math.fabs(total) * GAMMA_TOLERANCE { break }
    }
    return total * math.exp(-x + 0.5 * math.ln(x) - LN_SQRT_PI)
}

#: Return `Q(1/2, x)`, the regularised upper incomplete gamma, by its
#: continued fraction.
#:
#: Evaluated with the modified Lentz recurrence, which builds the fraction from
#: the front and so needs no guess at how deep to go; `TINY` stands in whenever
#: a denominator lands on zero.
fn _incomplete_gamma_fraction(x: float) -> float {
    var b = x + 0.5
    var c = 1.0 / TINY
    var d = 1.0 / b
    var h = d
    for step in 1..=GAMMA_MAX_ITERATIONS {
        let numerator = -step * (step - 0.5)
        b += 2.0
        d = numerator * d + b
        if math.fabs(d) < TINY { d = TINY }
        c = b + numerator / c
        if math.fabs(c) < TINY { c = TINY }
        d = 1.0 / d
        let ratio = d * c
        h *= ratio
        if math.fabs(ratio - 1.0) < GAMMA_TOLERANCE { break }
    }
    return h * math.exp(-x + 0.5 * math.ln(x) - LN_SQRT_PI)
}

#: Return the natural logarithm of the binomial coefficient `C(n, k)`.
#:
#: Accumulated as a sum of logarithms of the multiplicative form, so nothing
#: overflows and the smaller of `k` and `n - k` bounds the work.
fn _log_binomial(n: int, k: int) -> float {
    let smaller = min(k, n - k)
    var total = 0.0
    for step in 1..=smaller {
        total += math.ln(float(n - smaller + step)) - math.ln(float(step))
    }
    return total
}

#: Return the natural logarithm of `n` factorial, in O(n).
fn _log_factorial(n: int) -> float {
    var total = 0.0
    for step in 2..=n { total += math.ln(float(step)) }
    return total
}
