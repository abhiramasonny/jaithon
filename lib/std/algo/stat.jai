#: Descriptive statistics, correlation, regression, and the four distributions
#: that come up often enough to belong in the standard library.
#:
#: `variance`/`stdev` are the **sample** estimators (divide by `n - 1`); the
#: `population_*` forms divide by `n` and are correct only when the list *is*
#: the whole population — picking the wrong one is the most common statistical
#: bug in a program. A NaN anywhere in the input makes every result NaN.
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
#: The naive formula `(sum_squares - sum*sum/n) / (n - 1)` suffers catastrophic
#: cancellation for large values; Welford's recurrence tracks deviations from
#: the running mean instead, which stays accurate regardless of scale.
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
        self.squared_deviations += delta * (value - self.running_mean)
        if self.observations == 1 {
            self.lowest = value
            self.highest = value
        } else {
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
    #: Chan-Golub-LeVeque's pairwise combination — stable for the same reason as
    #: the running update: the correction term is a difference of means, not
    #: sums, so parallel chunks can be merged safely.
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
#: ```
#: median([3.0, 1.0, 2.0])         # 2.0
#: median([1.0, 2.0, 3.0, 4.0])    # 2.5
#: ```
pub fn median(values: list[float]) -> float { return quantile(values, 0.5) }

#: Return the most frequent value, the smallest of them if several tie.
#:
#: ```
#: mode([1.0, 2.0, 2.0, 3.0])    # 2.0
#: ```
pub fn mode(values: list[float]) -> float { return multimode(values)[0] }

#: Return every value tied for most frequent, in ascending order.
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
#: Linear interpolation at position `q * (n - 1)` — R's type 7, the default of
#: NumPy, pandas and most spreadsheets, but not the only convention: matching
#: another tool's percentile output needs that tool's definition, not this one.
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
#: Deviations are taken from the means before multiplying (the two-pass form)
#: to avoid the cancellation that `sum(x*y) - sum(x)*sum(y)/n` suffers from.
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
#: Measures *linear* association only — points on a parabola through the
#: origin correlate at zero despite being perfectly determined by x.
#:
#: Raises `ValueError` when either series is constant, where the correlation
#: is undefined rather than zero.
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
    return math.clamp(sxy / math.sqrt(sxx * syy), -1.0, 1.0)
}

#: Return the Spearman rank correlation of two paired series, in `[-1, 1]`.
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
#: Sensitive to outliers, which squeeze everything else toward a corner;
#: `standardize` is the robust alternative.
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
#: Unlike `normalize`, has no fixed output range, so it stays comparable
#: across differently sized samples.
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
#: Named `value_range` because `range` is a builtin.
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
#: Built on quartiles rather than mean/stdev because outliers corrupt those
#: two; quartiles do not notice them. A convention, not a statistical test.
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
#: Uses a running total (O(n), not O(n * window)); over a very long series
#: this accumulates a little rounding drift, negligible next to what a moving
#: average is normally used on.
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
#: Seeded with the first value, so the result is as long as the input and its
#: first entry exact; seeding with the mean of the first few values instead
#: would trade that exactness for a shorter warm-up.
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
#: Sums are accumulated from deviations about the means, not raw values, which
#: keeps the slope accurate far from the origin — the textbook
#: `n*sum(xy) - sum(x)*sum(y)` form loses most of its digits there.
#:
#: A high `r_squared` is not evidence the model is right; it says the line
#: fits these points, not that the relationship is linear or causal.
#:
#: A constant y series fits with slope zero and `r_squared` 1.0, since the
#: line reproduces the data exactly.
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
        r_squared = 1.0 - max(syy - slope * sxy, 0.0) / syy
    }
    return LinearFit(slope, mean_y - slope * mean_x, r_squared)
}

#: Fit a polynomial of degree `degree` by least squares.
#:
#: Coefficients are ascending, so `[c0, c1, c2]` means `c0 + c1*x + c2*x*x`;
#: feed the result to `polynomial_value`.
#:
#: Two real hazards: the normal equations square the condition number of an
#: already badly-conditioned Vandermonde matrix, so past degree five or so (or
#: with x values far from zero and clustered) the coefficients can have few
#: correct digits despite a small residual — centre and scale x first, or use
#: QR/orthogonal polynomials instead. And a high-degree fit through few points
#: interpolates noise rather than signal.
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
#: Horner's method: fewer multiplications than summing `c[k] * x**k` directly,
#: and the more accurate order to sum them in.
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
#: The last bin is closed at both ends, so the maximum lands inside it rather
#: than in a bin of its own. A list of all-identical values gets a single bin
#: of width one, centred on the value.
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
#: Uses the interquartile range rather than the standard deviation, which
#: makes it robust to outliers (unlike Scott's rule).
#:
#: Falls back to Sturges' rule when the middle half of the data is a single
#: repeated value (zero bin width); capped at 10000 bins.
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
    let width = 2.0 * middle_half / math.cbrt(float(count))
    if width <= 0.0 {
        return max(1, int(math.ceil(math.log2(float(count)))) + 1)
    }
    return max(1, min(MAX_BINS, int(math.ceil(span / width))))
}

#: Return the empirical distribution function as `(value, probability)` pairs.
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
#: A density, not a probability: it can exceed 1.0 for a narrow distribution;
#: use `normal_cdf` for an actual probability.
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
#: Written with `erfc` rather than `1 + erf`, which cancels to nothing
#: significant deep in the left tail; `erfc` delivers the small probability directly.
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
#: Evaluated via the incomplete gamma function: a power series below the
#: crossover, a continued fraction above it, each used only where it converges
#: quickly, giving about fifteen correct digits across the range.
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
#: Past `x` ~ 2, `erf(x)` is within an ulp of 1.0, so `1.0 - erf(x)` loses
#: every significant digit; this stays accurate out to `erfc(26)` (~1e-296).
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
#: Computed in log space — the binomial coefficient alone overflows a float
#: past a few hundred trials, long before the probabilities bring it back down.
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
#: Computed in log space for the same reason as `binomial_pmf`; the log
#: factorial costs O(occurrences).
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
        let shared = (position + last) / 2.0 + 1.0
        for tied in position..=last { ranks[order[tied]] = shared }
        position = last + 1
    }
    return ranks
}

#: Solve `matrix * solution = rhs` by Gaussian elimination with partial pivoting.
#:
#: Partial pivoting keeps the multipliers below one, so a small pivot can't
#: amplify the rounding of every row beneath it.
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
#: Every term is positive, so there's nothing to cancel; converges quickly
#: below `x ~ a + 1`, slowly above it, where `_incomplete_gamma_fraction` takes over.
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
#: Uses the modified Lentz recurrence, which builds the fraction from the
#: front with no need to guess its depth; `TINY` stands in for a zero denominator.
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
