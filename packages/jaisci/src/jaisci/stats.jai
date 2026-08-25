#: Statistics and hypothesis tests under scipy's names — `scipy.stats`.
#:
#: Host f64 on `list[float]`, and everything here is a reduction over a sample
#: plus a distribution lookup: a handful of passes and one special function.
#: `std.algo.stat` already carries the descriptive half (`mean`, `variance`,
#: `correlation`, `erf`, `normal_cdf`) and this builds on it rather than
#: restating it; what it adds is scipy's SHAPES -- named result tuples, a
#: p-value beside every statistic, and the two-sample tests.
#:
#: The normal distribution is `normal` rather than `norm`, which is what
#: `mod.jai` reserves: scipy keeps `scipy.stats.norm` and `scipy.linalg.norm`
#: apart by module, and a flat facade has no room for that.
#:
#: ```
#: import jaisci as sci
#:
#: let fit = sci.linregress(xs, ys)
#: let (t, p) = sci.ttest_ind(before, after)
#: ```
import std.math
from std.core import Printable
from std.algo.stat import erfc, mean, variance

#: Lanczos's approximation, g = 7 and nine coefficients, which is the standard
#: table and gives about fifteen digits over the half plane this needs.
let LANCZOS: list[float] = [
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

#: `log(gamma(x))` for positive `x`.
#:
#: Log space because the tests below need the ratio of two gammas whose
#: arguments differ by half a degree of freedom, and gamma itself overflows a
#: float64 past about 171 -- a sample of two hundred is not unusual.
pub fn lgamma(x: float) -> float {
    if x <= 0.0 { throw ValueError(f"lgamma needs a positive argument, got {x}") }
    if x < 0.5 {
        #: Reflection, so the series is only ever evaluated where it converges.
        return math.log(math.PI / math.fabs(math.sin(math.PI * x))) - lgamma(1.0 - x)
    }
    let z = x - 1.0
    var series = LANCZOS[0]
    for index in 1..LANCZOS.len() {
        series += LANCZOS[index] / (z + float(index))
    }
    let t = z + 7.5
    return 0.5 * math.log(2.0 * math.PI) + (z + 0.5) * math.log(t) - t + math.log(series)
}

#: The regularised incomplete beta `I_x(a, b)`.
#:
#: Lentz's continued fraction, with the standard symmetry swap: the fraction
#: converges quickly only for `x < (a + 1) / (a + b + 2)`, and `I_x(a,b) = 1 -
#: I_{1-x}(b,a)` puts every argument on that side. This is what turns a t
#: statistic into a p-value, and it is the only special function these tests
#: need that `std.algo.stat` does not already have.
pub fn betainc(a: float, b: float, x: float) -> float {
    if x < 0.0 or x > 1.0 { throw ValueError(f"betainc needs 0 <= x <= 1, got {x}") }
    if x == 0.0 { return 0.0 }
    if x == 1.0 { return 1.0 }
    let front = math.exp(
        lgamma(a + b) - lgamma(a) - lgamma(b) + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0) {
        return front * _beta_fraction(a, b, x) / a
    }
    return 1.0 -
        math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + b * math.log(1.0 - x) + a * math.log(x)) * _beta_fraction(b, a, 1.0 - x) / b
}

#: A floor on the modified Lentz denominators, which is what keeps a zero term
#: from dividing.
const TINY = 1e-300

fn _beta_fraction(a: float, b: float, x: float) -> float {
    let qab = a + b
    let qap = a + 1.0
    let qam = a - 1.0
    var c = 1.0
    var d = 1.0 - qab * x / qap
    if math.fabs(d) < TINY { d = TINY }
    d = 1.0 / d
    var out = d

    var m = 1
    while m <= 300 {
        let fm = float(m)
        let even = fm * (b - fm) * x / ((qam + 2.0 * fm) * (a + 2.0 * fm))
        d = 1.0 + even * d
        if math.fabs(d) < TINY { d = TINY }
        c = 1.0 + even / c
        if math.fabs(c) < TINY { c = TINY }
        d = 1.0 / d
        out = out * d * c

        let odd = -(a + fm) * (qab + fm) * x / ((a + 2.0 * fm) * (qap + 2.0 * fm))
        d = 1.0 + odd * d
        if math.fabs(d) < TINY { d = TINY }
        c = 1.0 + odd / c
        if math.fabs(c) < TINY { c = TINY }
        d = 1.0 / d
        let step = d * c
        out = out * step
        if math.fabs(step - 1.0) < 1e-15 { return out }
        m += 1
    }
    return out
}

# ------------------------------------------------------------------
# Distributions
# ------------------------------------------------------------------

#: The standard normal, under a name that does not collide with the matrix
#: norm. `scipy.stats.norm`'s four methods, and the same argument order.
pub class Normal: Printable {
    pub let loc: float
    pub let scale: float

    pub fn init(self, loc: float = 0.0, scale: float = 1.0) {
        if scale <= 0.0 { throw ValueError(f"a normal needs a positive scale, got {scale}") }
        self.loc = loc
        self.scale = scale
    }

    pub fn pdf(self, x: float) -> float {
        let z = (x - self.loc) / self.scale
        return math.exp(-0.5 * z * z) / (self.scale * math.sqrt(2.0 * math.PI))
    }

    #: Through `erfc` rather than `1 - erf`: past z ~ 2 the difference loses
    #: every significant digit, and a survival function is exactly where a
    #: caller cares about the tail.
    pub fn cdf(self, x: float) -> float {
        return 0.5 * erfc(-(x - self.loc) / (self.scale * math.sqrt(2.0)))
    }

    pub fn sf(self, x: float) -> float {
        return 0.5 * erfc((x - self.loc) / (self.scale * math.sqrt(2.0)))
    }

    #: The inverse CDF, by Acklam's rational approximation refined with one
    #: Halley step. The approximation alone is good to about 1.15e-9 relative;
    #: the refinement takes it to full double precision, which is what scipy
    #: reports and what a critical value should be.
    pub fn ppf(self, p: float) -> float {
        if p <= 0.0 or p >= 1.0 {
            throw ValueError(f"a quantile needs 0 < p < 1, got {p}")
        }
        return self.loc + self.scale * _standard_ppf(p)
    }

    pub fn to_str(self) -> str { return f"Normal(loc={self.loc}, scale={self.scale})" }

    fn __str__(self) -> str { return self.to_str() }
}

#: The standard normal, ready to use, as `scipy.stats.norm` is.
pub let normal = Normal(0.0, 1.0)

let ACKLAM_A: list[float] = [
    -3.969683028665376e+01,
    2.209460984245205e+02,
    -2.759285104469687e+02,
    1.383577518672690e+02,
    -3.066479806614716e+01,
    2.506628277459239e+00,
]
let ACKLAM_B: list[float] = [
    -5.447609879822406e+01,
    1.615858368580409e+02,
    -1.556989798598866e+02,
    6.680131188771972e+01,
    -1.328068155288572e+01,
]
let ACKLAM_C: list[float] = [
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e+00,
    -2.549732539343734e+00,
    4.374664141464968e+00,
    2.938163982698783e+00,
]
let ACKLAM_D: list[float] = [
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e+00,
    3.754408661907416e+00,
]

fn _standard_ppf(p: float) -> float {
    let low = 0.02425
    var x = 0.0
    if p < low {
        let q = math.sqrt(-2.0 * math.log(p))
        x = (((((ACKLAM_C[0] * q + ACKLAM_C[1]) * q + ACKLAM_C[2]) * q + ACKLAM_C[3]) * q + ACKLAM_C[4]) * q + ACKLAM_C[5]) /
            ((((ACKLAM_D[0] * q + ACKLAM_D[1]) * q + ACKLAM_D[2]) * q + ACKLAM_D[3]) * q + 1.0)
    } elif p <= 1.0 - low {
        let q = p - 0.5
        let r = q * q
        x = (((((ACKLAM_A[0] * r + ACKLAM_A[1]) * r + ACKLAM_A[2]) * r + ACKLAM_A[3]) * r + ACKLAM_A[4]) * r + ACKLAM_A[5]) * q /
            (((((ACKLAM_B[0] * r + ACKLAM_B[1]) * r + ACKLAM_B[2]) * r + ACKLAM_B[3]) * r + ACKLAM_B[4]) * r + 1.0)
    } else {
        let q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((ACKLAM_C[0] * q + ACKLAM_C[1]) * q + ACKLAM_C[2]) * q + ACKLAM_C[3]) * q + ACKLAM_C[4]) * q + ACKLAM_C[5]) /
            ((((ACKLAM_D[0] * q + ACKLAM_D[1]) * q + ACKLAM_D[2]) * q + ACKLAM_D[3]) * q + 1.0)
    }
    #: One Halley step on `Phi(x) - p`, which squares the digits: the
    #: approximation's 1.15e-9 becomes full precision, and a second step would
    #: buy nothing a float64 can hold.
    let e = 0.5 * erfc(-x / math.sqrt(2.0)) - p
    let u = e * math.sqrt(2.0 * math.PI) * math.exp(x * x / 2.0)
    return x - u / (1.0 + x * u / 2.0)
}

#: The Student t CDF with `df` degrees of freedom.
pub fn t_cdf(t: float, df: float) -> float {
    if df <= 0.0 { throw ValueError(f"a t distribution needs df > 0, got {df}") }
    let half = betainc(0.5 * df, 0.5, df / (df + t * t)) * 0.5
    if t > 0.0 { return 1.0 - half }
    return half
}

#: `1 - t_cdf`, computed on the side that does not cancel.
pub fn t_sf(t: float, df: float) -> float { return t_cdf(-t, df) }

#: The two-sided p-value for a t statistic, which is what every test here
#: reports.
fn two_sided(t: float, df: float) -> float {
    return betainc(0.5 * df, 0.5, df / (df + t * t))
}

# ------------------------------------------------------------------
# Correlation and regression
# ------------------------------------------------------------------
fn need_pairs(x: list[float], y: list[float], least: int, what: str) -> void {
    if x.len() != y.len() {
        throw ValueError(f"{what} got {x.len()} and {y.len()} values")
    }
    if x.len() < least {
        throw ValueError(f"{what} needs at least {least} pairs, got {x.len()}")
    }
}

#: What `scipy.stats.linregress` returns, with its field names.
pub class LinregressResult: Printable {
    pub let slope: float
    pub let intercept: float
    pub let rvalue: float
    pub let pvalue: float
    pub let stderr: float
    pub let intercept_stderr: float

    pub fn init(
        self,
        slope: float,
        intercept: float,
        rvalue: float,
        pvalue: float,
        stderr: float,
        intercept_stderr: float
    ) {
        self.slope = slope
        self.intercept = intercept
        self.rvalue = rvalue
        self.pvalue = pvalue
        self.stderr = stderr
        self.intercept_stderr = intercept_stderr
    }

    pub fn to_str(self) -> str {
        return f"LinregressResult(slope={self.slope}, intercept={self.intercept}, rvalue={self.rvalue})"
    }

    fn __str__(self) -> str { return self.to_str() }
}

#: Least-squares line with its correlation and its standard errors, as
#: `scipy.stats.linregress` gives them.
#:
#: The p-value tests the slope against zero, on `n - 2` degrees of freedom,
#: which is the same two-sided t test the coefficient table of any regression
#: reports.
pub fn linregress(x: list[float], y: list[float]) -> LinregressResult {
    need_pairs(x, y, 2, "linregress")
    let n = x.len()
    let mx = mean(x)
    let my = mean(y)
    var sxx = 0.0
    var syy = 0.0
    var sxy = 0.0
    for index in 0..n {
        let dx = x[index] - mx
        let dy = y[index] - my
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
    }
    if sxx == 0.0 { throw ValueError("linregress needs x values that are not all equal") }

    let slope = sxy / sxx
    let intercept = my - slope * mx
    let r = if syy == 0.0 { 0.0 } else { sxy / math.sqrt(sxx * syy) }
    if n == 2 {
        #: Two points determine the line exactly, so there is no residual to
        #: form a standard error from; scipy reports zeros and a p-value of
        #: one, and so does this.
        return LinregressResult(slope, intercept, r, 1.0, 0.0, 0.0)
    }
    let df = float(n - 2)
    #: The residual variance, written from the sums rather than by a second
    #: pass over the residuals: `syy - slope * sxy` IS the residual sum of
    #: squares, exactly, and one pass is enough.
    let residual = (syy - slope * sxy) / df
    let stderr = math.sqrt(residual / sxx)
    let intercept_stderr = math.sqrt(residual * (1.0 / float(n) + mx * mx / sxx))
    let t = slope / stderr
    return LinregressResult(slope, intercept, r, two_sided(t, df), stderr, intercept_stderr)
}

#: Pearson's correlation and its two-sided p-value, as `scipy.stats.pearsonr`.
pub fn pearsonr(x: list[float], y: list[float]) -> tuple[float, float] {
    need_pairs(x, y, 2, "pearsonr")
    let n = x.len()
    let mx = mean(x)
    let my = mean(y)
    var sxx = 0.0
    var syy = 0.0
    var sxy = 0.0
    for index in 0..n {
        let dx = x[index] - mx
        let dy = y[index] - my
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
    }
    if sxx == 0.0 or syy == 0.0 {
        throw ValueError("pearsonr needs values that are not all equal")
    }
    var r = sxy / math.sqrt(sxx * syy)
    #: Rounding can put `r` a hair outside [-1, 1] on perfectly correlated
    #: data, and the p-value's `1 - r*r` then goes negative.
    if r > 1.0 { r = 1.0 }
    if r < -1.0 { r = -1.0 }
    if n == 2 { return (r, 1.0) }
    let df = float(n - 2)
    if math.fabs(r) == 1.0 { return (r, 0.0) }
    let t = r * math.sqrt(df / (1.0 - r * r))
    return (r, two_sided(t, df))
}

#: Ranks with ties averaged, which is what a rank correlation needs.
fn ranks_of(values: list[float]) -> list[float] {
    let n = values.len()
    var order: list[int] = []
    for index in 0..n { order.push(index) }
    for i in 1..n {
        let held = order[i]
        var j = i - 1
        while j >= 0 and values[order[j]] > values[held] {
            order[j + 1] = order[j]
            j -= 1
        }
        order[j + 1] = held
    }
    var out = [0.0] * n
    var start = 0
    while start < n {
        var stop = start + 1
        while stop < n and values[order[stop]] == values[order[start]] { stop += 1 }
        let shared = 0.5 * float(start + stop - 1) + 1.0
        for index in start..stop { out[order[index]] = shared }
        start = stop
    }
    return out
}

#: Spearman's rank correlation and its p-value, as `scipy.stats.spearmanr`.
pub fn spearmanr(x: list[float], y: list[float]) -> tuple[float, float] {
    need_pairs(x, y, 2, "spearmanr")
    return pearsonr(ranks_of(x), ranks_of(y))
}

# ------------------------------------------------------------------
# t tests
# ------------------------------------------------------------------

#: One sample against a hypothesised mean, as `scipy.stats.ttest_1samp`.
pub fn ttest_1samp(a: list[float], popmean: float) -> tuple[float, float] {
    if a.len() < 2 { throw ValueError(f"ttest_1samp needs at least 2 values, got {a.len()}") }
    let n = float(a.len())
    let df = n - 1.0
    let t = (mean(a) - popmean) / math.sqrt(variance(a) / n)
    return (t, two_sided(t, df))
}

#: Two independent samples, as `scipy.stats.ttest_ind`.
#:
#: `equal_var` chooses between the pooled-variance test and Welch's, exactly as
#: scipy's flag does. Welch is the safer default in general -- it does not
#: assume the variances match -- but scipy's default is the pooled test and
#: agreeing with scipy matters more here than picking a side.
pub fn ttest_ind(a: list[float], b: list[float], equal_var: bool = true) -> tuple[float, float] {
    if a.len() < 2 or b.len() < 2 {
        throw ValueError("ttest_ind needs at least 2 values in each sample")
    }
    let na = float(a.len())
    let nb = float(b.len())
    let va = variance(a)
    let vb = variance(b)
    let difference = mean(a) - mean(b)

    if equal_var {
        let df = na + nb - 2.0
        let pooled = ((na - 1.0) * va + (nb - 1.0) * vb) / df
        let t = difference / math.sqrt(pooled * (1.0 / na + 1.0 / nb))
        return (t, two_sided(t, df))
    }
    let sa = va / na
    let sb = vb / nb
    let t = difference / math.sqrt(sa + sb)
    #: Welch–Satterthwaite: the degrees of freedom are not an integer, which is
    #: the whole point -- the incomplete beta takes a real first argument, so
    #: nothing has to be rounded.
    let df = (sa + sb) * (sa + sb) / (sa * sa / (na - 1.0) + sb * sb / (nb - 1.0))
    return (t, two_sided(t, df))
}

#: Paired samples, as `scipy.stats.ttest_rel`: the one-sample test on the
#: differences.
pub fn ttest_rel(a: list[float], b: list[float]) -> tuple[float, float] {
    if a.len() != b.len() {
        throw ValueError(f"ttest_rel got {a.len()} and {b.len()} values")
    }
    if a.len() < 2 { throw ValueError("ttest_rel needs at least 2 pairs") }
    var differences: list[float] = []
    for index in 0..a.len() { differences.push(a[index] - b[index]) }
    return ttest_1samp(differences, 0.0)
}

# ------------------------------------------------------------------
# Description
# ------------------------------------------------------------------

#: What `scipy.stats.describe` returns, with its field names.
pub class DescribeResult: Printable {
    pub let nobs: int
    pub let minmax: tuple[float, float]
    pub let mean: float
    pub let variance: float
    pub let skewness: float
    pub let kurtosis: float

    pub fn init(
        self,
        nobs: int,
        minmax: tuple[float, float],
        mean: float,
        variance: float,
        skewness: float,
        kurtosis: float
    ) {
        self.nobs = nobs
        self.minmax = minmax
        self.mean = mean
        self.variance = variance
        self.skewness = skewness
        self.kurtosis = kurtosis
    }

    pub fn to_str(self) -> str {
        return f"DescribeResult(nobs={self.nobs}, mean={self.mean}, variance={self.variance})"
    }

    fn __str__(self) -> str { return self.to_str() }
}

#: The six numbers `scipy.stats.describe` gives.
#:
#: `variance` is the sample variance (n - 1); `skewness` and `kurtosis` are the
#: biased moment estimates on n, and the kurtosis is EXCESS -- three subtracted
#: -- which is scipy's default and the one place the two conventions differ
#: silently.
pub fn describe(a: list[float]) -> DescribeResult {
    let n = a.len()
    if n < 2 { throw ValueError(f"describe needs at least 2 values, got {n}") }
    var low = a[0]
    var high = a[0]
    for value in a {
        if value < low { low = value }
        if value > high { high = value }
    }
    let m = mean(a)
    var m2 = 0.0
    var m3 = 0.0
    var m4 = 0.0
    for value in a {
        let d = value - m
        let d2 = d * d
        m2 += d2
        m3 += d2 * d
        m4 += d2 * d2
    }
    let count = float(n)
    m2 = m2 / count
    m3 = m3 / count
    m4 = m4 / count
    let skew = if m2 == 0.0 { 0.0 } else { m3 / (m2 * math.sqrt(m2)) }
    let kurt = if m2 == 0.0 { -3.0 } else { m4 / (m2 * m2) - 3.0 }
    return DescribeResult(n, (low, high), m, variance(a), skew, kurt)
}
