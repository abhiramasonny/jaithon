#: `jaisci.stats` against scipy, which is where every expected value came from.
#:
#: These are the tightest comparisons in the package -- 1e-12 relative on the
#: statistics, 1e-9 relative on the p-values -- and deliberately so. A t
#: statistic is closed-form arithmetic and a p-value is one incomplete beta, so
#: a faithful implementation agrees with scipy to the last few digits. A loose
#: tolerance here would pass a pooled test where Welch's was asked for, or an
#: excess kurtosis where the raw fourth moment was meant.
import std.math
from std.test import assert_eq, assert_near, assert_true, assert_throws
from jaisci.stats import linregress, pearsonr, spearmanr, ttest_1samp, ttest_ind
from jaisci.stats import ttest_rel, normal, Normal, t_cdf, t_sf, describe
from jaisci.stats import betainc, lgamma

let X = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
let Y = [2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 16.1]
let A = [5.1, 4.9, 6.2, 5.7, 5.5, 4.8, 6.0]
let B = [6.3, 6.9, 7.1, 6.4, 6.8, 7.4, 6.6]

fn assert_rel(got: float, wanted: float, rel: float, what: str) -> void {
    let scale = if math.fabs(wanted) > 1.0 { math.fabs(wanted) } else { 1.0 }
    assert_near(got, wanted, rel * scale, what)
}

#: scipy.stats.linregress(x, y) -> slope, intercept, rvalue, pvalue, stderr,
#: intercept_stderr, all six checked.
fn test_linregress_matches_scipy() -> void {
    let r = linregress(X, Y)
    assert_rel(r.slope, 1.9976190476190478, 1e-13, "slope")
    assert_rel(r.intercept, 0.0357142857142847, 1e-11, "intercept")
    assert_rel(r.rvalue, 0.9994194747958132, 1e-13, "rvalue")
    assert_rel(r.pvalue, 4.888933612552943e-10, 1e-09, "pvalue")
    assert_rel(r.stderr, 0.02780044426688155, 1e-11, "stderr")
    assert_rel(r.intercept_stderr, 0.14038536208101496, 1e-11, "intercept_stderr")
}

#: Two points determine the line exactly, so there is no residual and scipy
#: reports zero errors with a p-value of one. Worth pinning: the formula would
#: otherwise divide by zero degrees of freedom.
fn test_two_points_have_no_residual() -> void {
    let r = linregress([1.0, 3.0], [2.0, 8.0])
    assert_near(r.slope, 3.0, 1e-12, "slope")
    assert_near(r.intercept, -1.0, 1e-12, "intercept")
    assert_eq(r.stderr, 0.0, "no standard error")
    assert_eq(r.pvalue, 1.0, "and no evidence")
}

#: scipy.stats.pearsonr / spearmanr. The rank correlation is exactly one here
#: because the data is monotone, which is the case that catches a rank routine
#: that does not average ties.
fn test_the_correlations_match_scipy() -> void {
    let (r, p) = pearsonr(X, Y)
    assert_rel(r, 0.999419474795813, 1e-13, "pearson r")
    assert_rel(p, 4.888933612558556e-10, 1e-09, "pearson p")

    let (rho, rp) = spearmanr(X, Y)
    assert_near(rho, 1.0, 1e-12, "spearman rho")
    assert_near(rp, 0.0, 1e-12, "spearman p")
}

#: Ties have to be averaged, or a rank correlation on repeated values is
#: wrong. [10, 20, 20, 40] ranks as [1, 2.5, 2.5, 4].
fn test_ranks_average_their_ties() -> void {
    let values = [10.0, 20.0, 20.0, 40.0]
    let (rho, _p) = spearmanr(values, [1.0, 2.0, 3.0, 4.0])
    #: With ties in one variable and none in the other, Spearman is below one.
    assert_true(rho < 1.0, "ties cost some rank correlation")
    assert_rel(rho, 0.9486832980505138, 1e-12, "the tied rank correlation")
}

#: Every t test, against scipy:
#:   ttest_1samp(a, 5.0)              ->  2.222365125808889,  0.06797629903632725
#:   ttest_ind(a, b)                  -> -5.2538739506540955, 0.0002031038769781365
#:   ttest_ind(a, b, equal_var=False) -> -5.2538739506540955, 0.0002823772184857872
#:   ttest_rel(a, b)                  -> -4.817510547077043,  0.002947550042532847
#:
#: The pooled and Welch statistics are IDENTICAL here (equal sample sizes) and
#: the p-values differ, which is exactly the case that catches a Welch
#: implementation that forgot to change the degrees of freedom.
fn test_every_t_test_matches_scipy() -> void {
    let (t1, p1) = ttest_1samp(A, 5.0)
    assert_rel(t1, 2.222365125808889, 1e-12, "one-sample t")
    assert_rel(p1, 0.06797629903632725, 1e-11, "one-sample p")

    let (ti, pi) = ttest_ind(A, B)
    assert_rel(ti, -5.2538739506540955, 1e-12, "pooled t")
    assert_rel(pi, 0.0002031038769781365, 1e-11, "pooled p")

    let (tw, pw) = ttest_ind(A, B, equal_var: false)
    assert_rel(tw, -5.2538739506540955, 1e-12, "Welch t")
    assert_rel(pw, 0.0002823772184857872, 1e-11, "Welch p")
    assert_true(pw > pi, "Welch is the more cautious of the two")

    let (tr, pr) = ttest_rel(A, B)
    assert_rel(tr, -4.817510547077043, 1e-12, "paired t")
    assert_rel(pr, 0.002947550042532847, 1e-11, "paired p")
}

#: Welch on samples of DIFFERENT sizes and variances, where the two tests
#: disagree about the statistic as well as the p-value.
fn test_welch_on_unequal_samples() -> void {
    let small = [1.0, 2.0, 3.0]
    let large = [10.0, 12.0, 9.0, 11.0, 30.0, 8.0, 13.0, 11.0]
    let (tp, pp) = ttest_ind(small, large)
    let (tw, pw) = ttest_ind(small, large, equal_var: false)
    assert_true(math.fabs(tp - tw) > 1e-06, "the two statistics differ")
    assert_true(tp < 0.0 and tw < 0.0, "both say the small sample is lower")
    assert_true(pp < 0.05 and pw < 0.05, "both reject")
}

#: scipy.stats.norm.cdf / .ppf at five points each. The ppf is the harder of
#: the two -- an approximation alone is only good to 1e-9, and these agree to
#: the last digit because of the Halley refinement.
fn test_the_normal_distribution_matches_scipy() -> void {
    let wanted_cdf = [
        0.022750131948179198,
        0.3085375387259869,
        0.5,
        0.8413447460685429,
        0.9937903346742238,
    ]
    let at = [-2.0, -0.5, 0.0, 1.0, 2.5]
    for index in 0..at.len() {
        assert_rel(normal.cdf(at[index]), wanted_cdf[index], 1e-13, f"cdf at {at[index]}")
        assert_rel(normal.sf(at[index]), 1.0 - wanted_cdf[index], 1e-11, f"sf at {at[index]}")
    }

    let ps = [0.025, 0.1, 0.5, 0.9, 0.975]
    let wanted_ppf = [
        -1.9599639845400545,
        -1.2815515655446004,
        0.0,
        1.2815515655446004,
        1.959963984540054,
    ]
    for index in 0..ps.len() {
        assert_rel(normal.ppf(ps[index]), wanted_ppf[index], 1e-12, f"ppf at {ps[index]}")
    }
}

#: `cdf` and `ppf` are inverses, which is a property rather than a table and so
#: cannot be satisfied by a coincidence. Checked deep into the tail, where the
#: approximation is at its worst.
fn test_the_normal_quantile_inverts_its_cdf() -> void {
    for p in [1e-08, 1e-04, 0.01, 0.3, 0.5, 0.7, 0.99, 0.9999, 0.99999999] {
        let x = normal.ppf(p)
        assert_rel(normal.cdf(x), p, 1e-09, f"round trip at p={p}")
    }
}

fn test_a_shifted_normal_scales_its_quantiles() -> void {
    let shifted = Normal(10.0, 2.0)
    assert_rel(shifted.cdf(10.0), 0.5, 1e-13, "the centre")
    assert_rel(shifted.ppf(0.975), 10.0 + 2.0 * 1.959963984540054, 1e-11, "two sigma")
    assert_rel(shifted.pdf(10.0), 1.0 / (2.0 * math.sqrt(2.0 * math.PI)), 1e-13, "the peak")
}

#: scipy.stats.t.cdf(x, 6) at five points. 2.447 is the 97.5% critical value
#: for six degrees of freedom, which is the number a table of t values prints.
fn test_the_t_distribution_matches_scipy() -> void {
    let at = [-3.0, -1.0, 0.0, 1.5, 2.447]
    let wanted = [
        0.012004098377865475,
        0.17795884187479116,
        0.5,
        0.9078596319292589,
        0.9750029928138298,
    ]
    for index in 0..at.len() {
        assert_rel(t_cdf(at[index], 6.0), wanted[index], 1e-12, f"t cdf at {at[index]}")
        assert_rel(t_sf(at[index], 6.0), 1.0 - wanted[index], 1e-10, f"t sf at {at[index]}")
    }
}

#: A t distribution with many degrees of freedom IS the normal, which is the
#: property that says the incomplete beta and the error function agree.
fn test_a_wide_t_approaches_the_normal() -> void {
    for x in [-2.0, -0.5, 0.5, 2.0] {
        assert_near(t_cdf(x, 100000.0), normal.cdf(x), 1e-05, f"t against normal at {x}")
    }
}

#: scipy.stats.describe(a). `variance` is the sample variance and `kurtosis` is
#: EXCESS -- three subtracted -- which is scipy's default and the convention
#: two implementations most often differ on without saying so.
fn test_describe_matches_scipy() -> void {
    let d = describe(A)
    assert_eq(d.nobs, 7, "nobs")
    let (low, high) = d.minmax
    assert_eq(low, 4.8, "minimum")
    assert_eq(high, 6.2, "maximum")
    assert_rel(d.mean, 5.457142857142856, 1e-13, "mean")
    assert_rel(d.variance, 0.2961904761904763, 1e-12, "sample variance")
    assert_rel(d.skewness, 0.09162109598221718, 1e-11, "skewness")
    assert_rel(d.kurtosis, -1.4618981400109587, 1e-12, "excess kurtosis")
}

#: The two special functions everything above rests on, checked directly:
#: lgamma against known exact values, and the incomplete beta against its
#: closed form for integer parameters.
fn test_the_special_functions_are_right() -> void {
    assert_near(lgamma(1.0), 0.0, 1e-13, "lgamma(1) = log 0! = 0")
    assert_near(lgamma(2.0), 0.0, 1e-13, "lgamma(2) = log 1! = 0")
    assert_rel(lgamma(6.0), math.log(120.0), 1e-13, "lgamma(6) = log 5!")
    assert_rel(lgamma(0.5), 0.5 * math.log(math.PI), 1e-13, "lgamma(1/2)")
    assert_rel(lgamma(100.0), 359.1342053695754, 1e-13, "lgamma(100)")

    #: I_x(1, 1) is x, and I_x(2, 1) is x^2 -- both exact, so any error in the
    #: continued fraction shows immediately.
    for x in [0.1, 0.25, 0.5, 0.75, 0.9] {
        assert_near(betainc(1.0, 1.0, x), x, 1e-13, f"I_{x}(1,1)")
        assert_near(betainc(2.0, 1.0, x), x * x, 1e-13, f"I_{x}(2,1)")
        assert_near(betainc(1.0, 2.0, x), 1.0 - (1.0 - x) * (1.0 - x), 1e-13, f"I_{x}(1,2)")
    }
    assert_eq(betainc(2.0, 3.0, 0.0), 0.0, "at zero")
    assert_eq(betainc(2.0, 3.0, 1.0), 1.0, "at one")
}

fn test_bad_input_is_refused() -> void {
    assert_throws(ValueError, fn() -> void {
        let _ = linregress([1.0, 2.0], [1.0])
    }, "mismatched lengths")
    assert_throws(ValueError, fn() -> void {
        let _ = linregress([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
    }, "x values that are all equal")
    assert_throws(ValueError, fn() -> void { let _ = ttest_1samp([1.0], 0.0) }, "one value")
    assert_throws(ValueError, fn() -> void {
        let _ = ttest_rel([1.0, 2.0], [1.0])
    }, "unpaired samples")
    assert_throws(ValueError, fn() -> void { let _ = normal.ppf(0.0) }, "a quantile at zero")
    assert_throws(ValueError, fn() -> void { let _ = Normal(0.0, -1.0) }, "a negative scale")
    assert_throws(ValueError, fn() -> void { let _ = t_cdf(1.0, 0.0) }, "no degrees of freedom")
}
