#: Float sequence generators and the range helpers an axis is drawn from.
#:
#: Every `range` in Jaithon is integer — the builtin, the `a..b` literal and
#: `std.iter.range_step` alike — so a float sequence has to be built by hand.
#: `linspace`, `arange` and `logspace` are the three shapes a plot asks for, and
#: `minmax`, `extent`, `percentile` and `histogram_bins` turn a data series into
#: the numbers an axis is laid out from.
#:
#: Each generator computes its i-th element from the index rather than by adding
#: a step to the element before, so a value carries one rounding error instead of
#: `i` of them and the endpoints land exactly where they were asked to.
#:
#: NaN is not something this module passes on. Jaithon raises `TypeError` on any
#: ordered comparison involving one — `nan < 1.0` throws rather than answering
#: false — so a NaN that reaches a scale, a sort or a tick search takes the whole
#: render down instead of drawing one wrong pixel. `finite_only` is the guard:
#: `minmax` skips NaN, and `extent` drops every non-finite value because an axis
#: limit of infinity cannot be drawn.
#:
#: ```
#: from jaiplot.data import linspace, extent
#: linspace(0.0, 1.0, 5)    # [0.0, 0.25, 0.5, 0.75, 1.0]
#: extent([1.0, 3.0])       # (0.9, 3.1)
#: ```
import std.math
from std.math import is_finite
from std.algo.stat import histogram, quantile

export { is_finite }

# --- Generators --------------------------------------------------------------

#: Return `count` evenly spaced values running from `start` to `stop` inclusive.
#:
#: Both endpoints are exact, bit for bit: `result[0]` is `start` and the last
#: element is `stop`. The obvious `start + index * step` is not, because `step`
#: is itself rounded and the index multiplies that error — for
#: `linspace(0.1, 0.3, 4)` it lands on 0.30000000000000004, an ulp *past* the
#: stop it was given, which draws a point outside the plot area and leaves the
#: axis limit and the data disagreeing about where the series ends. The
#: interpolation `std.math.lerp` performs answers `start` at 0.0 and `stop` at
#: 1.0 by construction, and `index / (count - 1)` hits both exactly, so the sign
#: of a zero survives as well as its value.
#:
#: Non-finite endpoints are not rejected; they produce non-finite samples, which
#: `finite_only` removes. Raises `ValueError` when `count` is below 2, since one
#: point cannot span two endpoints.
#:
#: ```
#: linspace(0.0, 1.0, 5)     # [0.0, 0.25, 0.5, 0.75, 1.0]
#: linspace(2.0, 3.0, 2)     # [2.0, 3.0]
#: linspace(1.0, -1.0, 3)    # [1.0, 0.0, -1.0]
#: ```
pub fn linspace(start: float, stop: float, count: int) -> list[float] {
    if count < 2 {
        throw ValueError(f"count must be at least 2 to span both endpoints, got {count}")
    }
    let last = count - 1
    return [math.lerp(start, stop, float(index) / float(last)) for index in 0..=last]
}

#: Return the values from `start` in steps of `step`, stopping before `stop`.
#:
#: Half-open like the integer `range`: `stop` is never a member. A negative
#: `step` counts down, in which case `stop` must lie below `start`.
#:
#: Each element is `start + index * step`, never a running total. Accumulation
#: adds one rounding error per element, and the errors decide the length as well
#: as the values: summing 0.1 towards 1.0 arrives at 0.9999999999999999, which is
#: still below the stop, so the loop yields an eleventh element where this
#: function yields ten ending at 0.9. Scaling the index instead holds the error
#: to one rounding per element, and makes the i-th value the same whether the
#: caller asked for ten elements or a million.
#:
#: The length is `ceil((stop - start) / step)`, and that quotient can round up
#: past a whole number and name one element too many. The extra element is
#: dropped by testing it against `stop` rather than by subtracting a tolerance,
#: which would only move the boundary somewhere else.
#:
#: Raises `ValueError` when `step` is zero, when the sign of `step` runs away
#: from `stop`, when any argument is not finite, and when `(stop - start) / step`
#: overflows, which is a step too small to walk the distance in any number of
#: elements a machine could hold.
#:
#: ```
#: arange(0.0, 1.0, 0.25)    # [0.0, 0.25, 0.5, 0.75]
#: arange(3.0, 0.0, -1.0)    # [3.0, 2.0, 1.0]
#: arange(2.0, 2.0, 0.5)     # []
#: ```
pub fn arange(start: float, stop: float, step: float) -> list[float] {
    if not is_finite(start) or not is_finite(stop) or not is_finite(step) {
        throw ValueError(f"arange needs finite bounds, got start={start}, stop={stop}, step={step}")
    }
    if step == 0.0 {
        throw ValueError("step must be non-zero, got 0.0")
    }
    let span = stop - start
    if span != 0.0 and not _is_before(start, stop, step) {
        throw ValueError(f"step {step} never reaches stop {stop} from start {start}")
    }
    let length = math.ceil(span / step)
    if not is_finite(length) {
        throw ValueError(f"step {step} is too small to walk from {start} to {stop}")
    }
    var count = int(length)
    while count > 0 and not _is_before(start + float(count - 1) * step, stop, step) {
        count -= 1
    }
    return [start + float(index) * step for index in 0..count]
}

#: Return `count` values spaced evenly on a logarithmic axis.
#:
#: The arguments are exponents, not values: the exponents run from
#: `start_exponent` to `stop_exponent` inclusive exactly as `linspace` lays them
#: out, and each becomes `base ** exponent`. So `logspace(0.0, 3.0, 4)` is the
#: decades 1, 10, 100, 1000, and `logspace(1.0, 100.0, 4)` asks for numbers up
#: to `10 ** 100` rather than for the range 1 to 100.
#:
#: The exponents are exact at both ends; the values are only as exact as `**`,
#: so a returned decade can miss the round number by an ulp and is not safe to
#: compare with `==`.
#:
#: Raises `ValueError` when `count` is below 2, and when `base` is not positive
#: — a base of zero or less has no real power at a fractional exponent.
#:
#: ```
#: logspace(0.0, 3.0, 4)               # [1.0, 10.0, 100.0, 1000.0]
#: logspace(0.0, 3.0, 4, base: 2.0)    # [1.0, 2.0, 4.0, 8.0]
#: ```
pub fn logspace(
    start_exponent: float,
    stop_exponent: float,
    count: int,
    base: float = 10.0
) -> list[float] {
    if not is_finite(base) or base <= 0.0 {
        throw ValueError(f"base must be a positive finite number, got {base}")
    }
    let exponents = linspace(start_exponent, stop_exponent, count)
    return [base ** exponent for exponent in exponents]
}

# --- Ranges and guards -------------------------------------------------------

#: Return the smallest and largest of `values` as `(lowest, highest)`, or null.
#:
#: NaN is skipped rather than compared. `std.algo.stat.min_max` reaches for the
#: `min` builtin, which raises `TypeError` on a NaN because the ordering is
#: undefined, so a single missing sample in a series would abort the render
#: instead of being left out of the range. A list that is empty *or* holds
#: nothing but NaN has no extremes to report and answers null, which puts both
#: absences in the one branch the caller has to write anyway.
#:
#: Infinities are kept: `-inf` really is the smallest value present, and saying
#: otherwise would hide it. Use `extent`, or filter with `finite_only`, when the
#: answer has to be drawable.
#:
#: ```
#: minmax([3.0, 1.0, 4.0, 1.0, 5.0])    # (1.0, 5.0)
#: minmax([2.0])                        # (2.0, 2.0)
#: ```
pub fn minmax(values: list[float]) -> tuple[float, float]? {
    var lowest = 0.0
    var highest = 0.0
    var seen = false
    for value in values {
        if math.is_nan(value) { continue }
        if not seen {
            lowest = value
            highest = value
            seen = true
        } else {
            if value < lowest { lowest = value }
            if value > highest { highest = value }
        }
    }
    if not seen { return null }
    return (lowest, highest)
}

#: Return the range of `values` padded outward, as `(low, high)`, or null.
#:
#: The ordinary axis-limit rule: `pad_fraction` of the data's width is added to
#: each side, so a series never runs into the frame it is drawn in. Only finite
#: values are considered, because a limit of infinity has no device coordinate
#: and one infinite sample would otherwise flatten every other point onto a
#: single pixel. Null exactly when nothing finite is left.
#:
#: A constant series has no width to take a fraction of, and a zero-width range
#: divides by zero in every transform downstream. Its padding comes from the
#: value's own magnitude instead — the range becomes `value` plus and minus
#: `|value| * pad_fraction` — which keeps the same relative framing as a series
#: that does vary. When even that is zero, because the value is exactly zero or
#: `pad_fraction` is, it falls back to `value` plus and minus 0.5: a unit-wide
#: range centred on the data, which is the widest thing that can be said about a
#: series that says nothing about its own scale.
#:
#: Raises `ValueError` for a negative or non-finite `pad_fraction`; a negative
#: one would pull the limits inside the data and clip it.
#:
#: ```
#: extent([1.0, 3.0])                       # (0.9, 3.1)
#: extent([1.0, 3.0], pad_fraction: 0.0)    # (1.0, 3.0)
#: extent([2.0, 2.0])                       # (1.9, 2.1)
#: extent([0.0, 0.0])                       # (-0.5, 0.5)
#: ```
pub fn extent(values: list[float], pad_fraction: float = 0.05) -> tuple[float, float]? {
    if not is_finite(pad_fraction) or pad_fraction < 0.0 {
        throw ValueError(f"pad_fraction must be a non-negative finite fraction, got {pad_fraction}")
    }
    let bounds = minmax(finite_only(values))
    if bounds == null { return null }
    let (lowest, highest) = bounds
    let width = highest - lowest
    if width > 0.0 {
        let padding = width * pad_fraction
        return (lowest - padding, highest + padding)
    }
    var half = math.fabs(lowest) * pad_fraction
    if half == 0.0 { half = 0.5 }
    return (lowest - half, highest + half)
}

#: Return the value below which `p` percent of `values` fall.
#:
#: A percent-scale front for `std.algo.stat.quantile`, which takes a fraction:
#: `percentile(values, 90.0)` is `quantile(values, 0.9)`. Axis, whisker and
#: outlier code all read in percent, and a conversion repeated at every call
#: site is where a factor of a hundred goes missing.
#:
#: Interpolated linearly between order statistics — R's type 7, the default of
#: NumPy and of most spreadsheets — so `p = 0` is the minimum, `p = 100` the
#: maximum, `p = 50` the median, and the answer need not be a member of
#: `values`. Sorting is what does the comparing, so a NaN anywhere in `values`
#: raises `TypeError`; pass `finite_only(values)` when that is possible.
#:
#: Raises `ValueError` when `p` lies outside `[0, 100]` or is NaN, and when
#: `values` is empty.
#:
#: ```
#: percentile([1.0, 2.0, 3.0, 4.0], 25.0)     # 1.75
#: percentile([1.0, 2.0, 3.0, 4.0], 50.0)     # 2.5
#: percentile([1.0, 2.0, 3.0, 4.0], 100.0)    # 4.0
#: ```
pub fn percentile(values: list[float], p: float) -> float {
    if math.is_nan(p) or p < 0.0 or p > 100.0 {
        throw ValueError(f"p must be a percentage in [0, 100], got {p}")
    }
    return quantile(values, p / 100.0)
}

#: Return the bin edges and the count in each bin, as `(edges, counts)`.
#:
#: `std.algo.stat.histogram` under a name that says what comes back, so a
#: histogram series needs only `jaiplot.data` in scope. `edges` holds one more
#: entry than `counts`, since `n` bins have `n + 1` boundaries, and the bin
#: between `edges[i]` and `edges[i + 1]` holds `counts[i]`. The last bin is
#: closed at both ends so the maximum lands inside it rather than in a bin of
#: its own, and a `bins` of null picks the count by the Freedman-Diaconis rule.
#:
#: A NaN in `values` raises `TypeError` from the comparison that finds the
#: extremes; filter with `finite_only` first when that is possible.
#:
#: Raises `ValueError` for an empty list or a `bins` below 1.
#:
#: ```
#: let (edges, counts) = histogram_bins([1.0, 2.0, 3.0, 4.0], 2)
#: edges     # [1.0, 2.5, 4.0]
#: counts    # [2, 2]
#: ```
pub fn histogram_bins(values: list[float], bins: int? = null) -> tuple[list[float], list[int]] {
    return histogram(values, bins)
}

#: Return the values that are neither NaN nor infinite, in their original order.
#:
#: The guard between data and geometry. A NaN that reaches a comparison raises
#: `TypeError` and takes down the render; an infinity that reaches a scale sends
#: every finite point to the same coordinate. Neither has a position on a page,
#: so leaving the point out is the only drawing that is honest about what the
#: data says — and dropping it here, once, is what lets everything downstream
#: compare freely.
#:
#: ```
#: finite_only([1.0, math.NAN, math.INF, 2.0])    # [1.0, 2.0]
#: finite_only([])                                # []
#: ```
pub fn finite_only(values: list[float]) -> list[float] {
    return values.filter(is_finite)
}

fn _is_before(value: float, stop: float, step: float) -> bool {
    if step > 0.0 { return value < stop }
    return value > stop
}
