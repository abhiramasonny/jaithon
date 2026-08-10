#: Where to put tick marks, and what to write beside them.
#:
#: An axis has to choose positions a reader can do arithmetic with. 0, 25, 50,
#: 75, 100 is a useful axis; 0, 23.7, 47.4 is the same information and no help
#: at all. The rule is Heckbert's: round the rough step up to the nearest 1, 2 or
#: 5 times a power of ten, then place ticks at multiples of it.
#:
#: Labels are chosen from the step rather than from the values, so every tick on
#: one axis is written to the same number of decimals and the column lines up.
#:
#: ```
#: linear_ticks(0.0, 100.0, 5)      # [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
#: linear_ticks(0.0, 1.0, 5)        # [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
#: ```
import std.math
from std.fmt import format

#: Largest magnitude written in plain decimal before labels turn scientific.
pub const SCIENTIFIC_ABOVE = 100000.0

#: Smallest non-zero magnitude written in plain decimal before labels turn
#: scientific.
pub const SCIENTIFIC_BELOW = 0.0001

#: Tick positions inside `[low, high]`, at nice round multiples.
#:
#: The step comes from `count_hint`, but the count returned does not: rounding
#: the step to something readable is worth more than hitting a target number of
#: ticks, so expect `count_hint` give or take two. Both ends are included when
#: they happen to fall on a multiple of the step.
#:
#: Returns the single value when `low == high`, and raises `ValueError` on a
#: reversed or non-finite interval, or on a hint below two.
#:
#: ```
#: linear_ticks(0.0, 10.0, 5)       # [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
#: linear_ticks(3.0, 3.0, 5)        # [3.0]
#: ```
pub fn linear_ticks(low: float, high: float, count_hint: int) -> list[float] {
    _check_interval(low, high, count_hint)
    if low == high { return [low] }

    let step = tick_step(low, high, count_hint)
    # Start at the first multiple of `step` at or above `low`. Multiplying an
    # index by the step rather than accumulating keeps every tick exact to the
    # same relative error, so 0.1 * 3 is one rounding rather than three.
    let first = math.ceil(low / step)
    let last = math.floor(high / step)
    if first > last { return [low, high] }

    var values = []
    var index = first
    while index <= last {
        let value = index * step
        # `first` and `last` were rounded outward from a division, so a value
        # can land a few ulp outside the interval it was derived from. Clamping
        # matters because a tick outside the limits maps outside the axes.
        if value >= low and value <= high { values.push(_snap(value, step)) }
        index += 1.0
    }
    if values.len() == 0 { return [low, high] }
    return values
}

#: The distance between neighbouring ticks for this interval and hint.
#:
#: Always 1, 2 or 5 times a power of ten.
#:
#: ```
#: tick_step(0.0, 100.0, 5)     # 20.0
#: tick_step(0.0, 1.0, 11)      # 0.1
#: ```
pub fn tick_step(low: float, high: float, count_hint: int) -> float {
    _check_interval(low, high, count_hint)
    if low == high { return 1.0 }
    # Heckbert rounds the range outward before dividing it. That is one rounding
    # too many: a range of 6.28 becomes 10, the step becomes 2, and an axis that
    # should have ended at 7 runs to 8 with a quarter of it empty. Dividing the
    # true range and rounding only the step keeps the same nice values and stops
    # the axis overshooting.
    return _nice_number((high - low) / (count_hint - 1).to_float(), true)
}

#: Limits rounded outward to whole ticks, for an axis that ranges itself.
#:
#: An axis whose limits the user set keeps them exactly; one that ranges itself
#: from the data extends to the enclosing ticks, so the axis ends on a labelled
#: mark instead of somewhere arbitrary.
#:
#: ```
#: nice_bounds(2.3, 97.1, 5)    # (0.0, 100.0)
#: ```
pub fn nice_bounds(low: float, high: float, count_hint: int) -> tuple[float, float] {
    _check_interval(low, high, count_hint)
    if low == high { return (low, high) }
    let step = tick_step(low, high, count_hint)
    return (math.floor(low / step) * step, math.ceil(high / step) * step)
}

#: Labelled tick positions for a logarithmic axis, at whole powers of `base`.
#:
#: These are the ones that get a label. `log_minor_ticks` supplies the
#: intermediate multiples, which are drawn as unlabelled rules.
#:
#: Raises `ValueError` when either limit is not positive, since the logarithm of
#: zero or a negative number is not a position on the axis.
#:
#: ```
#: log_ticks(1.0, 1000.0, 10.0, 4)      # [1.0, 10.0, 100.0, 1000.0]
#: ```
pub fn log_ticks(low: float, high: float, base: float, count_hint: int) -> list[float] {
    _check_log(low, high, base, count_hint)
    if low == high { return [low] }

    let first = math.floor(math.log_base(low, base))
    let last = math.ceil(math.log_base(high, base))
    # A decade per tick is right until the axis spans more decades than a reader
    # will follow, at which point every second or fifth power is enough.
    let decades = last - first
    var stride = 1.0
    while decades / stride > (count_hint * 2).to_float() { stride += 1.0 }

    var values = []
    var exponent = first
    while exponent <= last {
        let value = base ** exponent
        if value >= low and value <= high { values.push(value) }
        exponent += stride
    }
    if values.len() == 0 { return [low, high] }
    return values
}

#: Unlabelled tick positions between the powers, at 2 to 9 times each.
#:
#: A decade showing only its two ends reads as though nothing lies between
#: them, so a log axis draws these as short rules. They are dropped entirely
#: once the axis spans more than `max_decades` powers, where they would crowd
#: into a solid band.
#:
#: ```
#: log_minor_ticks(1.0, 10.0, 10.0)     # [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
#: log_minor_ticks(1.0, 1.0e12, 10.0)   # []
#: ```
pub fn log_minor_ticks(low: float, high: float, base: float, max_decades: int = 4) -> list[float] {
    _check_log(low, high, base, 2)
    if low == high { return [] }

    let first = math.floor(math.log_base(low, base))
    let last = math.ceil(math.log_base(high, base))
    if last - first > max_decades.to_float() { return [] }

    var values = []
    var exponent = first
    while exponent <= last {
        let power = base ** exponent
        var multiple = 2.0
        while multiple < base {
            let value = power * multiple
            if value >= low and value <= high { values.push(value) }
            multiple += 1.0
        }
        exponent += 1.0
    }
    return values
}

#: Reject the limits a logarithmic axis cannot represent.
fn _check_log(low: float, high: float, base: float, count_hint: int) -> void {
    _check_interval(low, high, count_hint)
    if low <= 0.0 {
        throw ValueError(f"a logarithmic axis needs positive limits, got a low of {low}")
    }
    if base <= 1.0 {
        throw ValueError(f"a logarithmic base must exceed 1, got {base}")
    }
}

#: How many decimal places a label needs to distinguish neighbouring ticks.
#:
#: ```
#: decimals_for_step(20.0)      # 0
#: decimals_for_step(0.05)      # 2
#: ```
pub fn decimals_for_step(step: float) -> int {
    if not math.is_finite(step) or step <= 0.0 {
        throw ValueError(f"a tick step must be finite and positive, got {step}")
    }
    let exponent = math.floor(math.log10(step))
    if exponent >= 0.0 { return 0 }
    return int(-exponent)
}

#: Write one tick value, to the precision `step` calls for.
#:
#: Values too large or too small to read as plain decimals are written in
#: scientific notation instead. Passing the same `step` for every tick on an
#: axis is what keeps the labels a tidy column.
#:
#: ```
#: format_tick(0.5, 0.05)       # "0.50"
#: format_tick(50.0, 20.0)      # "50"
#: format_tick(0.0, 0.05)       # "0.00"
#: ```
pub fn format_tick(value: float, step: float) -> str {
    if math.is_nan(value) { return "NaN" }
    if math.is_inf(value) { return if value > 0.0 { "inf" } else { "-inf" } }

    let magnitude = math.fabs(value)
    if magnitude != 0.0 and (magnitude >= SCIENTIFIC_ABOVE or magnitude < SCIENTIFIC_BELOW) {
        return _trim_exponent(format("{:.2e}", value))
    }
    # The spec is built by concatenation because a format spec is lifted as a
    # compile-time literal, so `{:.{digits}f}` cannot work here.
    let text = format("{:." + str(decimals_for_step(step)) + "f}", value)
    if text.starts_with("-") and text.to_float() == 0.0 { return text[1:] }
    return text
}

#: Round `value` onto the grid `step` defines, killing the drift a division and
#: a multiplication leave behind.
#:
#: Without this, an axis stepping by 0.1 produces 0.30000000000000004, which is
#: the right position and the wrong label.
fn _snap(value: float, step: float) -> float {
    let decimals = decimals_for_step(step)
    if decimals > 12 { return value }
    return math.round_to(value, decimals + 1)
}

#: The 1, 2 or 5 times a power of ten nearest `value`.
#:
#: `round_it` picks the nearest such number; otherwise the smallest one that is
#: at least `value`, which is what an enclosing span wants.
fn _nice_number(value: float, round_it: bool) -> float {
    let magnitude = math.fabs(value)
    if magnitude == 0.0 { return 1.0 }
    let exponent = math.floor(math.log10(magnitude))
    let power = 10.0 ** exponent
    return _nice_fraction(magnitude / power, round_it) * power
}

#: The 1, 2, 5 or 10 that stands in for a mantissa in `[1, 10)`.
fn _nice_fraction(fraction: float, round_it: bool) -> float {
    if round_it {
        if fraction < 1.5 { return 1.0 }
        if fraction < 3.0 { return 2.0 }
        if fraction < 7.0 { return 5.0 }
        return 10.0
    }
    if fraction <= 1.0 { return 1.0 }
    if fraction <= 2.0 { return 2.0 }
    if fraction <= 5.0 { return 5.0 }
    return 10.0
}

#: Drop the padding zeroes C's `%e` puts in an exponent, so 1.00e+05 reads as
#: 1.00e5 the way an axis label should.
fn _trim_exponent(text: str) -> str {
    # `find` reports absence as -1, not null.
    let marker = text.find("e")
    if marker < 0 { return text }
    let cut = marker
    let mantissa = text[0:cut]
    var exponent = text[cut + 1:]
    var negative = false
    if exponent.starts_with("-") {
        negative = true
        exponent = exponent[1:]
    } elif exponent.starts_with("+") {
        exponent = exponent[1:]
    }
    while exponent.len() > 1 and exponent.starts_with("0") { exponent = exponent[1:] }
    let sign = if negative { "-" } else { "" }
    return f"{mantissa}e{sign}{exponent}"
}

#: Reject the intervals no tick rule can answer for.
fn _check_interval(low: float, high: float, count_hint: int) -> void {
    if not math.is_finite(low) or not math.is_finite(high) {
        throw ValueError(f"tick limits must be finite, got {low} to {high}")
    }
    if low > high {
        throw ValueError(f"tick limits must be ordered, got a low of {low} above a high of {high}")
    }
    if count_hint < 2 {
        throw ValueError(f"a tick count hint must be at least 2, got {count_hint}")
    }
}
