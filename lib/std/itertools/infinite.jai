#: CPython's three infinite iterators: `count`, `cycle` and `repeat`.
#:
#: None of the three ever answers null on its own — `count` and `cycle` never
#: at all, `repeat` only once a finite `times` runs out — so every one of them
#: has to be bounded by `take`, `takewhile` or `islice` before a consumer such
#: as `collect` is allowed near it.
#:
#: ```
#: from std.itertools import count, cycle, repeat
#: count(10).take(3).collect()        # [10, 11, 12]
#: cycle("ab").take(5).collect()      # ["a", "b", "a", "b", "a"]
#: repeat(7, 3).collect()             # [7, 7, 7]
#: ```
import std.iter as lazy
from std.iter import Iter

class _Count extends Iter[any] {
    var value: any
    let step: any

    #: Count from `start` in steps of `step`, forever.
    pub fn init(self, start: any, step: any) {
        self.value = start
        self.step = step
    }

    pub fn next(self) -> any {
        let current = self.value
        self.value = self.value + self.step
        return current
    }
}

class _Repeat extends Iter[any] {
    let value: any
    let endless: bool
    var remaining: int

    #: Yield `value` `times` times, or forever when `times` is null.
    pub fn init(self, value: any, times: int?) {
        self.value = value
        self.endless = times is null
        self.remaining = (times ?? 0) < 0 ? 0 : times ?? 0
    }

    pub fn next(self) -> any {
        if self.endless { return self.value }
        if self.remaining <= 0 { return null }
        self.remaining -= 1
        return self.value
    }
}

#: Count from `start` forever in steps of `step`.
#:
#: Both arguments may be `int` or `float`, and the arithmetic is whatever the
#: pair of them does: an int start with a float step yields floats from the
#: first addition on, exactly as CPython's does. Unlike `std.iter.generators`'
#: `count`, a `step` of zero is allowed and repeats `start` forever, which is
#: what CPython accepts.
#:
#: ```
#: count().take(3).collect()          # [0, 1, 2]
#: count(2.5, 0.5).take(3).collect()  # [2.5, 3.0, 3.5]
#: count(10, -2).take(3).collect()    # [10, 8, 6]
#: ```
pub fn count(start: any = 0, step: any = 1) -> Iter[any] {
    _require_number(start, "start")
    _require_number(step, "step")
    return _Count(start, step)
}

#: Repeat an iterable's elements forever, remembering them as they pass.
#:
#: The source is pulled once and replayed from a buffer, so it need not be
#: re-iterable; an empty source stays empty rather than spinning. This is
#: `std.iter.cycle` under a CPython name, because the two already agree.
#:
#: ```
#: cycle([1, 2]).take(5).collect()   # [1, 2, 1, 2, 1]
#: cycle([]).collect()               # []
#: ```
pub fn cycle(iterable: any) -> Iter[any] {
    return lazy.cycle(iterable)
}

#: Yield `value` `times` times, or forever when `times` is null.
#:
#: A negative `times` yields nothing, as CPython's does — that is the one
#: difference from `std.iter.generators`' `repeat`, which raises on it.
#:
#: ```
#: repeat("x", 3).collect()      # ["x", "x", "x"]
#: repeat("x", -1).collect()     # []
#: repeat(0).take(2).collect()   # [0, 0]
#: ```
pub fn repeat(value: any, times: int? = null) -> Iter[any] {
    return _Repeat(value, times)
}

fn _require_number(value: any, name: str) -> void {
    let kind = type_of(value)
    if kind != "int" and kind != "float" {
        throw TypeError(f"count() {name} must be int or float, not '{kind}'")
    }
}
