#: Sequence generators: iterators built from scratch rather than by adapting
#: one that already exists.
#:
#: Everything here answers "start me a sequence" — `count`, `repeat`, `once`,
#: `empty`, `successors` and `range_step` — as opposed to `std.iter`'s adapters,
#: which all take an existing `Iter` and wrap it.
#:
#: `std.iter`'s own `iter`, `cycle` and `from_fn` look like generators too, but
#: they stay there rather than moving here: `cycle` and `from_fn` share private
#: helper classes with `Iter.cycle` and `Unzip`, and `iter` is called from
#: inside `Iter.chain`, `zip`, `interleave` and `cross`. Moving any of the three
#: out would force `std.iter` to import them back from here to keep serving
#: those methods, and `std.iter` is exactly what this module already imports
#: `Iter` from — a two-file cycle the compiler refuses as `E0801`.
#:
#: ```
#: from std.iter.generators import count, range_step
#: count(0).take(3).collect()        # [0, 1, 2]
#: range_step(0, 10, 3).collect()    # [0, 3, 6, 9]
#: ```
from std.iter import Iter

class _Count extends Iter[int] {
    var value: int
    let step: int

    #: Count from `start` in steps of `step`.
    pub fn init(self, start: int, step: int) {
        self.value = start
        self.step = step
    }

    pub fn next(self) -> int? {
        let current = self.value
        self.value += self.step
        return current
    }
}

class _Repeat extends Iter[any] {
    let value: any
    var remaining: int

    #: Yield `value` `remaining` times; a negative count means forever.
    pub fn init(self, value: any, remaining: int) {
        self.value = value
        self.remaining = remaining
    }

    pub fn next(self) -> any {
        if self.remaining == 0 { return null }
        if self.remaining > 0 { self.remaining -= 1 }
        return self.value
    }
}

class _Once extends Iter[any] {
    var value: any

    #: Yield `value` and then nothing.
    pub fn init(self, value: any) {
        self.value = value
    }

    pub fn next(self) -> any {
        let taken = self.value
        self.value = null
        return taken
    }
}

class _Empty extends Iter[any] {
    pub fn next(self) -> any {
        return null
    }
}

class _Successors extends Iter[any] {
    let f: fn(any) -> any
    var current: any

    #: Yield `seed` and then the iterates of `f`.
    pub fn init(self, seed: any, f: fn(any) -> any) {
        self.current = seed
        self.f = f
    }

    pub fn next(self) -> any {
        let value = self.current
        if value is null { return null }
        self.current = self.f(value)
        return value
    }
}

class _RangeStep extends Iter[int] {
    var value: int
    let stop: int
    let step: int

    #: Step from `start` towards `stop` by `step`.
    pub fn init(self, start: int, stop: int, step: int) {
        self.value = start
        self.stop = stop
        self.step = step
    }

    pub fn next(self) -> int? {
        if self.step > 0 and self.value >= self.stop { return null }
        if self.step < 0 and self.value <= self.stop { return null }
        let current = self.value
        self.value += self.step
        return current
    }
}

#: Count from `start` forever in steps of `step`.
#:
#: ```
#: count(1).take(4).collect()        # [1, 2, 3, 4]
#: count(10, -2).take(3).collect()   # [10, 8, 6]
#: ```
pub fn count(start: int = 0, step: int = 1) -> Iter[int] {
    if step == 0 { throw ValueError("step must be non-zero, got 0") }
    return _Count(start, step)
}

#: Yield `value` `times` times, or forever when `times` is null.
#:
#: ```
#: repeat("x", 3).collect()        # ["x", "x", "x"]
#: repeat(0).take(2).collect()     # [0, 0]
#: ```
pub fn repeat(value: any, times: int? = null) -> Iter[any] {
    if times is not null and times < 0 {
        throw ValueError(f"times must be non-negative, got {times}")
    }
    return _Repeat(value, times ?? -1)
}

#: Yield `value` exactly once.
#:
#: ```
#: once(7).collect()   # [7]
#: ```
pub fn once(value: any) -> Iter[any] {
    return _Once(value)
}

#: Yield nothing.
#:
#: Useful as the identity for `chain` and as a base case in recursion.
#:
#: ```
#: empty().collect()   # []
#: ```
pub fn empty() -> Iter[any] {
    return _Empty()
}

#: Yield `seed`, then `f` of it, and so on until `f` returns null.
#:
#: ```
#: successors(1, |x| if x < 100 { x * 2 } else { null }).collect()
#: # [1, 2, 4, 8, 16, 32, 64, 128]
#: ```
pub fn successors(seed: any, f: fn(any) -> any) -> Iter[any] {
    return _Successors(seed, f)
}

#: Count from `start` towards `stop` in steps of `step`, excluding `stop`.
#:
#: The lazy equivalent of a `..` range with a stride, and unlike `iter` over a
#: range it never materialises its elements.
#:
#: ```
#: range_step(0, 10, 3).collect()   # [0, 3, 6, 9]
#: range_step(5, 0, -2).collect()   # [5, 3, 1]
#: ```
pub fn range_step(start: int, stop: int, step: int) -> Iter[int] {
    if step == 0 { throw ValueError("step must be non-zero, got 0") }
    return _RangeStep(start, stop, step)
}
