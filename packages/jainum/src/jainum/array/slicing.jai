#: The objects a subscript is written with, and what a slice resolves to
#: against a real extent.
from std.core import Printable

#: A half-open range of one axis, with an optional step.
#:
#: `null` means "the end the step runs from": `sl(null, null, -1)` reverses.
#: Negative bounds count back from the end, as they do in numpy. Prefer a
#: `range` literal — `a[2..7]` — where the bounds are known and positive; this
#: exists for the open ends and the negative steps a `range` cannot spell.
pub class Slice: Printable {
    #: First index, or `null` for the natural start.
    pub let start: int?
    #: One past the last index, or `null` for the natural end.
    pub let stop: int?
    #: Distance between the indices taken. Never zero.
    pub let step: int

    pub fn init(self, start: int? = null, stop: int? = null, step: int = 1) {
        if step == 0 { throw ValueError("a slice step cannot be zero") }
        self.start = start
        self.stop = stop
        self.step = step
    }

    #: First index and count, clamped against an axis of `extent` elements.
    pub fn resolve(self, extent: int) -> (int, int) {
        let step = self.step
        var first = 0
        var last = 0
        if step > 0 {
            first = _clamp_bound(self.start ?? 0, extent, 0, extent)
            last = _clamp_bound(self.stop ?? extent, extent, 0, extent)
        } else {
            first = _clamp_bound(self.start ?? extent - 1, extent, -1, extent - 1)
            last = _clamp_bound(self.stop ?? -extent - 1, extent, -1, extent - 1)
        }
        let span = last - first
        if span == 0 or (span > 0) != (step > 0) { return (first, 0) }
        return (first, (span + step - _sign(step)) // step)
    }

    pub fn to_str(self) -> str {
        let start = if self.start is null { "" } else { f"{self.start}" }
        let stop = if self.stop is null { "" } else { f"{self.stop}" }
        return f"sl({start}, {stop}, {self.step})"
    }

    fn __str__(self) -> str { return self.to_str() }
}

pub fn _sign(value: int) -> int { return if value < 0 { -1 } else { 1 } }

fn _clamp_bound(value: int, extent: int, low: int, high: int) -> int {
    let shifted = if value < 0 { value + extent } else { value }
    return max(low, min(shifted, high))
}

#: `Slice(start, stop, step)`, spelled the way numpy's `s_` reads.
pub fn sl(start: int? = null, stop: int? = null, step: int = 1) -> Slice {
    return Slice(start, stop, step)
}

#: One of the two index sentinels. There is no reason to build another.
pub class IndexToken: Printable {
    pub let name: str

    pub fn init(self, name: str) { self.name = name }

    pub fn to_str(self) -> str { return self.name }

    fn __str__(self) -> str { return self.name }
}

#: Insert an axis of length one at this position, as `np.newaxis` does.
pub let NEWAXIS = IndexToken("newaxis")

#: Stand in for as many full axes as the key leaves over, as `...` does.
pub let ELLIPSIS = IndexToken("Ellipsis")
