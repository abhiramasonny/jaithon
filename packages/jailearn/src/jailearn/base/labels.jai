#: Arbitrary class values mapped onto the codes every classifier fits against.
import jaitensor as jt

from jaitensor import Tensor
from std.algo.sort import merge_sort

from .kernels import base_kernel
from .reduce import sort_columns

#: The most distinct class values `distinct_values` will report.
pub const MAX_CLASSES: int = 4096

#: The distinct values of a vector, ascending.
#:
#: Sorted on the device and compacted with an atomic counter, so the host only
#: touches the answer -- at most `MAX_CLASSES` long -- and never the samples.
pub fn distinct_values(y: Tensor) -> list[float] {
    if y.size == 0 { throw ValueError("cannot take the distinct values of an empty vector") }
    let column = y.view(0, [y.size, 1])
    let sorted = sort_columns(column)
    column.free()
    let counter = Tensor.zeros([1])
    let found = Tensor([MAX_CLASSES])
    let total = Tensor([1])
    base_kernel("base_unique_from_sorted").dispatch_async(
        [sorted.buffer(), counter.buffer(), found.buffer()],
        y.size,
        [y.size, MAX_CLASSES]
    )
    base_kernel("base_counter_value").dispatch_async([counter.buffer(), total.buffer()], 1, [])
    let count = int(total.download()[0] + 0.5)
    let values = found.download()
    sorted.free()
    counter.free()
    found.free()
    total.free()
    if count > MAX_CLASSES {
        throw ValueError(f"{count} distinct values exceeds the {MAX_CLASSES} this supports")
    }
    var kept: list[float] = []
    for index in 0..count { kept.push(values[index]) }
    return merge_sort(kept)
}

#: Arbitrary class values mapped onto the codes `0..k-1` and back.
#:
#: Every classifier fits against codes, `StratifiedKFold` counts them and the
#: metrics compare them, so the mapping lives here once and each of those takes
#: a `LabelIndex` rather than deriving its own.
pub class LabelIndex {
    #: The class values, ascending. Position `i` is code `i`.
    pub let classes: list[float]
    let table: Tensor

    #: Build an index over class values that are already known.
    pub fn init(self, classes: list[float]) {
        if classes.len() == 0 { throw ValueError("a LabelIndex needs at least one class") }
        self.classes = [value for value in classes]
        self.table = Tensor([classes.len()], self.classes)
    }

    #: Build an index over whatever values a label vector holds.
    pub static fn from_labels(y: Tensor) -> LabelIndex {
        return LabelIndex(distinct_values(y))
    }

    #: How many classes there are.
    pub fn count(self) -> int { return self.classes.len() }

    #: The class values as a `[k]` device tensor, for kernels that need them.
    pub fn class_tensor(self) -> Tensor { return self.table }

    #: Class values to codes `0..k-1`.
    pub fn encode(self, y: Tensor) -> Tensor {
        let out = Tensor(y.shape)
        self._map(y, out, 0)
        #: A value in no class encodes to -1, and a silent -1 turns into a
        #: wrong class much further downstream, so it is caught here for the
        #: price of one pass and one scalar read. Negated and maximised rather
        #: than minimised because that is the reduction jaitensor spreads over
        #: threadgroups; the minimum would be one thread walking the samples.
        let flipped = jt.neg(out)
        let peak = jt.amax(flipped)
        let lowest = -peak.download()[0]
        flipped.free()
        peak.free()
        if lowest < 0.0 {
            out.free()
            throw ValueError(f"a label is outside the {self.count()} known classes")
        }
        return out
    }

    #: Codes `0..k-1` back to class values.
    pub fn decode(self, codes: Tensor) -> Tensor {
        let out = Tensor(codes.shape)
        self._map(codes, out, 1)
        return out
    }

    fn _map(self, input: Tensor, out: Tensor, decode: int) -> void {
        base_kernel("base_map_labels").dispatch_async(
            [input.buffer(), self.table.buffer(), out.buffer()],
            input.size,
            [input.size, self.count(), decode]
        )
    }

    #: Release the device copy of the class table. Freeing twice is harmless.
    pub fn free(self) -> void { self.table.free() }

    fn __str__(self) -> str { return f"LabelIndex(classes={self.classes})" }
}
