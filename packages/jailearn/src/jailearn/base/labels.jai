#: Arbitrary class values mapped onto the codes every classifier fits against.
import jaitensor as jt
import std.math
from std.os import env_or

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
    if _HASH_LABELS { return hashed_distinct(y) }
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
    for index in 0..count {
        #: The host sort raises a TypeError on a NaN; name the real problem.
        if math.is_nan(values[index]) { throw ValueError("a label is NaN, which is no class") }
        kept.push(values[index])
    }
    return merge_sort(kept)
}

#: `JAILEARN_HASH_LABELS=0` finds a label vector's distinct values by a
#: device bitonic sort and compaction (a dispatch per network stage, about a
#: hundred at ten thousand samples) instead of one pass into a hash table.
let _HASH_LABELS = env_or("JAILEARN_HASH_LABELS", "1") != "0"

#: Twice `MAX_CLASSES`, so a full set of classes leaves the table half empty
#: and a probe sequence stays short.
const LABEL_SLOTS: int = 8192

fn hashed_distinct(y: Tensor) -> list[float] {
    #: The table and its two counters share one allocation, so the answer
    #: comes back in a single read.
    let whole = Tensor([LABEL_SLOTS + 2])
    let table = whole.view(0, [LABEL_SLOTS])
    let counter = whole.view(LABEL_SLOTS, [2])
    base_kernel("base_distinct_reset").dispatch_async(
        [table.buffer(), counter.buffer()],
        LABEL_SLOTS,
        [LABEL_SLOTS]
    )
    base_kernel("base_distinct_hash").dispatch_async(
        [y.buffer(), table.buffer(), counter.buffer()],
        y.size,
        [y.size, LABEL_SLOTS]
    )
    table.free()
    counter.free()
    #: The table's bit patterns ARE the float values, and an empty slot's
    #: all-ones pattern reads back as a NaN. More distinct values than slots
    #: fills every slot, so the count alone catches overflow; the second
    #: counter says a sample was NaN, which is no class at all.
    let slots = whole.download()
    whole.free()
    var kept: list[float] = []
    for index in 0..LABEL_SLOTS {
        let value = slots[index]
        #: `value == value` is false only for a NaN, and unlike a call to
        #: `math.is_nan` it keeps this loop compiled.
        if value == value { kept.push(value) }
    }
    if kept.len() > MAX_CLASSES {
        #: A full table stopped counting: the labels hold at least this many.
        let many = if kept.len() >= LABEL_SLOTS {
            f"at least {kept.len()}"
        } else {
            f"{kept.len()}"
        }
        throw ValueError(f"{many} distinct values exceeds the {MAX_CLASSES} this supports")
    }
    if slots[LABEL_SLOTS + 1] != 0.0 { throw ValueError("a label is NaN, which is no class") }
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

    #: An index over `y` and `y`'s codes, for a fit. Every value of the
    #: vector the index was built from has a class by construction, and the
    #: hashed `distinct_values` has already refused a NaN, so the check
    #: `encode` makes -- a reduction and a wait on the device -- is skipped.
    pub static fn fit_codes(y: Tensor) -> tuple[LabelIndex, Tensor] {
        let index = LabelIndex.from_labels(y)
        if not _HASH_LABELS { return (index, index.encode(y)) }
        let out = Tensor(y.shape)
        index._map(y, out, 0)
        return (index, out)
    }

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
