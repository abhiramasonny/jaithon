#: The set operations over labels, and the two ways of asking whether two
#: indexes agree.
#:
#: Every one of them is `unique` plus one probe of an `IndexLookup`, and every
#: one of them keeps first-appearance order rather than sorting — which is
#: also the only order a union of text labels has.
from std.gpu import Buffer
from ..column import Column, Index
from ..column import concat_columns, iota
from .kernels import hit_mask, require_device, selected_positions, tally
from .labels import borrowed_levels, names_of, recoded_like
from .labels import release_all, shared_names
from .lookup import IndexLookup, get_indexer_for_index

#: The labels with the repeats dropped, keeping the first of each.
pub fn unique(index: Index) -> Index {
    let lookup = IndexLookup.of(index)
    defer { lookup.free() }
    if lookup.is_unique() { return index.copy() }
    let mask = lookup.unique_mask()
    defer { mask.free() }
    let picked = selected_positions(mask)
    defer { picked[0].free() }
    return index.take(picked[0], picked[1])
}

#: The two indexes stacked end to end, repeats and all.
#:
#: Text levels are put on one dictionary as they are stacked, which is what
#: makes the codes of the result mean one thing.
pub fn append(left: Index, right: Index) -> Index {
    if left.nlevels() != right.nlevels() {
        throw ValueError(f"cannot append {right.nlevels()} index levels onto {left.nlevels()}")
    }
    let here = borrowed_levels(left)
    let there = borrowed_levels(right)
    var levels: list[Column] = []
    for (at, part) in here[0].enumerate() {
        levels.push(concat_columns([part, there[0][at]]))
    }
    release_all(here[0], here[1])
    release_all(there[0], there[1])
    return Index.of(levels, shared_names(left, right))
}

#: Every label of either index, each once, in the order they first appear.
#:
#: pandas sorts a union when it can. This one does not: sorting is
#: `sort_index`, one module over, and a caller who wants sorted labels can say
#: so. Order that comes from the inputs is also the only order a union of text
#: labels has, since dictionary codes are interning order, not alphabet.
pub fn union(left: Index, right: Index) -> Index {
    let stacked = append(left, right)
    let deduped = unique(stacked)
    stacked.free()
    return deduped
}

#: The labels both indexes carry, each once, in the left index's order.
pub fn intersection(left: Index, right: Index) -> Index {
    return filtered_by_membership(left, right, true)
}

#: The labels the left index carries and the right does not, each once.
pub fn difference(left: Index, right: Index) -> Index {
    return filtered_by_membership(left, right, false)
}

#: The labels exactly one of the two carries, the left index's first.
pub fn symmetric_difference(left: Index, right: Index) -> Index {
    let only_left = difference(left, right)
    let only_right = difference(right, left)
    let joined = append(only_left, only_right)
    only_left.free()
    only_right.free()
    return joined
}

#: Whether the two label the same rows with the same labels.
#:
#: Names are not compared — `identical` is the one that compares those. Nor
#: are dtypes promoted: an `Int64` label and a `Float64` label of the same
#: value are not equal here, where pandas would promote them. A range index
#: does compare equal to the stored ramp it stands for, because that is the
#: comparison every alignment makes.
pub fn equals(left: Index, right: Index) -> bool {
    if left.len() != right.len() { return false }
    if left.nlevels() != right.nlevels() { return false }
    if left.is_range() and right.is_range() {
        return left.len() == 0 or left.start == right.start and left.step == right.step
    }
    if left.len() == 0 { return true }
    require_device("compare two indexes")
    let here = borrowed_levels(left)
    let there = borrowed_levels(right)
    let above = here[0]
    let below = there[0]
    #: Two labels of different dtypes are not equal, and asking `equal_rows`
    #: to compare them would raise rather than answer. pandas promotes an int
    #: label to a float one; the promotion has a precision cost, so it belongs
    #: to a caller who asked for it and not to a comparison.
    var comparable = true
    for (at, level) in above.enumerate() {
        if level.dtype.is_dictionary() != below[at].dtype.is_dictionary() {
            comparable = false
        } elif not level.dtype.is_dictionary() and level.dtype != below[at].dtype {
            comparable = false
        }
    }
    var same = false
    if comparable {
        let rowwise = iota(left.len())
        let flags = Buffer(left.len())
        flags.fill(1.0)
        for (at, level) in above.enumerate() {
            let probe = recoded_like(below[at], level)
            level.equal_rows(rowwise, probe, rowwise, left.len(), flags)
            if probe != below[at] { probe.free() }
        }
        same = tally(flags, left.len()) == left.len()
        flags.free()
        rowwise.free()
    }
    release_all(above, here[1])
    release_all(below, there[1])
    return same
}

#: Whether the two are the same index: the same labels, the same names, and
#: the same kind — a range index is not identical to the ramp it stands for.
pub fn identical(left: Index, right: Index) -> bool {
    if left.is_range() != right.is_range() { return false }
    if names_of(left) != names_of(right) { return false }
    if not left.is_range() {
        for (at, level) in left.levels.enumerate() {
            if level.dtype != right.levels[at].dtype { return false }
        }
    }
    return equals(left, right)
}

#: The unique labels of `left` that either are or are not in `right`.
fn filtered_by_membership(left: Index, right: Index, present: bool) -> Index {
    let deduped = unique(left)
    let positions = get_indexer_for_index(right, deduped)
    let mask = hit_mask(positions, deduped.len(), present)
    positions.free()
    let picked = selected_positions(mask)
    mask.free()
    let kept = deduped.take(picked[0], picked[1])
    picked[0].free()
    deduped.free()
    return kept
}
