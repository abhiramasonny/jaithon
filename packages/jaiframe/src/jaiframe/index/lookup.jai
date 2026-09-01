#: The three plans, the object that holds one, and the questions it answers.
#:
#: `IndexLookup.of` decides between arithmetic, a bisect and a hash table from
#: the labels themselves, and `index_order` is the single pass over adjacent
#: pairs that both that decision and the monotonic predicates read.
from std.gpu import Buffer
from ..column import Column, Dtype, Index, Value
from ..column import iota
from .kernels import PAIR_GROUP, SETTLE_ROUNDS, TALLY_GROUP, TALLY_GROUPS
from .kernels import fill_u32, flag, kernel_for, mask_column, require_device
from .kernels import table_capacity, tally, u32, valid_lane, value_lane, wide_lane
from .labels import aligned_levels, borrowed_levels, index_of_key
from .labels import index_of_key_columns, level_dtype, recoded_like
from .labels import release_all, release_levels, spell_key

#: The label position that stands for "no such label", which is what `take`
#: reads as a null and what a caller tests an indexer against.
pub const NOT_FOUND = -1

#: Where a tie goes in `searchsorted`: before the run of equal labels.
pub const SIDE_LEFT = "left"
#: Where a tie goes in `searchsorted`: after the run of equal labels.
pub const SIDE_RIGHT = "right"

#: Arithmetic: the labels are `start + i * step` and a lookup is a division.
pub const PLAN_RANGE = 0
#: Bisection over a monotonic increasing single level.
pub const PLAN_SORTED = 1
#: An open-addressed table over the per-row key hash.
pub const PLAN_HASHED = 2

#: How each adjacent pair of labels is ordered, and what that says about the
#: index as a whole.
#:
#: Returns the comparison lane — which `PLAN_SORTED` keeps, because the runs of
#: equal labels it marks are also the answer to `unique` — together with
#: whether every pair rises, whether every pair falls, and how many tie. The
#: caller owns the lane.
fn index_order(index: Index) -> (Buffer, bool, bool, int) {
    if index.is_range() {
        throw ValueError("a range index is ordered by its step, not by comparing labels")
    }
    require_device("order an index")
    let pairs = if index.len() > 0 { index.len() - 1 } else { 0 }
    let cmp = Buffer(if pairs > 0 { pairs } else { 1 })
    cmp.fill_zero()
    if pairs == 0 { return (cmp, true, true, 0) }
    let groups = (pairs + PAIR_GROUP - 1) // PAIR_GROUP
    for level in index.levels {
        kernel_for("idx_pair_cmp").dispatch_async(
            [value_lane(level), wide_lane(level), valid_lane(level), cmp],
            groups * PAIR_GROUP,
            [pairs, flag(level.dtype.is_wide()), flag(level.has_validity())],
            group_size: PAIR_GROUP
        )
    }
    let partials = Buffer(3 * TALLY_GROUPS)
    kernel_for("idx_order_tally").dispatch_async(
        [cmp, partials],
        TALLY_GROUPS * TALLY_GROUP,
        [pairs],
        group_size: TALLY_GROUP
    )
    let counts = partials.download()
    partials.free()
    var rising = 0.0
    var falling = 0.0
    var tied = 0.0
    for slot in 0..TALLY_GROUPS {
        rising += counts[slot]
        falling += counts[TALLY_GROUPS + slot]
        tied += counts[2 * TALLY_GROUPS + slot]
    }
    return (cmp, int(rising) == pairs, int(falling) == pairs, int(tied))
}

#: Whether the labels never decrease, which is what pandas means by monotonic
#: increasing: `[1, 1, 2]` is one and `[1, 0]` is not.
#:
#: A multi-index is compared lexicographically. A null orders against nothing,
#: so an index that has one is neither increasing nor decreasing.
pub fn is_monotonic_increasing(index: Index) -> bool {
    if index.is_range() { return index.len() <= 1 or index.step > 0 }
    let order = index_order(index)
    order[0].free()
    return order[1]
}

#: Whether the labels never increase.
pub fn is_monotonic_decreasing(index: Index) -> bool {
    if index.is_range() { return index.len() <= 1 or index.step < 0 }
    let order = index_order(index)
    order[0].free()
    return order[2]
}

#: Whether no label is repeated.
#:
#: Costs whatever the lookup for this index costs — nothing for a range index,
#: one pass over the comparison lane for a monotonic one, and the hash table
#: for anything else. Hold an `IndexLookup` when the answer is wanted beside
#: other lookups over the same index.
pub fn is_unique(index: Index) -> bool {
    let lookup = IndexLookup.of(index)
    defer { lookup.free() }
    return lookup.is_unique()
}

#: A prepared way of asking one index where a label sits.
#:
#: Held rather than hidden because building it is the expensive half of a
#: lookup and probing it is the cheap half: a join, a set operation and a
#: reindex all probe the same index several times, and each of them builds one
#: of these and reuses it. `get_loc` and friends build one per call, which is
#: the right cost for a single question and the wrong one for a loop.
#:
#: Which plan it takes is decided here, once, from the labels themselves. The
#: caller never chooses.
#:
#: ```
#: let lookup = IndexLookup.of(frame.index)
#: defer { lookup.free() }
#: let positions = lookup.locate(target)
#: ```
pub class IndexLookup {
    #: `PLAN_RANGE`, `PLAN_SORTED` or `PLAN_HASHED`.
    pub let plan: int
    #: How many labels the index carries.
    pub let length: int
    #: How many distinct labels it carries.
    pub let distinct: int

    var levels: list[Column]
    var order: Buffer?
    var hashes: Buffer?
    var slots: Buffer?
    var rep: Buffer?
    var settled: Buffer?
    var mask: int
    var start: int
    var step: int
    var freed: bool

    fn init(self) {
        self.plan = PLAN_RANGE
        self.length = 0
        self.distinct = 0
        self.levels = []
        self.order = null
        self.hashes = null
        self.slots = null
        self.rep = null
        self.settled = null
        self.mask = 0
        self.start = 0
        self.step = 1
        self.freed = false
    }

    #: Prepare a lookup over `index`.
    #:
    #: The level columns are borrowed, not copied: a lookup lives inside one
    #: operation and the index outlives it. Freeing the index while a lookup
    #: over it is still live is the one thing that breaks that.
    pub static fn of(index: Index) -> IndexLookup {
        require_device("look a label up")
        let lookup = IndexLookup()
        lookup.length = index.len()
        lookup.start = index.start
        lookup.step = index.step
        if index.is_range() {
            lookup.plan = PLAN_RANGE
            lookup.distinct = index.len()
            return lookup
        }
        lookup.levels = [level for level in index.levels]
        if index.levels.len() == 1 {
            let ordered = index_order(index)
            #: A null makes a pair unorderable, so an index with one is never
            #: reported as rising — except an index of a single row, which has
            #: no pair at all. The bisect reads the value lane without a mask,
            #: so that one row has to be ruled out here.
            if ordered[1] and index.levels[0].null_count() == 0 {
                lookup.plan = PLAN_SORTED
                lookup.order = ordered[0]
                lookup.distinct = index.len() - ordered[3]
                return lookup
            }
            ordered[0].free()
        }
        lookup.plan = PLAN_HASHED
        lookup.build()
        return lookup
    }

    #: Whether no label is repeated.
    pub fn is_unique(self) -> bool { return self.distinct == self.length }

    #: How many distinct labels the index carries.
    pub fn nunique(self) -> int { return self.distinct }

    #: True at each row that is the first to carry its label.
    #:
    #: The mask `unique` compresses by, and the one a caller wanting the
    #: duplicates inverts.
    pub fn unique_mask(self) -> Column {
        self.check_live()
        let mask = mask_column(self.length)
        if self.length == 0 { return mask }
        if self.plan == PLAN_RANGE {
            mask.values.fill(1.0)
            return mask
        }
        if self.plan == PLAN_SORTED {
            #: A monotonic index puts every repeat of a label next to the
            #: first of them, so the runs the comparison lane already marks are
            #: the answer and no table is needed to find them.
            kernel_for("idx_run_starts").dispatch_async(
                [self.order ?? mask.values, mask.values],
                self.length,
                [self.length]
            )
            return mask
        }
        kernel_for("idx_first_of_kind").dispatch_async(
            [self.rep ?? mask.values, self.settled ?? mask.values, mask.values],
            self.length,
            [self.length]
        )
        return mask
    }

    #: Where each label of `target` sits in this index, `NOT_FOUND` for a miss.
    #:
    #: The lane is `target.len()` long and is exactly what `Column.take` with
    #: `allow_missing` consumes, which is why a reindex is this call plus one
    #: take per column.
    pub fn locate(self, target: Index) -> Buffer {
        self.check_live()
        let count = target.len()
        let out = Buffer(count)
        if count == 0 { return out }
        if self.plan == PLAN_RANGE {
            self.locate_over_range(target, out, count)
            return out
        }
        let aligned = aligned_levels(self.levels, target)
        let probes = aligned[0]
        if self.plan == PLAN_SORTED {
            self.locate_by_bisect(probes[0], out, count)
        } else {
            self.locate_by_table(probes, out, count)
        }
        release_levels(probes, aligned[1])
        return out
    }

    #: Where one label sits, raising `KeyError` when the index has no such row.
    pub fn locate_one(self, key: list[Value]) -> int {
        self.check_live()
        let target = index_of_key(key, self.level_dtypes())
        let positions = self.locate(target)
        let at = int(positions.download(0, 1)[0])
        positions.free()
        target.free()
        if at < 0 { throw KeyError(f"{spell_key(key)} is not in the index") }
        return at
    }

    #: Release the table and the lanes behind it. Freeing twice is harmless.
    #:
    #: The level columns are the index's, not the lookup's, and are left alone.
    pub fn free(self) -> void {
        if self.freed { return }
        self.freed = true
        for lane in [self.order, self.hashes, self.slots, self.rep, self.settled] {
            if lane is not null { lane.free() }
        }
    }

    #: Whether `free` has already run.
    pub fn is_freed(self) -> bool { return self.freed }

    fn level_dtypes(self) -> list[Dtype] {
        if self.plan == PLAN_RANGE { return [Dtype.Int64] }
        return self.levels.map(|level| level.dtype)
    }

    fn check_live(self) -> void {
        if self.freed { throw RuntimeError("this IndexLookup has already been released") }
    }

    #: Fold the levels into one hash per row and settle every row into a slot.
    fn build(self) -> void {
        let rows = self.length
        self.mask = table_capacity(rows) - 1
        let hashes = Buffer(if rows > 0 { rows } else { 1 })
        let slots = Buffer(self.mask + 1)
        let rep = Buffer(self.mask + 1)
        let settled = Buffer(if rows > 0 { rows } else { 1 })
        hashes.fill_zero()
        slots.fill_zero()
        self.hashes = hashes
        self.slots = slots
        self.rep = rep
        self.settled = settled
        if rows == 0 {
            self.distinct = 0
            return
        }
        for (at, level) in self.levels.enumerate() { level.hash_into(hashes, seed: at + 1) }
        let cursor = Buffer(rows)
        let candidate = Buffer(rows)
        let flags = Buffer(rows)
        let pending = Buffer(rows)
        let rowwise = iota(rows)
        kernel_for("idx_start_probe").dispatch_async([hashes, cursor], rows, [rows, self.mask])
        fill_u32(settled, rows, 0xFFFFFFFF)
        var round = 0
        loop {
            kernel_for("idx_insert").dispatch_async(
                [slots, hashes, cursor, settled, candidate],
                rows,
                [rows, self.mask]
            )
            flags.fill(1.0)
            for level in self.levels { level.equal_rows(candidate, level, rowwise, rows, flags) }
            kernel_for("idx_settle").dispatch_async(
                [flags, candidate, cursor, settled, pending],
                rows,
                [rows, self.mask]
            )
            if tally(pending, rows) == 0 { break }
            round += 1
            if round > SETTLE_ROUNDS {
                throw RuntimeError(
                    f"an index of {rows} labels did not settle into its table in {round} rounds"
                )
            }
        }
        fill_u32(rep, self.mask + 1, 0x7FFFFFFF)
        kernel_for("idx_elect").dispatch_async([rep, settled], rows, [rows])
        kernel_for("idx_first_of_kind").dispatch_async([rep, settled, flags], rows, [rows])
        self.distinct = tally(flags, rows)
        cursor.free()
        candidate.free()
        flags.free()
        pending.free()
        rowwise.free()
    }

    fn locate_over_range(self, target: Index, out: Buffer, count: int) -> void {
        let borrowed = borrowed_levels(target)
        let raw = borrowed[0]
        let built = borrowed[1]
        if raw.len() != 1 {
            release_all(raw, built)
            throw ValueError(f"a range index has one level, not {raw.len()}")
        }
        let probe = raw[0]
        if probe.dtype.is_dictionary() {
            release_all(raw, built)
            throw TypeError(f"a range index cannot hold a {probe.dtype.name()} label")
        }
        kernel_for("idx_range_locate").dispatch_async(
            [value_lane(probe), wide_lane(probe), valid_lane(probe), out],
            count,
            [
                count,
                self.length,
                u32(self.start),
                u32(self.step),
                flag(probe.dtype.is_wide()),
                flag(probe.has_validity()),
                0,
            ]
        )
        release_all(raw, built)
    }

    fn locate_by_bisect(self, probe: Column, out: Buffer, count: int) -> void {
        let keys = self.levels[0]
        kernel_for("idx_search").dispatch_async(
            [
                value_lane(keys),
                wide_lane(keys),
                value_lane(probe),
                wide_lane(probe),
                valid_lane(probe),
                out,
            ],
            count,
            [count, self.length, flag(keys.dtype.is_wide()), flag(probe.has_validity()), 2]
        )
    }

    fn locate_by_table(self, probes: list[Column], out: Buffer, count: int) -> void {
        let slots = self.slots ?? out
        let hashes = self.hashes ?? out
        let rep = self.rep ?? out
        let queries = Buffer(count)
        let cursor = Buffer(count)
        let candidate = Buffer(count)
        let flags = Buffer(count)
        let pending = Buffer(count)
        let rowwise = iota(count)
        queries.fill_zero()
        for (at, level) in probes.enumerate() { level.hash_into(queries, seed: at + 1) }
        kernel_for("idx_start_probe").dispatch_async([queries, cursor], count, [count, self.mask])
        #: -3 is "still walking". A slot the walk ran off the end of writes -1,
        #: which is the miss the caller reads, and both are below every row.
        out.fill(-3.0)
        var round = 0
        loop {
            kernel_for("idx_probe").dispatch_async(
                [slots, hashes, queries, cursor, out, candidate],
                count,
                [count, self.mask]
            )
            flags.fill(1.0)
            for (at, level) in self.levels.enumerate() {
                level.equal_rows(candidate, probes[at], rowwise, count, flags)
            }
            kernel_for("idx_accept").dispatch_async(
                [flags, candidate, rep, cursor, out, pending],
                count,
                [count, self.mask]
            )
            if tally(pending, count) == 0 { break }
            round += 1
            if round > SETTLE_ROUNDS {
                throw RuntimeError(f"a probe of {count} labels did not finish in {round} rounds")
            }
        }
        queries.free()
        cursor.free()
        candidate.free()
        flags.free()
        pending.free()
        rowwise.free()
    }
}

#: Where one label sits, raising `KeyError` when the index has no such row.
#:
#: Builds a lookup, asks it one question and throws it away. A loop over labels
#: should not be written at all — that is what `get_indexer` is — but if one
#: is, hold the `IndexLookup` across it.
pub fn get_loc(index: Index, label: Value) -> int {
    return get_loc_tuple(index, [label])
}

#: Where one multi-index label sits, one `Value` per level.
pub fn get_loc_tuple(index: Index, key: list[Value]) -> int {
    let lookup = IndexLookup.of(index)
    defer { lookup.free() }
    return lookup.locate_one(key)
}

#: Where each of `labels` sits, `NOT_FOUND` for the ones that are not there.
#:
#: One level only; a multi-index is asked through `get_indexer_for_index`.
pub fn get_indexer(index: Index, labels: list[Value]) -> Buffer {
    let target = index_of_key_columns([labels], [level_dtype(index, 0)])
    let positions = get_indexer_for_index(index, target)
    target.free()
    return positions
}

#: Where each label of `target` sits in `index`, `NOT_FOUND` for a miss.
#:
#: The direction is pandas': the result is `target.len()` long and holds
#: positions into `index`.
pub fn get_indexer_for_index(index: Index, target: Index) -> Buffer {
    let lookup = IndexLookup.of(index)
    defer { lookup.free() }
    return lookup.locate(target)
}

#: Where each of `values` would be inserted to keep `index` sorted.
#:
#: `side` decides which end of a run of equal labels the answer sits at:
#: `SIDE_LEFT` before it, `SIDE_RIGHT` after. The index has to be monotonic
#: increasing, because the answer is defined by that order and a bisect over
#: anything else is a wrong number rather than a slow one.
#:
#: Text compares by dictionary code, so a value the index has never seen sorts
#: past everything — its code is assigned after the index's.
pub fn searchsorted(index: Index, values: Column, side: str = SIDE_LEFT) -> Buffer {
    let mode = match side {
        "left" => 0,
        "right" => 1,
        _ => throw ValueError(f"searchsorted takes side left or right, not {side}"),
    }
    require_device("search an index")
    let count = values.len()
    let out = Buffer(if count > 0 { count } else { 1 })
    if count == 0 { return out }
    if index.is_range() {
        if index.step <= 0 {
            throw ValueError("searchsorted needs a monotonic increasing index")
        }
        if values.dtype.is_dictionary() {
            throw TypeError(f"a range index cannot be searched for a {values.dtype.name()}")
        }
        kernel_for("idx_range_locate").dispatch_async(
            [value_lane(values), wide_lane(values), valid_lane(values), out],
            count,
            [
                count,
                index.len(),
                u32(index.start),
                u32(index.step),
                flag(values.dtype.is_wide()),
                flag(values.has_validity()),
                mode + 1,
            ]
        )
        return out
    }
    if index.levels.len() != 1 {
        throw ValueError("searchsorted reads one level, not a whole multi-index")
    }
    if not is_monotonic_increasing(index) {
        throw ValueError("searchsorted needs a monotonic increasing index")
    }
    let keys = index.levels[0]
    let probe = recoded_like(values, keys)
    kernel_for("idx_search").dispatch_async(
        [
            value_lane(keys),
            wide_lane(keys),
            value_lane(probe),
            wide_lane(probe),
            valid_lane(probe),
            out,
        ],
        count,
        [count, index.len(), flag(keys.dtype.is_wide()), flag(probe.has_validity()), mode]
    )
    if probe != values { probe.free() }
    return out
}
