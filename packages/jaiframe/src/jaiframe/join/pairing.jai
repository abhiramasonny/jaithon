#: The hash table, the radix passes, and the two position lanes they produce.
#:
#: `pair_rows` is the middle of `join`: build a bucket table over the shorter
#: side, probe it with the longer one, and hand back a `Pairing`. The stable
#: radix order the table is grouped by is the same one `key_order` sorts a
#: finished pairing with, so both live here.
from std.gpu import Buffer
from ..column import Column, Dtype, MAX_ROWS
from ..column import cached_kernel, concat_columns, iota, prefix_sum, unify_levels
from .kernels import MIN_BUCKETS, flag, kernel_for, require_device
from .types import How

#: How many bits it takes to write `value`.
pub fn bit_width(value: int) -> int {
    var bits = 0
    var rest = value
    while rest > 0 {
        bits += 1
        rest = rest >> 1
    }
    return bits
}

#: The power-of-two bucket count a build side of `rows` gets.
#:
#: One bucket per build row, so the chain a probe row walks is one entry in
#: expectation. Two per row would halve the collisions and double a table that
#: is already three lanes wide; at load one the extra candidates are cleared by
#: the same equality pass that has to run anyway.
fn bucket_count(rows: int) -> int {
    var size = MIN_BUCKETS
    while size < rows { size = size * 2 }
    return size
}

pub fn copy_lane(src: Buffer, dst: Buffer, count: int) -> void {
    if count == 0 { return }
    cached_kernel("col_copy_lane").dispatch_async([src, dst], count, [count, 0, 0])
}

#: A lane of hash bits copied as bits.
#:
#: `col_copy_lane` moves the same bytes but reads them as floats on the way,
#: and a hash is as likely as not to be a signalling NaN pattern that a float
#: move is allowed to quieten.
fn copy_uint(src: Buffer, dst: Buffer, count: int) -> void {
    if count == 0 { return }
    kernel_for("jn_copy_uint").dispatch_async([src, dst], count, [count])
}

#: One hash per row, folding every key column into the same lane.
fn hash_lane(keys: list[Column], rows: int, seed: int) -> Buffer {
    let lane = Buffer(rows)
    lane.fill_zero()
    for key in keys { key.hash_into(lane, seed) }
    return lane
}

#: A stable ascending ordering of `count` entries by an unsigned key lane.
#:
#: One binary LSD pass per significant bit: a scan of the "this bit is clear"
#: flags, then a scatter that keeps the clear entries in scanned order and puts
#: the set ones after them. Stability is the property everything here leans on
#: — ordering build rows by bucket leaves each bucket in row order, and
#: ordering output pairs by their left row leaves each left row's matches in
#: right order — so neither caller needs a second key.
#:
#: `bits` comes from the range the keys actually span, so a join against a
#: hundred-row table pays seven passes rather than thirty-two. The scan hands
#: its total back to the host, which is one four-byte read per pass and the
#: only place this waits on the device.
#:
#: Returns fresh `(order, keys)` lanes the caller owns; the inputs are untouched.
pub fn radix_order(order_in: Buffer, keys_in: Buffer, count: int, bits: int) -> (Buffer, Buffer) {
    var order_front = Buffer(count)
    var keys_front = Buffer(count)
    copy_lane(order_in, order_front, count)
    copy_uint(keys_in, keys_front, count)
    if count <= 1 or bits <= 0 { return (order_front, keys_front) }
    var order_back = Buffer(count)
    var keys_back = Buffer(count)
    let flags = Buffer(count)
    for bit in 0..bits {
        kernel_for("jn_radix_flags").dispatch_async([keys_front, flags], count, [count, bit])
        let scanned = prefix_sum(flags, count)
        kernel_for("jn_radix_scatter").dispatch_async(
            [order_front, keys_front, flags, scanned[0], order_back, keys_back],
            count,
            [count]
        )
        scanned[0].free()
        let spare_order = order_front
        let spare_keys = keys_front
        order_front = order_back
        keys_front = keys_back
        order_back = spare_order
        keys_back = spare_keys
    }
    flags.free()
    order_back.free()
    keys_back.free()
    return (order_front, keys_front)
}

#: The bucket table one side of a join is built into.
#:
#: `rows_of` holds the build rows grouped by bucket and ascending inside each
#: one; `starts` and `ends` are the half-open span of each bucket. A bucket
#: nothing landed in reads back as [0, 0), so the probe needs no emptiness test.
class BuildTable {
    #: How many buckets, always a power of two.
    pub var buckets: int
    #: How many build rows the table holds.
    pub var rows: int
    #: First entry of each bucket.
    pub var starts: Buffer
    #: One past the last entry of each bucket.
    pub var ends: Buffer
    #: Build rows, grouped by bucket.
    pub var rows_of: Buffer

    #: Adopt the lanes. `build_table` is the only correct caller.
    pub fn init(self, buckets: int, rows: int, starts: Buffer, ends: Buffer, rows_of: Buffer) {
        self.buckets = buckets
        self.rows = rows
        self.starts = starts
        self.ends = ends
        self.rows_of = rows_of
    }

    #: Release the lanes.
    pub fn free(self) -> void {
        self.starts.free()
        self.ends.free()
        self.rows_of.free()
    }
}

fn build_table(keys: list[Column], rows: int) -> BuildTable {
    let buckets = bucket_count(rows)
    let starts = Buffer(buckets)
    let ends = Buffer(buckets)
    starts.fill_zero()
    ends.fill_zero()
    if rows == 0 { return BuildTable(buckets, rows, starts, ends, Buffer(0)) }
    let hashes = hash_lane(keys, rows, 0)
    let lane = Buffer(rows)
    kernel_for("jn_bucket_of").dispatch_async([hashes, lane], rows, [rows, buckets - 1])
    hashes.free()
    let base = iota(rows)
    let sorted = radix_order(base, lane, rows, bit_width(buckets - 1))
    base.free()
    lane.free()
    kernel_for("jn_bucket_bounds").dispatch_async([sorted[1], starts, ends], rows, [rows])
    sorted[1].free()
    return BuildTable(buckets, rows, starts, ends, sorted[0])
}

#: The two position lanes a join produces.
#:
#: One entry per output row on each side, `MISSING_POSITION` where that side
#: contributes nothing. Every result column is one `take` off these.
pub class Pairing {
    #: The left row each output row came from.
    pub var left: Buffer
    #: The right row each output row came from.
    pub var right: Buffer
    #: How many output rows.
    pub var rows: int

    #: Adopt two lanes of `rows` positions.
    pub fn init(self, left: Buffer, right: Buffer, rows: int) {
        self.left = left
        self.right = right
        self.rows = rows
    }

    #: Release both lanes.
    pub fn free(self) -> void {
        self.left.free()
        self.right.free()
    }
}

fn cross_pairing(n_left: int, n_right: int) -> Pairing {
    let rows = n_left * n_right
    if rows > MAX_ROWS {
        throw ValueError(
            f"a cross join of {n_left} by {n_right} is more than the {MAX_ROWS} rows a position lane addresses"
        )
    }
    let left = Buffer(rows)
    let right = Buffer(rows)
    if rows > 0 {
        kernel_for("jn_cross_pairs").dispatch_async([left, right], rows, [rows, n_right])
    }
    return Pairing(left, right, rows)
}

pub fn permute_lane(lane: Buffer, perm: Buffer, rows: int) -> void {
    let scratch = Buffer(rows)
    kernel_for("jn_gather_lane").dispatch_async([lane, perm, scratch], rows, [rows])
    copy_lane(scratch, lane, rows)
    scratch.free()
}

#: Put the output rows back in the ordering side's row order, in place.
#:
#: The probe side comes out in its own order and pandas reports the left
#: frame's — the right frame's for a right join. A stable pass by that side's
#: position fixes both at once: rows the ordering side does not name take a key
#: one past its last row and so trail everything, and the pass's stability
#: leaves each ordering row's matches in the other side's order.
fn reorder_pairing(pairing: Pairing, by_right: bool, n_left: int, n_right: int) -> void {
    let rows = pairing.rows
    if rows <= 1 { return }
    let lane = if by_right { pairing.right } else { pairing.left }
    let absent = if by_right { n_right } else { n_left }
    let keys = Buffer(rows)
    kernel_for("jn_order_key").dispatch_async([lane, keys], rows, [rows, absent])
    let base = iota(rows)
    let sorted = radix_order(base, keys, rows, bit_width(absent))
    base.free()
    keys.free()
    sorted[1].free()
    permute_lane(pairing.left, sorted[0], rows)
    permute_lane(pairing.right, sorted[0], rows)
    sorted[0].free()
}

#: The rows of two frames paired up on their key columns.
#:
#: The table goes over whichever side is shorter and the other side probes it;
#: the result is then put back into the order `how` reports in. This is the one
#: place a merge, an index join and a `concat` alignment all go through.
pub fn pair_rows(
    left_keys: list[Column],
    right_keys: list[Column],
    n_left: int,
    n_right: int,
    how: How = How.Inner
) -> Pairing {
    require_device("join two frames")
    if how == How.Cross { return cross_pairing(n_left, n_right) }
    if left_keys.len() == 0 or left_keys.len() != right_keys.len() {
        throw ValueError(
            f"a join needs the same number of key columns on each side, not {left_keys.len()} and {right_keys.len()}"
        )
    }
    for (at, key) in left_keys.enumerate() {
        if key.dtype != right_keys[at].dtype {
            throw TypeError(
                f"key {at} is {key.dtype.name()} on the left and {right_keys[at].dtype.name()} on the right"
            )
        }
    }

    let build_left = n_left <= n_right
    let build_keys = if build_left { left_keys } else { right_keys }
    let probe_keys = if build_left { right_keys } else { left_keys }
    let n_build = if build_left { n_left } else { n_right }
    let n_probe = if build_left { n_right } else { n_left }
    let keep_build = if build_left { how.keeps_left() } else { how.keeps_right() }
    let keep_probe = if build_left { how.keeps_right() } else { how.keeps_left() }

    let table = build_table(build_keys, n_build)
    defer { table.free() }

    let probe_hashes = hash_lane(probe_keys, n_probe, 0)
    let probe_buckets = Buffer(n_probe)
    kernel_for("jn_bucket_of").dispatch_async(
        [probe_hashes, probe_buckets],
        n_probe,
        [n_probe, table.buckets - 1]
    )
    probe_hashes.free()

    let counts = Buffer(n_probe)
    kernel_for("jn_probe_counts").dispatch_async(
        [probe_buckets, table.starts, table.ends, counts],
        n_probe,
        [n_probe]
    )
    let scanned = prefix_sum(counts, n_probe)
    counts.free()
    let candidates = scanned[1]

    let cand_probe = Buffer(candidates)
    let cand_build = Buffer(candidates)
    kernel_for("jn_probe_pairs").dispatch_async(
        [probe_buckets, table.starts, table.ends, table.rows_of, scanned[0], cand_probe, cand_build],
        n_probe,
        [n_probe]
    )
    probe_buckets.free()

    #: Every candidate starts a match and each key column can only clear it, so
    #: chaining the columns leaves the pairs that agree on all of them. That is
    #: also what keeps a bucket collision from becoming a join.
    let matched = Buffer(candidates)
    if candidates > 0 { matched.fill(1.0) }
    for (at, key) in probe_keys.enumerate() {
        key.equal_rows(cand_probe, build_keys[at], cand_build, candidates, matched)
    }
    cand_probe.free()

    let sizes = Buffer(n_probe)
    kernel_for("jn_emit_sizes").dispatch_async(
        [scanned[0], matched, sizes],
        n_probe,
        [n_probe, flag(keep_probe)]
    )
    let emitted = prefix_sum(sizes, n_probe)
    sizes.free()
    let probe_rows = emitted[1]

    var extra = 0
    var unmatched: Buffer? = null
    var unmatched_at: Buffer? = null
    if keep_build and n_build > 0 {
        let seen = Buffer(n_build)
        seen.fill_zero()
        kernel_for("jn_mark_matched").dispatch_async(
            [cand_build, matched, seen],
            candidates,
            [candidates]
        )
        let flags = Buffer(n_build)
        kernel_for("jn_unmatched_flags").dispatch_async([seen, flags], n_build, [n_build])
        seen.free()
        let leftover = prefix_sum(flags, n_build)
        unmatched = flags
        unmatched_at = leftover[0]
        extra = leftover[1]
    }

    let rows = probe_rows + extra
    if rows > MAX_ROWS {
        throw ValueError(
            f"this join produces {rows} rows, past the {MAX_ROWS} a position lane addresses"
        )
    }
    let probe_lane = Buffer(rows)
    let build_lane = Buffer(rows)
    kernel_for("jn_emit_pairs").dispatch_async(
        [scanned[0], matched, cand_build, emitted[0], probe_lane, build_lane],
        n_probe,
        [n_probe, flag(keep_probe)]
    )
    let leftover_flags = unmatched
    let leftover_at = unmatched_at
    if leftover_flags is not null and leftover_at is not null {
        kernel_for("jn_append_unmatched").dispatch_async(
            [leftover_flags, leftover_at, probe_lane, build_lane],
            n_build,
            [n_build, probe_rows]
        )
        leftover_flags.free()
        leftover_at.free()
    }
    scanned[0].free()
    emitted[0].free()
    cand_build.free()
    matched.free()

    let pairing = if build_left {
        Pairing(build_lane, probe_lane, rows)
    } else {
        Pairing(probe_lane, build_lane, rows)
    }
    let by_right = how == How.Right
    let needs_order = if by_right { not build_left } else { build_left }
    if needs_order { reorder_pairing(pairing, by_right, n_left, n_right) }
    return pairing
}

#: The column names an `on` / `left_on` / `right_on` argument spells.
pub fn names_of(spec: any?, what: str) -> list[str]? {
    if spec is null { return null }
    if spec is str { return [spec] }
    if spec is list {
        var out: list[str] = []
        for entry in spec {
            if not entry is str { throw TypeError(f"{what} must name columns") }
            out.push(entry)
        }
        if out.len() == 0 { throw ValueError(f"{what} names no columns") }
        return out
    }
    throw TypeError(f"{what} must be a column name or a list of them")
}

#: The key of every output row, taken from whichever side supplied one.
#:
#: The two key columns are stacked into one and read with a single `take`, so
#: an outer join's right-only rows get their key without a second pass and
#: without a kernel that has to know what a dtype's lanes are. Stacking also
#: puts two dictionary columns on one table, which they have to be for the
#: codes to mean the same thing.
pub fn merged_key_column(left_key: Column, right_key: Column, pairing: Pairing) -> Column {
    let stacked = concat_columns([left_key, right_key])
    defer { stacked.free() }
    let lane = Buffer(pairing.rows)
    kernel_for("jn_key_positions").dispatch_async(
        [pairing.left, pairing.right, lane],
        pairing.rows,
        [pairing.rows, left_key.length]
    )
    let out = stacked.take(lane, pairing.rows)
    lane.free()
    return out
}

#: pandas' `_merge` column: which side each output row came from.
pub fn indicator_column(pairing: Pairing) -> Column {
    let out = Column.allocate(Dtype.Category, pairing.rows)
    out.set_levels(["left_only", "right_only", "both"])
    if pairing.rows > 0 {
        kernel_for("jn_indicator_codes").dispatch_async(
            [pairing.left, pairing.right, out.values],
            pairing.rows,
            [pairing.rows]
        )
    }
    return out
}

#: A dictionary column's codes remapped to the rank of their level.
#:
#: Sorting text means sorting the strings, and the codes are in
#: first-appearance order. The host loop is over the distinct levels, not the
#: rows; a kernel does the rows.
fn level_rank_lane(column: Column) -> Buffer {
    let levels = column.levels ?? []
    var rank: dict[str, int] = {}
    for (at, name) in sorted(levels).enumerate() { rank[name] = at }
    var mapping: list[float] = []
    for name in levels { mapping.push(float(rank.get(name, 0))) }
    let table = Buffer(mapping.len())
    if mapping.len() > 0 { table.upload(mapping) }
    let lane = Buffer(column.length)
    cached_kernel("col_remap").dispatch_async(
        [column.values, lane, table],
        column.length,
        [column.length, levels.len()]
    )
    table.free()
    return lane
}

fn wide_key_lane(column: Column, high_half: bool) -> Buffer {
    let raw = Buffer(column.length)
    kernel_for("jn_sort_key_wide").dispatch_async(
        [column.values, column.wide ?? column.values, column.valid ?? column.values, raw],
        column.length,
        [column.length, flag(column.valid is not null), flag(high_half)]
    )
    return raw
}

#: One stable radix pass of `order` against a raw key lane. Frees `order`.
fn apply_sort_pass(order: Buffer, raw_keys: Buffer, rows: int, bits: int) -> Buffer {
    let lane = Buffer(rows)
    kernel_for("jn_gather_uint").dispatch_async([raw_keys, order, lane], rows, [rows])
    let sorted = radix_order(order, lane, rows, bits)
    lane.free()
    sorted[1].free()
    order.free()
    return sorted[0]
}

#: The passes one key column contributes to a lexicographic ordering.
#:
#: A wide column is two digits, low half first, because 48 bits do not fit one
#: radix key. A dictionary column sorts on the rank of its level rather than on
#: its code. Everything else is one 32-bit order-preserving image.
fn sort_column_passes(order: Buffer, column: Column, rows: int) -> Buffer {
    var next = order
    let has_valid = flag(column.valid is not null)
    if column.dtype.is_wide() {
        let low = wide_key_lane(column, false)
        next = apply_sort_pass(next, low, rows, 24)
        low.free()
        let high = wide_key_lane(column, true)
        next = apply_sort_pass(next, high, rows, 26)
        high.free()
        return next
    }
    let raw = Buffer(rows)
    if column.dtype.is_dictionary() {
        let ranks = level_rank_lane(column)
        kernel_for("jn_sort_key_code").dispatch_async(
            [ranks, column.valid ?? column.values, raw],
            rows,
            [rows, has_valid, column.level_count()]
        )
        ranks.free()
        next = apply_sort_pass(next, raw, rows, bit_width(column.level_count()))
    } elif column.dtype == Dtype.Bool {
        kernel_for("jn_sort_key_code").dispatch_async(
            [column.values, column.valid ?? column.values, raw],
            rows,
            [rows, has_valid, 2]
        )
        next = apply_sort_pass(next, raw, rows, 2)
    } else {
        kernel_for("jn_sort_key_float").dispatch_async(
            [column.values, column.valid ?? column.values, raw],
            rows,
            [rows, has_valid]
        )
        next = apply_sort_pass(next, raw, rows, 32)
    }
    raw.free()
    return next
}

#: A stable ordering of `rows` rows by `keys`, read left to right.
#:
#: Least significant digit last, so the columns are sorted back to front and
#: each pass's stability carries the earlier columns' decision. Missing values
#: take the largest key in every image, which puts them last as pandas does.
pub fn key_order(keys: list[Column], rows: int) -> Buffer {
    var order = iota(rows)
    if rows <= 1 { return order }
    var at = keys.len() - 1
    while at >= 0 {
        if keys[at].length != rows {
            throw ValueError(f"a key of {keys[at].length} rows cannot order {rows}")
        }
        order = sort_column_passes(order, keys[at], rows)
        at -= 1
    }
    return order
}

#: The output rows put into key order, position lanes and keys together.
#:
#: The lanes move in place and the keys are re-taken rather than re-derived, so
#: a caller cannot end up with a frame whose key column disagrees with the rows
#: it labels. The pass is stable, which is what lets an ordered merge append
#: its right-only rows after the matched ones and still have them land between
#: the right neighbours once this runs.
pub fn sort_pairing(pairing: Pairing, keys: list[Column]) -> list[Column] {
    if pairing.rows <= 1 or keys.len() == 0 { return keys }
    let perm = key_order(keys, pairing.rows)
    permute_lane(pairing.left, perm, pairing.rows)
    permute_lane(pairing.right, perm, pairing.rows)
    var out: list[Column] = []
    for key in keys { out.push(key.take(perm, pairing.rows)) }
    for key in keys { key.free() }
    perm.free()
    return out
}

#: Two dictionary key columns put on one table, or the pair unchanged.
pub fn unified_pair(left_key: Column, right_key: Column, owned: list[Column]) -> (Column, Column) {
    if not left_key.dtype.is_dictionary() { return (left_key, right_key) }
    let pair = unify_levels([left_key, right_key])
    owned.push(pair[0])
    owned.push(pair[1])
    return (pair[0], pair[1])
}
