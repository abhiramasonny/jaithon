#: The packed key encoding: one uint32 digit lane per key column, ordered.
from std.gpu import Buffer

from ..column import Column, Dtype
from ..column import iota
from .lanes import RADIX_BITS, RADIX_GROUP, RADIX_SIZE
from .lanes import bits_for, flag, kernel_for, require_device, u32

#: The key columns of a frame turned into digits: one packed lane of uint32 per
#: level of the key, most significant first.
#:
#: The encoding is the point of this module. Unsigned integer order over a lane
#: is the value's order, and equality on every lane is row equality — so a
#: radix sort and a hash table both stop caring what dtype went in. What it
#: costs is a second copy of the key columns as uints: `lane_count * rows`
#: words, which for a three-column key over a million rows is about 25 MB.
#:
#: Lanes are held in one allocation rather than one buffer apiece so that the
#: table's equality test is a strided walk of a single binding, and so that the
#: whole key is one `free`.
pub class KeyEncoding {
    #: How many rows are keyed.
    pub var length: int
    #: `lane_count() * length` uint32 digits, lane `l` starting at `l * length`.
    #: Device-side bits: downloading this as floats gives nonsense.
    pub var packed: Buffer
    #: Significant bits in each lane, which is how many radix passes it needs.
    pub var widths: list[int]
    #: Which key column each lane came from, so `ascending` can be per column.
    pub var owners: list[int]
    #: Which lanes are a column's missing flag rather than its value.
    pub var null_lanes: list[bool]
    #: How many key columns went in.
    pub var columns: int
    #: 1.0 where any key column is missing, `null` when none of them can be.
    pub var missing: Buffer?
    var freed: bool

    #: An empty encoding. `encode_keys` is the constructor.
    pub fn init(self) {
        self.length = 0
        self.packed = Buffer(0)
        self.widths = []
        self.owners = []
        self.null_lanes = []
        self.columns = 0
        self.missing = null
        self.freed = false
    }

    #: How many digit lanes the key spans.
    pub fn lane_count(self) -> int { return self.widths.len() }

    #: One uint32 hash per row, folded over every lane.
    #:
    #: This agrees with the equality the hash table here tests, because both
    #: read the same lanes. `Column.hash_into` is the other self-consistent
    #: pair, with `Column.equal_rows`; do not mix a hash from one with an
    #: equality from the other.
    #:
    #: Device-side uint bits, like `packed`.
    pub fn hash(self, seed: int = 0) -> Buffer {
        self.check_live()
        let out = Buffer(self.length)
        if self.length == 0 { return out }
        kernel_for("key_hash_lanes").dispatch_async(
            [self.packed, out],
            self.length,
            [self.length, self.lane_count(), u32(seed)]
        )
        return out
    }

    #: The permutation that sorts the rows, stable, least significant key first.
    #:
    #: A radix sort is stable, and stability is exactly what turns a multi-key
    #: sort into a sequence of single-key ones: sort by the last key, then by
    #: the one before it, and the second pass leaves rows that tie on it in the
    #: order the first pass left them. So there is no comparator that has to
    #: know about all the keys at once — the lanes are walked from least to
    #: most significant and each pass only ever looks at eight bits.
    #:
    #: `ascending` is one flag per key column, or a single flag for all of
    #: them, or `null` for all ascending. `na_position` is absolute the way
    #: pandas' is: `"last"` puts missing rows at the end whether the sort is
    #: ascending or descending, which is why a null lane is never flipped along
    #: with its column.
    pub fn order(self, ascending: list[bool]? = null, na_position: str = "last") -> Buffer {
        self.check_live()
        if na_position != "first" and na_position != "last" {
            throw ValueError(f"na_position must be first or last, not {na_position}")
        }
        let ups = spread_flags(ascending, self.columns)
        var forward = iota(self.length)
        if self.length == 0 { return forward }

        let blocks = (self.length + RADIX_GROUP - 1) // RADIX_GROUP
        var keys = Buffer(self.length)
        var spare_keys = Buffer(self.length)
        var spare = Buffer(self.length)
        let counts = Buffer(blocks * RADIX_SIZE)

        var lane = self.lane_count() - 1
        while lane >= 0 {
            let width = self.widths[lane]
            let mask = (1 << width) - 1
            let invert = if self.null_lanes[lane] {
                na_position == "first"
            } else {
                not ups[self.owners[lane]]
            }
            kernel_for("key_lane_gather").dispatch_async(
                [self.packed, forward, keys],
                self.length,
                [self.length, lane * self.length, mask, flag(invert)]
            )
            var shift = 0
            while shift < width {
                kernel_for("key_radix_histogram").dispatch_async(
                    [keys, counts],
                    blocks * RADIX_GROUP,
                    [self.length, shift],
                    group_size: RADIX_GROUP
                )
                kernel_for("key_radix_offsets").dispatch_async(
                    [counts],
                    RADIX_GROUP,
                    [blocks],
                    group_size: RADIX_GROUP
                )
                kernel_for("key_radix_scatter").dispatch_async(
                    [keys, forward, spare_keys, spare, counts],
                    blocks * RADIX_GROUP,
                    [self.length, shift],
                    group_size: RADIX_GROUP
                )
                let swapped_keys = keys
                keys = spare_keys
                spare_keys = swapped_keys
                let swapped = forward
                forward = spare
                spare = swapped
                shift += RADIX_BITS
            }
            lane -= 1
        }
        keys.free()
        spare_keys.free()
        spare.free()
        counts.free()
        return forward
    }

    #: 1.0 where a row starts a run of equal keys along `positions`.
    #:
    #: The caller owns the lane. Ties are read off the same encoding the sort
    #: used, so a rank and a `keep: all` cutoff cannot disagree with the order
    #: they were computed from.
    pub fn run_flags(self, positions: Buffer) -> Buffer {
        self.check_live()
        let flags = Buffer(self.length)
        if self.length == 0 { return flags }
        kernel_for("key_run_starts").dispatch_async(
            [self.packed, positions, flags],
            self.length,
            [self.length, self.lane_count()]
        )
        return flags
    }

    #: The same key restricted to the rows `positions` names.
    #:
    #: The digits are moved rather than recomputed, so a sort of a group-by's
    #: uniques agrees with the grouping by construction: re-encoding the taken
    #: columns would rank a dictionary against whatever level set survived the
    #: take.
    pub fn gather(self, positions: Buffer, count: int) -> KeyEncoding {
        self.check_live()
        let out = KeyEncoding()
        out.length = count
        out.columns = self.columns
        out.widths = [width for width in self.widths]
        out.owners = [owner for owner in self.owners]
        out.null_lanes = [is_null for is_null in self.null_lanes]
        out.packed = Buffer(self.lane_count() * count)
        let absent = self.missing
        if absent is not null {
            let taken = Buffer(count)
            if count > 0 {
                kernel_for("key_gather_float").dispatch_async(
                    [absent, positions, taken],
                    count,
                    [count, self.length]
                )
            }
            out.missing = taken
        }
        if count > 0 {
            kernel_for("key_subset_lanes").dispatch_async(
                [self.packed, positions, out.packed],
                count,
                [count, self.lane_count(), self.length]
            )
        }
        return out
    }

    #: Release the packed lanes. Freeing twice is harmless.
    pub fn free(self) -> void {
        if self.freed { return }
        self.freed = true
        self.packed.free()
        let absent = self.missing
        if absent is not null { absent.free() }
    }

    fn check_live(self) -> void {
        if self.freed { throw RuntimeError("this KeyEncoding has already been released") }
    }
}

#: The key columns encoded into one hashable, order-preserving lane set.
#:
#: Text is ranked by its level string, not by its dictionary code, because
#: codes are in first-appearance order and nobody sorts by that. The rank table
#: is a host sort over the distinct values — not over the rows, which one
#: dispatch remaps — so a column of a million rows over twenty cities costs
#: twenty comparisons on the host.
#:
#: Two dictionary columns only encode alike once `unify_levels` has put them on
#: one table, exactly as they only hash alike after it: a level set of
#: `["a", "c"]` ranks `"c"` at 1 and one of `["a", "b", "c"]` ranks it at 2.
#: A join across two frames unifies first.
pub fn encode_keys(columns: list[Column]) -> KeyEncoding {
    if columns.len() == 0 { throw ValueError("a key needs at least one column") }
    require_device("encode a key")
    let length = columns[0].len()
    for column in columns {
        if column.len() != length {
            throw ValueError(f"key columns disagree on length: {column.len()} against {length}")
        }
    }

    let encoding = KeyEncoding()
    encoding.length = length
    encoding.columns = columns.len()
    var widths: list[int] = []
    var owners: list[int] = []
    var null_lanes: list[bool] = []
    var any_nullable = false
    for (at, column) in columns.enumerate() {
        if nullable(column) {
            widths.push(1)
            owners.push(at)
            null_lanes.push(true)
            any_nullable = true
        }
        for width in value_widths(column) {
            widths.push(width)
            owners.push(at)
            null_lanes.push(false)
        }
    }
    encoding.widths = widths
    encoding.owners = owners
    encoding.null_lanes = null_lanes
    encoding.packed = Buffer(widths.len() * length)

    var absent: Buffer? = null
    if any_nullable {
        absent = Buffer(length)
        if length > 0 { (absent ?? encoding.packed).fill_zero() }
    }
    encoding.missing = absent

    if length == 0 { return encoding }
    var lane = 0
    for column in columns {
        let mask = column.valid ?? column.values
        let has_valid = flag(column.has_validity())
        if nullable(column) {
            kernel_for("key_encode_missing").dispatch_async(
                [column.values, mask, encoding.packed],
                length,
                [length, lane * length, has_valid, flag(column.dtype == Dtype.Float64)]
            )
            if absent is not null {
                kernel_for("key_any_missing").dispatch_async(
                    [encoding.packed, absent ?? encoding.packed],
                    length,
                    [length, lane * length]
                )
            }
            lane += 1
        }
        if column.dtype.is_wide() {
            kernel_for("key_encode_wide").dispatch_async(
                [column.values, column.wide ?? column.values, mask, encoding.packed],
                length,
                [length, lane * length, (lane + 1) * length, has_valid]
            )
            lane += 2
            continue
        }
        if column.dtype.is_dictionary() {
            let levels = column.levels ?? []
            let ranks = Buffer(if levels.len() > 0 { levels.len() } else { 1 })
            if levels.len() > 0 { ranks.upload(level_ranks(levels)) }
            kernel_for("key_encode_code").dispatch_async(
                [column.values, mask, ranks, encoding.packed],
                length,
                [length, lane * length, has_valid, levels.len()]
            )
            ranks.free()
            lane += 1
            continue
        }
        kernel_for("key_encode_plain").dispatch_async(
            [column.values, mask, encoding.packed],
            length,
            [length, lane * length, has_valid, flag(column.dtype == Dtype.Bool)]
        )
        lane += 1
    }
    return encoding
}

#: One uint32 hash per row over the key columns.
#:
#: Device-side bits, as `KeyEncoding.hash` returns: this is that call with the
#: encoding built and thrown away, for a caller that wants the hash and nothing
#: else. A caller that also wants equality or an order should keep the
#: encoding.
pub fn hash_keys(columns: list[Column], seed: int = 0) -> Buffer {
    let encoding = encode_keys(columns)
    defer { encoding.free() }
    return encoding.hash(seed)
}

#: Whether a column can hold a missing value at all, and so needs a null lane.
#:
#: A `Float64` column always gets one even with no validity lane, because a
#: NaN written by a kernel — a zero divide, a mean over an empty group — is
#: missing to everything downstream and has to sort and group as one. The lane
#: is one bit, so the cost of being wrong the safe way is a single radix pass.
fn nullable(column: Column) -> bool {
    return column.has_validity() or column.dtype == Dtype.Float64
}

fn value_widths(column: Column) -> list[int] {
    if column.dtype.is_wide() { return [25, 24] }
    if column.dtype.is_dictionary() {
        return [bits_for(if column.level_count() > 1 { column.level_count() } else { 2 })]
    }
    if column.dtype == Dtype.Bool { return [1] }
    return [32]
}

fn level_ranks(levels: list[str]) -> list[float] {
    var codes: list[int] = []
    for code in 0..levels.len() { codes.push(code) }
    let ordered = codes.sort(key: |code| levels[code])
    var ranks: list[float] = []
    for _code in 0..levels.len() { ranks.push(0.0) }
    for (rank, code) in ordered.enumerate() { ranks[code] = float(rank) }
    return ranks
}

fn spread_flags(flags: list[bool]?, wanted: int) -> list[bool] {
    if flags is null { return [true for _at in 0..wanted] }
    let given = flags ?? []
    if given.len() == 1 { return [given[0] for _at in 0..wanted] }
    if given.len() != wanted {
        throw ValueError(f"{given.len()} ascending flags cannot cover {wanted} keys")
    }
    return [value for value in given]
}
