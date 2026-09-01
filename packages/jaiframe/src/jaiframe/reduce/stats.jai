#: What one pass over a column learns, and the passes that learn it.
#:
#: `ColumnStats` is the currency the rest of `reduce` trades in: a batched
#: summary of every column, the order statistic behind a quantile, and the
#: column-major matrix the per-row and pairwise kernels read.
import std.math
from std.gpu import Buffer
from std.math import NAN
from ..column import Column, f32_bits, wide_unpack
from .kernels import FOLDED, RGROUP, RGROUPS, STATS
from .kernels import flag, kernel_for, require_device

#: Everything one pass over a column learns about it.
#:
#: Every reduction here is arithmetic on one of these, which is why `describe`
#: — six statistics of every column — costs the same single round trip that
#: `mean` of one column does. The power sums are central about `center`; the
#: first pass leaves that at zero, so `total` is the sum and the rest are raw.
#:
#: Positions are row numbers, -1 where the column had nothing to report.
pub class ColumnStats {
    #: Rows in the column, missing ones included.
    pub var length: int
    #: Rows that hold a value.
    pub var present: int
    #: Sum of the present values less `center`.
    pub var total: float
    #: Sum of the squared deviations from `center`.
    pub var sum_squares: float
    #: Sum of the cubed deviations from `center`.
    pub var sum_cubes: float
    #: Sum of the fourth powers of the deviations from `center`.
    pub var sum_quads: float
    #: Product of the present values.
    pub var product: float
    #: How many present values are non-zero.
    pub var truthy: int
    #: Smallest present value, as a float.
    pub var lowest: float
    #: Largest present value, as a float.
    pub var highest: float
    #: The row `lowest` was found at, the earliest one on a tie.
    pub var lowest_at: int
    #: The row `highest` was found at, the earliest one on a tie.
    pub var highest_at: int
    #: First row holding a value.
    pub var first_present: int
    #: Last row holding a value.
    pub var last_present: int
    #: First row holding nothing, which is what a `skipna=false` reduction asks.
    pub var first_missing: int
    #: What the power sums were taken about.
    pub var center: float

    #: An empty summary. `column_stats` is the constructor.
    pub fn init(self) {
        self.length = 0
        self.present = 0
        self.total = 0.0
        self.sum_squares = 0.0
        self.sum_cubes = 0.0
        self.sum_quads = 0.0
        self.product = 1.0
        self.truthy = 0
        self.lowest = 0.0
        self.highest = 0.0
        self.lowest_at = -1
        self.highest_at = -1
        self.first_present = -1
        self.last_present = -1
        self.first_missing = -1
        self.center = 0.0
    }

    #: How many rows hold nothing.
    pub fn missing(self) -> int { return self.length - self.present }

    #: The arithmetic mean of the present values, NaN over none.
    pub fn mean(self) -> float {
        if self.present == 0 { return NAN }
        return self.center + self.total / float(self.present)
    }

    #: The variance about the mean, with `ddof` degrees of freedom spent.
    #:
    #: Off the centred second moment, corrected for the gap between `center`
    #: and the mean it turned out to be. Clamped at zero because that
    #: correction can land a hair below it on a column of identical values.
    pub fn variance(self, ddof: int) -> float {
        let degrees = self.present - ddof
        if self.present == 0 or degrees <= 0 { return NAN }
        let offset = self.total / float(self.present)
        let centred = self.sum_squares - offset * self.total
        return max(centred, 0.0) / float(degrees)
    }
}

#: Whether a lane can be read four floats at a time.
#:
#: Needs no validity lane — the quad path has no place to put a mask test —
#: and a row count that divides by four. A frame straight out of a reader with
#: no missing data is the case this is for.
fn packed_run(length: int, valid: Buffer?) -> bool {
    return valid is null and length > 0 and length % 4 == 0
}

#: Each column summarised, in one round trip for all of them.
#:
#: Two stage-one dispatches and one join per kind, then one download of sixteen
#: floats a column. Batching matters because the wait is what costs: a frame
#: reduced a column at a time pays the round trip once per column and spends
#: more on the bus than on the columns.
pub fn batched_stats(
    lanes: list[Buffer],
    valids: list[Buffer?],
    length: int,
    centers: list[float]
) -> list[ColumnStats] {
    let width = lanes.len()
    var out: list[ColumnStats] = []
    for at in 0..width {
        let empty = ColumnStats()
        empty.length = length
        empty.center = centers[at]
        out.push(empty)
    }
    if width == 0 or length == 0 { return out }
    require_device("reduce a column")
    let moments = Buffer(width * RGROUPS * STATS)
    let extremes = Buffer(width * RGROUPS * STATS)
    let folded = Buffer(width * FOLDED)
    for (at, lane) in lanes.enumerate() {
        let mask = valids[at]
        let quad = packed_run(length, mask)
        kernel_for("red_moments").dispatch_async(
            [lane, mask ?? lane, moments],
            RGROUPS * RGROUP,
            [length, flag(mask is not null), f32_bits(centers[at]), at, flag(quad)],
            group_size: RGROUP
        )
        kernel_for("red_extrema").dispatch_async(
            [lane, mask ?? lane, extremes],
            RGROUPS * RGROUP,
            [length, flag(mask is not null), at, flag(quad)],
            group_size: RGROUP
        )
    }
    kernel_for("red_join_moments").dispatch_async([moments, folded], width * STATS, [width])
    kernel_for("red_join_extrema").dispatch_async([extremes, folded], width, [width])
    let numbers = folded.download()
    moments.free()
    extremes.free()
    folded.free()
    for at in 0..width {
        let base = at * FOLDED
        var stats = out[at]
        stats.present = int(numbers[base + 0])
        stats.total = numbers[base + 1]
        stats.sum_squares = numbers[base + 2]
        stats.sum_cubes = numbers[base + 3]
        stats.sum_quads = numbers[base + 4]
        stats.product = numbers[base + 5]
        stats.truthy = int(numbers[base + 6])
        stats.lowest = numbers[base + 8]
        stats.highest = numbers[base + 9]
        stats.lowest_at = int(numbers[base + 10])
        stats.highest_at = int(numbers[base + 11])
        stats.first_present = int(numbers[base + 12])
        stats.last_present = int(numbers[base + 13])
        stats.first_missing = int(numbers[base + 14])
    }
    return out
}

#: The float lanes of every column, and which of them the caller must free.
#:
#: A wide column has to be flattened to be reduced, and that costs an
#: allocation; a plain one already is its own lane and must not be freed here.
pub fn borrow_lanes(columns: list[Column]) -> (list[Buffer], list[Buffer?], list[bool]) {
    var lanes: list[Buffer] = []
    var masks: list[Buffer?] = []
    var owned: list[bool] = []
    for column in columns {
        let borrowed = column.borrow_float_lane()
        lanes.push(borrowed[0])
        masks.push(column.valid)
        owned.push(borrowed[1])
    }
    return (lanes, masks, owned)
}

pub fn release_lanes(lanes: list[Buffer], owned: list[bool]) -> void {
    for (at, lane) in lanes.enumerate() {
        if owned[at] { lane.free() }
    }
}

#: One column summarised. `center` is what the power sums are taken about.
pub fn column_stats(column: Column, center: float = 0.0) -> ColumnStats {
    let borrowed = borrow_lanes([column])
    let summary = batched_stats(borrowed[0], borrowed[1], column.len(), [center])
    release_lanes(borrowed[0], borrowed[2])
    return summary[0]
}

#: Groups per target in a radix pass. 16384 threads is enough to saturate the
#: read while keeping the merge into 256 device bins one write per group.
const SEL_GROUPS = 64

#: The k-th smallest present value of one column of a packed matrix, for each
#: (column, rank) the caller names.
#:
#: Four radix passes narrow the answer from 32 bits of ordered key to none, and
#: the digit search between the passes runs on the device too, so the whole
#: thing is eight dispatches and one download whatever the row count. Nothing
#: sorts: a quantile wants one value out of a column, and ordering the other
#: million to find it is work the answer does not need. Targets are batched
#: because `describe` asks for three quantiles of every column and the pass
#: over the rows is the same pass for all of them.
pub fn select_order_stats(
    matrix: Buffer,
    mask: Buffer?,
    rows: int,
    columns: list[int],
    ranks: list[int]
) -> list[float] {
    let lanes = columns.len()
    if lanes == 0 or rows == 0 { return [] }
    require_device("find an order statistic")
    let targets = Buffer(lanes)
    targets.upload(columns.map(|at| float(at)))
    let wanted = Buffer(lanes)
    wanted.upload(ranks.map(|k| float(k)))
    let prefixes = Buffer(lanes)
    prefixes.fill_zero()
    let bins = Buffer(lanes * 256)
    let answers = Buffer(lanes)
    for pass in 0..4 {
        bins.fill_zero()
        kernel_for("sel_histogram").dispatch_async(
            [matrix, mask ?? matrix, targets, prefixes, bins],
            lanes * SEL_GROUPS * RGROUP,
            [rows, SEL_GROUPS, 24 - pass * 8, flag(mask is not null)],
            group_size: RGROUP
        )
        kernel_for("sel_pick").dispatch_async(
            [bins, wanted, prefixes, answers],
            lanes,
            [lanes, flag(pass == 3)]
        )
    }
    let out = answers.download()
    targets.free()
    wanted.free()
    prefixes.free()
    bins.free()
    answers.free()
    return out
}

#: The two order statistics a quantile sits between, interpolated linearly.
#:
#: pandas' default interpolation. Both ranks go into one batched search, so a
#: quantile costs what one order statistic costs.
fn quantile_from(
    matrix: Buffer,
    mask: Buffer?,
    rows: int,
    at: int,
    present: int,
    q: float
) -> float {
    if present == 0 { return NAN }
    if q < 0.0 or q > 1.0 { throw ValueError(f"a quantile is between 0 and 1, not {q}") }
    let position = q * float(present - 1)
    let lower = int(math.floor(position))
    let fraction = position - float(lower)
    let upper = if lower + 1 < present { lower + 1 } else { lower }
    let picks = select_order_stats(matrix, mask, rows, [at, at], [lower, upper])
    return picks[0] + fraction * (picks[1] - picks[0])
}

pub fn column_quantile(column: Column, present: int, q: float) -> float {
    let borrowed = column.borrow_float_lane()
    let answer = quantile_from(borrowed[0], column.valid, column.len(), 0, present, q)
    if borrowed[1] { borrowed[0].free() }
    return answer
}

#: The row a two-lane column's extreme sits at, folded in `long`.
#:
#: One dispatch and one download of 768 floats, over the same fixed grid the
#: float reduction uses. The 256 group winners are folded on the host because
#: 256 of them is microseconds and a second kernel to save it would be a second
#: round trip.
fn wide_extreme_row(column: Column, want_high: bool) -> int {
    require_device("find the extreme of a two-lane column")
    let partials = Buffer(RGROUPS * 3)
    let high = column.wide ?? column.values
    let mask = column.valid ?? column.values
    kernel_for("red_wide_extreme").dispatch_async(
        [column.values, high, mask, partials],
        RGROUPS * RGROUP,
        [column.len(), flag(column.valid is not null), flag(want_high)],
        group_size: RGROUP
    )
    let numbers = partials.download()
    partials.free()
    var best = 0
    var row = -1
    for group in 0..RGROUPS {
        let at = int(numbers[group * 3 + 2])
        if at < 0 { continue }
        let value = wide_unpack(numbers[group * 3 + 0], numbers[group * 3 + 1])
        var wins = row < 0
        if not wins { wins = if want_high { value > best } else { value < best } }
        if not wins { wins = value == best and at < row }
        if wins {
            best = value
            row = at
        }
    }
    return row
}

#: The row an extreme sits at, exact whatever the column stores.
#:
#: An `Int64` or a `Datetime` column is two float32 lanes, and the summary was
#: folded from the one lane they flatten to — which past 2^24 is not the
#: number, so the rows that share a float32 all tie and the summary reports the
#: first of them. Those columns pay one more pass to be told which row it
#: really was; every other column already knows.
pub fn extreme_row(column: Column, stats: ColumnStats, want_high: bool) -> int {
    let folded = if want_high { stats.highest_at } else { stats.lowest_at }
    if folded < 0 or not column.dtype.is_wide() { return folded }
    return wide_extreme_row(column, want_high)
}

#: The columns in one column-major matrix, with a presence matrix beside it.
#:
#: Column-major so that the width-th read of neighbouring rows is contiguous,
#: which is what the per-row and pairwise kernels do. `centers` is subtracted
#: as the copy happens; a covariance pass passes the column means there and a
#: per-row pass passes zeros.
pub fn pack_frame(columns: list[Column], rows: int, centers: list[float]) -> (Buffer, Buffer) {
    let width = columns.len()
    require_device("pack a frame")
    let matrix = Buffer(max(width * rows, 1))
    let mask = Buffer(max(width * rows, 1))
    for (at, column) in columns.enumerate() {
        if rows == 0 { continue }
        kernel_for("red_pack").dispatch_async(
            [
                column.values,
                column.wide ?? column.values,
                column.valid ?? column.values,
                matrix,
                mask,
            ],
            rows,
            [
                rows,
                at * rows,
                flag(column.dtype.is_wide()),
                flag(column.valid is not null),
                f32_bits(centers[at]),
            ]
        )
    }
    return (matrix, mask)
}

pub fn zeros(length: int) -> list[float] {
    var out: list[float] = []
    for _at in 0..length { out.push(0.0) }
    return out
}
