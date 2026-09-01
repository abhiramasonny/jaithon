#: Moving windows: rolling, expanding, and the exponentially weighted mean.
#:
#: A window turns a column into a column of per-window answers. Three paths
#: answer one — a thread per output walking its window, three scans read as
#: differences, or the scan that already is the answer — and `window_column`
#: is where the choice between them is made.
import std.math
from std.gpu import Buffer
from std.math import NAN, is_nan
from ..column import Agg, Column, Dtype
from ..column import f32_bits
from .kernels import AGG_COUNT, AGG_MAX, AGG_MEAN, AGG_MIN, AGG_PROD, AGG_SEM
from .kernels import AGG_STD, AGG_SUM, AGG_VAR
from .kernels import SCAN_ADD, SCAN_GROUP, SCAN_MAX, SCAN_MIN, SCAN_MUL
from .kernels import WIN_GROUP, WIN_TILE, flag, kernel_for
from .aggregate import agg_code
from .scan import dispatch_column_wise, scan_into
from .stats import column_stats

#: Where the host stops dispatching a window kernel and starts scanning.
#:
#: The direct path reads `window` rows per output; the prefix path reads three
#: and pays three extra passes over the column to build the scans. The crossover
#: is where `window` first exceeds what those passes cost, which is a few tens
#: of rows. 64 is on the safe side of it: below 64 the direct path also keeps
#: more digits, because a window sum taken as a difference of two running totals
#: cancels once the totals have grown far past the window's own sum.
const PREFIX_WINDOW = 64

#: The most rows `Window.apply` will run a host callback over.
#:
#: One boxed call per output row: at a million rows that is minutes. Every
#: named aggregation dispatches a kernel instead, so the cap is a signpost to
#: them rather than a limit on what the module can do.
const APPLY_MAX_ROWS = 4096

#: The most rows an expanding window will answer a per-window search over.
#:
#: The aggregations that cannot be scanned — a median, a quantile, a truth test
#: — have one thread read the whole history of its row, which over a whole
#: column is quadratic. Everything a scan can answer is routed to one and has
#: no cap; this is the boundary of what is left.
const EXPAND_DIRECT_MAX = 8192

fn window_code(agg: Agg) -> int {
    if agg == Agg.Sum or agg == Agg.Mean or agg == Agg.Min or agg == Agg.Max {
        return agg_code(agg)
    }
    if agg == Agg.Count or agg == Agg.Prod or agg == Agg.Std or agg == Agg.Var {
        return agg_code(agg)
    }
    if agg == Agg.Median or agg == Agg.Quantile { return agg_code(agg) }
    if agg == Agg.Any or agg == Agg.All or agg == Agg.First or agg == Agg.Last {
        return agg_code(agg)
    }
    throw ValueError(f"{agg.name()} has no window form")
}

fn additive_code(code: int) -> bool {
    let plain = code == AGG_SUM or code == AGG_MEAN or code == AGG_COUNT
    let spread = code == AGG_STD or code == AGG_VAR or code == AGG_SEM
    return plain or spread
}

fn scannable_code(code: int) -> int {
    if code == AGG_MIN { return SCAN_MIN }
    if code == AGG_MAX { return SCAN_MAX }
    if code == AGG_PROD { return SCAN_MUL }
    if code == AGG_SUM { return SCAN_ADD }
    return -1
}

#: The running count of present rows, which several window paths gate on.
fn presence_scan(column: Column, rows: int) -> Buffer {
    let flags = Buffer(rows)
    let squares = Buffer(rows)
    let values = Buffer(rows)
    kernel_for("roll_terms").dispatch_async(
        [
            column.values,
            column.wide ?? column.values,
            column.valid ?? column.values,
            values,
            squares,
            flags,
        ],
        rows,
        [rows, flag(column.dtype.is_wide()), flag(column.valid is not null)]
    )
    let tally = Buffer(rows)
    scan_into(flags, tally, rows, SCAN_ADD, true)
    flags.free()
    squares.free()
    values.free()
    return tally
}

#: Window statistics off three inclusive scans.
fn window_by_prefix(
    column: Column,
    code: int,
    window: int,
    min_periods: int,
    lead: int,
    widening: bool,
    ddof: int
) -> Column {
    let rows = column.len()
    let out = Column.allocate(Dtype.Float64, rows, with_validity: true)
    let values = Buffer(rows)
    let squares = Buffer(rows)
    let flags = Buffer(rows)
    kernel_for("roll_terms").dispatch_async(
        [
            column.values,
            column.wide ?? column.values,
            column.valid ?? column.values,
            values,
            squares,
            flags,
        ],
        rows,
        [rows, flag(column.dtype.is_wide()), flag(column.valid is not null)]
    )
    let sums = Buffer(rows)
    let sum_squares = Buffer(rows)
    let tally = Buffer(rows)
    scan_into(values, sums, rows, SCAN_ADD, true)
    scan_into(squares, sum_squares, rows, SCAN_ADD, true)
    scan_into(flags, tally, rows, SCAN_ADD, true)
    kernel_for("roll_prefix").dispatch_async(
        [sums, sum_squares, tally, out.values, out.valid ?? out.values],
        rows,
        [rows, window, min_periods, lead, code, ddof, flag(widening)]
    )
    values.free()
    squares.free()
    flags.free()
    sums.free()
    sum_squares.free()
    tally.free()
    return out
}

#: Window statistics with one thread per output row, each walking its window.
fn window_direct(
    column: Column,
    code: int,
    window: int,
    min_periods: int,
    lead: int,
    widening: bool,
    ddof: int,
    q: float
) -> Column {
    let rows = column.len()
    let out = Column.allocate(Dtype.Float64, rows, with_validity: true)
    let borrowed = column.borrow_float_lane()
    let lane = borrowed[0]
    let mask = column.valid
    let blocks = (rows + WIN_GROUP - 1) // WIN_GROUP
    let tiled = not widening and window - 1 + WIN_GROUP <= WIN_TILE
    kernel_for("roll_direct").dispatch_async(
        [lane, mask ?? lane, out.values, out.valid ?? out.values],
        blocks * WIN_GROUP,
        [
            rows,
            window,
            min_periods,
            lead,
            code,
            ddof,
            f32_bits(q),
            flag(mask is not null),
            flag(widening),
            flag(tiled),
        ],
        group_size: WIN_GROUP
    )
    if borrowed[1] { lane.free() }
    return out
}

#: An expanding minimum, maximum or product, off the scan that is already the
#: answer, with the rows that have seen too little blanked afterwards.
fn window_by_scan(column: Column, code: int, min_periods: int) -> Column {
    let rows = column.len()
    let out = Column.allocate(Dtype.Float64, rows, with_validity: true)
    let op = scannable_code(code)
    let terms = Buffer(rows)
    kernel_for("scan_prepare").dispatch_async(
        [column.values, column.wide ?? column.values, column.valid ?? column.values, terms],
        rows,
        [rows, flag(column.dtype.is_wide()), flag(column.valid is not null), op, 1]
    )
    scan_into(terms, out.values, rows, op, true)
    terms.free()
    let tally = presence_scan(column, rows)
    (out.valid ?? out.values).fill(1.0)
    kernel_for("win_gate").dispatch_async(
        [out.values, out.valid ?? out.values, tally],
        rows,
        [rows, min_periods]
    )
    tally.free()
    return out
}

#: One column's window statistic, by whichever of the three paths fits.
#:
#: The direct path reads `window` rows per output and stages its group's rows
#: in threadgroup memory, so it is one device read per row however wide the
#: window; the prefix path reads three per output but pays three scans of the
#: column to build them. Narrow windows go direct — fewer passes, fewer
#: allocations, and no cancellation between two running totals that have grown
#: far past the window's own sum. Wide ones and every expanding window go
#: through the prefix path, where the direct path's per-output cost would grow
#: without bound. An expanding extreme or product is neither: the scan that
#: answers it is the answer.
fn window_column(
    column: Column,
    code: int,
    window: int,
    min_periods: int,
    center: bool,
    widening: bool,
    ddof: int,
    q: float
) -> Column {
    if not column.dtype.is_numeric() {
        throw TypeError(f"a window statistic needs numbers, not {column.dtype.name()}")
    }
    let rows = column.len()
    if rows == 0 { return Column.allocate(Dtype.Float64, 0, with_validity: true) }
    let lead = if center { (window - 1) // 2 } else { 0 }
    if additive_code(code) {
        if widening or window >= PREFIX_WINDOW {
            return window_by_prefix(column, code, window, min_periods, lead, widening, ddof)
        }
        return window_direct(column, code, window, min_periods, lead, widening, ddof, q)
    }
    if widening and scannable_code(code) >= 0 {
        return window_by_scan(column, code, min_periods)
    }
    if widening and rows > EXPAND_DIRECT_MAX {
        throw ValueError(
            f"an expanding window of that kind reads its whole history per row, which over {rows} rows is quadratic; the sums, extremes and products are scanned instead"
        )
    }
    return window_direct(column, code, window, min_periods, lead, widening, ddof, q)
}

#: A window over a series or a frame, waiting to be told which statistic.
#:
#: pandas hands one of these back from `rolling` and `expanding` so that the
#: window and the aggregation can be written apart, and so that one window can
#: answer several. It owns no device memory: every method dispatches from the
#: target it was built over, which must outlive it.
pub class Window {
    #: The `Column`, `Series` or `DataFrame` the window runs over.
    pub var target: any
    #: How many rows a window spans. 0 for an expanding one, whose span is the
    #: history.
    pub var window: int
    #: How many present rows a window needs before it answers.
    pub var min_periods: int
    #: Whether the answer is labelled at the middle of the window rather than
    #: at its end.
    pub var center: bool
    #: Whether the window starts at row zero and grows.
    pub var widening: bool

    #: `rolling` and `expanding` are the constructors.
    pub fn init(
        self,
        target: any,
        window: int,
        min_periods: int,
        center: bool = false,
        widening: bool = false
    ) {
        self.target = target
        self.window = window
        self.min_periods = min_periods
        self.center = center
        self.widening = widening
    }

    #: The named aggregation over every window.
    #:
    #: The same `Agg` the whole-column reductions and group-by take, so
    #: `rolling(7).agg(Agg.Mean)` and `mean(series)` are the same arithmetic
    #: over different spans.
    pub fn agg(self, kind: Agg, ddof: int = 1, q: float = 0.5) -> any {
        let code = window_code(kind)
        let span = self.window
        let least = self.min_periods
        let middle = self.center
        let growing = self.widening
        return dispatch_column_wise(
            self.target,
            |column| window_column(column, code, span, least, middle, growing, ddof, q),
            true
        )
    }

    #: The sum over every window.
    pub fn sum(self) -> any { return self.agg(Agg.Sum) }

    #: The mean over every window.
    pub fn mean(self) -> any { return self.agg(Agg.Mean) }

    #: The smallest value in every window.
    pub fn min(self) -> any { return self.agg(Agg.Min) }

    #: The largest value in every window.
    pub fn max(self) -> any { return self.agg(Agg.Max) }

    #: The sample standard deviation over every window.
    pub fn std(self, ddof: int = 1) -> any { return self.agg(Agg.Std, ddof: ddof) }

    #: The sample variance over every window. Spelled out because `var` is a
    #: keyword here.
    pub fn variance(self, ddof: int = 1) -> any { return self.agg(Agg.Var, ddof: ddof) }

    #: The middle value of every window.
    pub fn median(self) -> any { return self.agg(Agg.Median) }

    #: The value `q` of the way through every window.
    pub fn quantile(self, q: float = 0.5) -> any { return self.agg(Agg.Quantile, q: q) }

    #: How many rows of every window hold a value. Never missing.
    pub fn count(self) -> any { return self.agg(Agg.Count) }

    #: The present values of every window, handed to a callback.
    #:
    #: One boxed call per output row, on the host, which is why it is capped.
    #: Everything the named aggregations cover dispatches a kernel instead and
    #: has no cap; this is the escape hatch for the statistic that is not one
    #: of them, and it is worth writing a kernel rather than reaching for it on
    #: a column of any size.
    pub fn apply(self, body: fn(list[float]) -> float) -> any {
        let span = self.window
        let least = self.min_periods
        let middle = self.center
        let growing = self.widening
        return dispatch_column_wise(
            self.target,
            |column| apply_windows(column, body, span, least, middle, growing),
            true
        )
    }
}

fn apply_windows(
    column: Column,
    body: fn(list[float]) -> float,
    window: int,
    min_periods: int,
    center: bool,
    widening: bool
) -> Column {
    let rows = column.len()
    if rows > APPLY_MAX_ROWS {
        throw ValueError(
            f"apply runs a host callback once per row, and {rows} rows is past the {APPLY_MAX_ROWS} that is reasonable; a named aggregation dispatches a kernel instead"
        )
    }
    if rows == 0 { return Column.allocate(Dtype.Float64, 0, with_validity: true) }
    let numbers = column.to_floats()
    let lead = if center { (window - 1) // 2 } else { 0 }
    var out: list[float] = []
    for row in 0..rows {
        let last = if widening { row } else { row + lead }
        let first = if widening { 0 } else { last - window + 1 }
        let lo = max(first, 0)
        let hi = min(last, rows - 1)
        var slice: list[float] = []
        for at in lo..hi + 1 {
            if is_nan(numbers[at]) { continue }
            slice.push(numbers[at])
        }
        if slice.len() == 0 or slice.len() < min_periods {
            out.push(NAN)
            continue
        }
        out.push(body(slice))
    }
    return Column.from_floats(out)
}

#: A window of `window` rows ending at each row.
#:
#: `min_periods` defaults to the whole window, as pandas does, so the leading
#: rows are missing until one is full. `center` labels the answer at the middle
#: of the window instead of at its end.
pub fn rolling(target: any, window: int, min_periods: int? = null, center: bool = false) -> Window {
    if window <= 0 { throw ValueError(f"a rolling window of {window} rows has nothing in it") }
    let least = min_periods ?? window
    if least < 0 { throw ValueError(f"min_periods cannot be {least}") }
    return Window(target, window, least, center: center)
}

#: A window that starts at row zero and grows to each row.
pub fn expanding(target: any, min_periods: int = 1) -> Window {
    if min_periods < 0 { throw ValueError(f"min_periods cannot be {min_periods}") }
    return Window(target, 0, min_periods, widening: true)
}

#: The decay one of pandas' four spellings asks for.
fn ewm_alpha(alpha: float?, span: float?, halflife: float?) -> float {
    var given = 0
    if alpha is not null { given += 1 }
    if span is not null { given += 1 }
    if halflife is not null { given += 1 }
    if given != 1 {
        throw ValueError("ewm needs exactly one of alpha, span and halflife")
    }
    let direct = alpha
    if direct is not null {
        if direct <= 0.0 or direct > 1.0 { throw ValueError(f"alpha is in (0, 1], not {direct}") }
        return direct
    }
    let width = span
    if width is not null {
        if width < 1.0 { throw ValueError(f"a span is at least 1, not {width}") }
        return 2.0 / (width + 1.0)
    }
    let life = halflife ?? 0.0
    if life <= 0.0 { throw ValueError(f"a halflife is positive, not {life}") }
    return 1.0 - math.exp(-math.LN_2 / life)
}

#: The exponentially weighted recurrence, run as a scan.
#:
#: Both lanes decay by the same factor, so one pass carries them together, and
#: the block totals recur with the factor raised to the block width — which
#: underflows to zero for any brisk decay, and that is the right answer: a
#: block's carry-in really has no weight left by the time it reaches the end.
fn ewm_scan(nums: Buffer, dens: Buffer, length: int, decay: float) -> (Buffer, Buffer) {
    let out_num = Buffer(length)
    let out_den = Buffer(length)
    let blocks = (length + SCAN_GROUP - 1) // SCAN_GROUP
    let total_num = Buffer(blocks)
    let total_den = Buffer(blocks)
    kernel_for("ewm_block").dispatch_async(
        [nums, dens, out_num, out_den, total_num, total_den],
        blocks * SCAN_GROUP,
        [length, f32_bits(decay)],
        group_size: SCAN_GROUP
    )
    if blocks > 1 {
        let carried = ewm_scan(total_num, total_den, blocks, decay ** float(SCAN_GROUP))
        kernel_for("ewm_carry").dispatch_async(
            [out_num, out_den, carried[0], carried[1]],
            length,
            [length, f32_bits(decay)]
        )
        carried[0].free()
        carried[1].free()
    }
    total_num.free()
    total_den.free()
    return (out_num, out_den)
}

fn ewm_column(column: Column, alpha: float, adjust: bool) -> Column {
    if not column.dtype.is_numeric() {
        throw TypeError(f"an exponential mean needs numbers, not {column.dtype.name()}")
    }
    let rows = column.len()
    let out = Column.allocate(Dtype.Float64, rows, with_validity: true)
    if rows == 0 { return out }
    var seed = rows
    if not adjust {
        let found = column_stats(column).first_present
        if found >= 0 { seed = found }
    }
    let nums = Buffer(rows)
    let dens = Buffer(rows)
    kernel_for("ewm_prepare").dispatch_async(
        [column.values, column.wide ?? column.values, column.valid ?? column.values, nums, dens],
        rows,
        [
            rows,
            flag(column.dtype.is_wide()),
            flag(column.valid is not null),
            f32_bits(alpha),
            flag(adjust),
            seed,
        ]
    )
    let running = ewm_scan(nums, dens, rows, 1.0 - alpha)
    nums.free()
    dens.free()
    kernel_for("ewm_finish").dispatch_async(
        [running[0], running[1], out.values, out.valid ?? out.values],
        rows,
        [rows]
    )
    running[0].free()
    running[1].free()
    return out
}

#: The exponentially weighted mean, with the decay given any of pandas' ways.
#:
#: A missing row contributes nothing but does not stop the clock, which is
#: pandas' `ignore_na=False`: the weights follow position, not the count of
#: present rows. Rows before the first present one are missing, because there
#: is nothing yet to average.
#:
#: `adjust` chooses between the two conventions. With it, every present row
#: carries weight 1 and the answer is their weighted mean, so the early rows
#: are means of what has arrived; without it, the recurrence is
#: `y = (1 - alpha) y_before + alpha x` seeded at the first present row.
pub fn ewm(
    target: any,
    alpha: float? = null,
    span: float? = null,
    halflife: float? = null,
    adjust: bool = true
) -> any {
    let decay = ewm_alpha(alpha, span, halflife)
    return dispatch_column_wise(target, |column| ewm_column(column, decay, adjust), true)
}
