#: The grid constants, the operation codes, and the compiled-kernel cache.
#:
#: Everything the rest of the module needs before it can dispatch: the sizes
#: and codes that mirror the ones in `shader`, `kernel_for`, which is where the
#: Metal source is compiled, and the lane plumbing — a result lane wrapped in a
#: `Column` without a copy, and a copy when one really is wanted.
import std.gpu as gpu
from std.gpu import Buffer, Kernel
from ..column import Column, Dtype
from ..column import cached_kernel as column_kernel
from .shader import SOURCE

#: Threadgroup width of the segmented kernels. The shader stages `ORDER_TILE`
#: order entries per pass through threadgroup memory; that width is its own.
pub const SEGMENT_GROUP = 256

#: How wide the stable layout's histogram may get, in float slots.
#:
#: The layout counts each chunk's rows per group into a `chunks x groups`
#: matrix and scans it, so the matrix width is the parallelism and its area is
#: the scan's cost. Eight million slots is 32 MB: a thousand groups get 4096
#: chunks, a hundred thousand get 83, and a million get 8. Past that the layout
#: pass runs on very few threads — the documented limit of `cumcount`, `head`,
#: `tail` and `nth`, which are the calls a user reaches for when the groups are
#: few anyway.
pub const LAYOUT_SLOTS = 8388608
pub const LAYOUT_CHUNKS = 4096

#: Slices a tall group is split into, and the thread budget that picks the
#: count. One thread per `(group, column)` under-threads a frame with a handful
#: of groups, so a segment is cut into slices that reduce independently and a
#: fold of at most `MAX_SLICES` partials finishes it.
pub const THREAD_BUDGET = 65536
pub const MAX_SLICES = 64

#: Digits of the per-group quantile selection, and how many buckets that is.
pub const SELECT_BITS = 4
pub const SELECT_BUCKETS = 16

pub const OP_SUM = 0
pub const OP_COUNT = 1
pub const OP_NONZERO = 2
pub const OP_DEV2 = 3
pub const OP_PROD = 4
pub const OP_MIN = 5
pub const OP_MAX = 6
pub const OP_FIRST = 7
pub const OP_LAST = 8

pub const FINISH_DIV = 0
pub const FINISH_VAR = 1
pub const FINISH_STD = 2
pub const FINISH_ANY = 3
pub const FINISH_ALL = 4
pub const FINISH_LERP = 5

pub const PICK_HEAD = 0
pub const PICK_TAIL = 1
pub const PICK_NTH = 2

var _kernels: dict[str, Kernel] = {}

pub fn kernel_for(entry: str) -> Kernel {
    let existing = _kernels.get(entry, null)
    if existing is not null { return existing }
    let compiled = Kernel.compile(SOURCE, entry)
    _kernels[entry] = compiled
    return compiled
}

pub fn flag(value: bool) -> int { return if value { 1 } else { 0 } }

pub fn u32(value: int) -> int { return value & 0xFFFFFFFF }

pub fn require_device(what: str) -> void {
    if not gpu.is_available() {
        throw RuntimeError(f"jaiframe needs a GPU to {what}")
    }
}

#: A `Column` that adopts `lane` rather than copying it.
#:
#: Every kernel here writes its result into a bare lane, and wrapping it costs
#: nothing: `Column.allocate` hands out a fresh buffer that is thrown away
#: immediately, which is one allocation, not one copy of the data.
pub fn adopt_lane(lane: Buffer, length: int, dtype: Dtype, valid: Buffer?) -> Column {
    let column = Column.allocate(dtype, length)
    column.values.free()
    column.values = lane
    column.valid = valid
    return column
}

#: The same lane read as `Int64`, which is exact because every integer a
#: reduction here produces is a row count and `MAX_ROWS` is 2^24.
pub fn adopt_int_lane(lane: Buffer, length: int, valid: Buffer?) -> Column {
    let column = Column.allocate(Dtype.Int64, length)
    column.valid = valid
    if length > 0 {
        kernel_for("gb_pack_int").dispatch_async(
            [lane, column.values, column.wide ?? column.values],
            length,
            [length]
        )
    }
    lane.free()
    return column
}

pub fn bounded(value: int, low: int, high: int) -> int { return max(low, min(high, value)) }

#: `count` slots of `src` copied into `dst`, through the column store's own
#: lane copy rather than a second kernel that would say the same thing.
pub fn copy_lane(
    src: Buffer,
    dst: Buffer,
    count: int,
    src_offset: int = 0,
    dst_offset: int = 0
) -> void {
    if count <= 0 { return }
    column_kernel("col_copy_lane").dispatch_async(
        [src, dst],
        count,
        [count, src_offset, dst_offset]
    )
}

pub fn free_lane(lane: Buffer?) -> void {
    if lane is not null { lane.free() }
}
