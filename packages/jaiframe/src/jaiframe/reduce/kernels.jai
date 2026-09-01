#: The grid constants, the aggregation codes, and the compiled-kernel cache.
#:
#: Everything the other submodules need before they can dispatch: the sizes
#: that mirror the ones in `shader`, the integer codes the kernels switch on,
#: and `kernel_for`, which is where the Metal source is compiled.
import std.gpu as gpu
from std.gpu import Kernel
from .shader import SOURCE

#: Threads in a reduction group, and groups in the fixed reduction grid.
#:
#: 65536 threads walk a column in strides. 256 groups rather than jaicv's 1024
#: because the join here is one thread folding them serially — 256 dependent
#: adds is microseconds, and it keeps the join a dozen lines instead of a
#: second simd tree.
pub const RGROUP = 256
pub const RGROUPS = 256

#: Floats a stage-one group writes, and floats a folded column occupies.
pub const STATS = 8
pub const FOLDED = 16

#: Threads in a scan block. The block totals are scanned by the same kernel,
#: so 256 covers 2^24 rows in three levels.
pub const SCAN_GROUP = 256

#: Threads in a window group, and the widest tile one can stage.
#:
#: A group of 256 whose threads each read a window of `w` covers `255 + w`
#: rows between them, so a 1024-float tile carries every window up to 769.
pub const WIN_GROUP = 256
pub const WIN_TILE = 1024

#: Groups per pair in the pairwise pass, and threads in each.
pub const PGROUP = 256
pub const PCHUNKS = 32

pub const AGG_SUM = 0
pub const AGG_MEAN = 1
pub const AGG_MIN = 2
pub const AGG_MAX = 3
pub const AGG_COUNT = 4
pub const AGG_PROD = 5
pub const AGG_STD = 6
pub const AGG_VAR = 7
pub const AGG_MEDIAN = 8
pub const AGG_QUANTILE = 9
pub const AGG_ANY = 10
pub const AGG_ALL = 11
pub const AGG_SEM = 12
pub const AGG_FIRST = 13
pub const AGG_LAST = 14
pub const AGG_SIZE = 15
pub const AGG_ARGMIN = 16
pub const AGG_ARGMAX = 17
pub const AGG_SKEW = 18
pub const AGG_KURT = 19

pub const SCAN_ADD = 0
pub const SCAN_MUL = 1
pub const SCAN_MAX = 2
pub const SCAN_MIN = 3

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
    if not gpu.is_available() { throw RuntimeError(f"jaiframe needs a GPU to {what}") }
}
