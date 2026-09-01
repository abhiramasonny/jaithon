#: The compiled-kernel cache, and the backward dispatches every op family
#: shares.
from std.gpu import Kernel
from ..tensor import Tensor, element_count
from .shader_attn import ATTN_SOURCE
from .shaders import SOURCE

var _kernel_cache: dict[str, Kernel] = {}

pub fn f32_bits(value: float) -> int {
    if value != value { return 0x7FC00000 }
    var mag = value
    var sign = 0
    if mag < 0.0 {
        sign = 1
        mag = -mag
    }
    if mag == 0.0 { return sign << 31 }
    if mag >= 3.4028234663852886e38 { return sign << 31 | 0x7F800000 }
    if mag <= 1.401298464324817e-45 { return sign << 31 }
    let split = __prim__.f64_frexp(mag)
    let biased = split[1] + 126
    if biased <= 0 {
        let frac = int(__prim__.f64_ldexp(mag, 149))
        return sign << 31 | frac & 0x7FFFFF
    }
    if biased >= 255 { return sign << 31 | 0x7F800000 }
    let frac = int((split[0] * 2.0 - 1.0) * 8388608.0 + 0.5)
    return sign << 31 | biased << 23 | frac & 0x7FFFFF
}

#: The compiled pipeline for one entry point in this module's shader library.
#:
#: Keyed by the entry name alone: there is exactly one `SOURCE`, and building a
#: key that included it meant concatenating and hashing 36 KB on every single
#: dispatch — 88 microseconds of host time per elementwise op, against 6 for
#: the dispatch itself.
#: The threadgroup width every kernel in this module that reduces an axis
#: assumes it will get.
const KERNEL_GROUP = 256

#: Refuse a kernel the device will not run at the width its source assumes.
#:
#: The reducing kernels here stride their axis with `i += 256` and are asked
#: for with a threadgroup of 256, but the dispatch clamps that silently to
#: whatever the pipeline allows. A kernel whose registers or threadgroup
#: memory push its limit below 256 would then run, say, 128 threads while the
#: stride stayed 256 -- and elements 128 to 255 of every row would simply
#: never be read. A softmax missing half its denominator, a mean over half
#: its row, no error anywhere. It does not happen on this machine, where all
#: of them allow 1024, but it is a property of the driver and the device
#: rather than of this source, so it is checked rather than assumed.
fn checked(entry: str, kernel: Kernel) -> Kernel {
    let allowed = kernel.max_group_size()
    if allowed < KERNEL_GROUP {
        throw RuntimeError(
            f"kernel '{entry}' assumes a threadgroup of {KERNEL_GROUP} but this device allows {allowed}"
        )
    }
    return kernel
}

pub fn cached_kernel(entry: str) -> Kernel {
    let existing = _kernel_cache.get(entry, null)
    if existing is not null { return existing }
    let kernel = checked(entry, Kernel.compile(SOURCE, entry))
    _kernel_cache[entry] = kernel
    return kernel
}

var _attn_cache: dict[str, Kernel] = {}

#: The same, for the attention shader library.
pub fn cached_attn_kernel(entry: str) -> Kernel {
    let existing = _attn_cache.get(entry, null)
    if existing is not null { return existing }
    let kernel = checked(entry, Kernel.compile(ATTN_SOURCE, entry))
    _attn_cache[entry] = kernel
    return kernel
}

pub fn reduce_leading(upstream: Tensor, small_shape: list[int], out: Tensor? = null) -> Tensor {
    let inner = element_count(small_shape)
    let outer = upstream.size // inner
    let output = if out is not null {
        if out.size != inner {
            throw ValueError(f"reduce_leading out size {out.size} does not match {inner}")
        }
        out
    } else {
        Tensor(small_shape)
    }
    #: One thread a column reads every row of that column, which is fine while
    #: the columns outnumber the rows -- a dense layer's bias gradient is a
    #: hundred rows against a thousand columns. A convolution's is the other
    #: way round: thirty-two columns against a hundred and thirty thousand
    #: rows, so thirty-two threads did four million serial reads while the rest
    #: of the GPU sat idle.
    #:
    #: Splitting the rows fixes that, but only if it is split far enough. One
    #: pass with a couple of hundred chunks still left five hundred threads
    #: running on a machine that wants tens of thousands, and the reduction
    #: measured a millisecond and a third whatever the layer's size -- the
    #: signature of work that is waiting rather than reading. So it splits
    #: repeatedly, each pass turning `outer` rows into `chunks`, until what is
    #: left is small enough for one thread a column to finish quickly.
    #:
    #: The column index is what varies fastest across threads, which is what
    #: keeps neighbouring threads reading neighbouring addresses.
    let FINAL_ROWS = 64
    let TARGET_THREADS = 65536
    let MIN_ROWS_PER_CHUNK = 8
    var rows = outer
    var source = upstream
    var owned: Tensor? = null
    while rows > FINAL_ROWS {
        let chunks = min(max(1, TARGET_THREADS // inner), max(1, rows // MIN_ROWS_PER_CHUNK))
        if chunks >= rows { break }
        let partials = Tensor([chunks * inner])
        cached_kernel("reduce_leading_split").dispatch_async(
            [source.buffer(), partials.buffer()],
            inner * chunks,
            [rows, inner, chunks]
        )
        let previous = owned
        if previous is not null { previous.free() }
        owned = partials
        source = partials
        rows = chunks
    }

    if inner % 4 == 0 {
        cached_kernel("reduce_leading4").dispatch_async(
            [source.buffer(), output.buffer()],
            inner // 4,
            [rows, inner]
        )
    } else {
        cached_kernel("reduce_leading").dispatch_async(
            [source.buffer(), output.buffer()],
            inner,
            [rows, inner]
        )
    }
    let last = owned
    if last is not null { last.free() }
    return output
}

pub fn unary_backward(saved: Tensor, upstream: Tensor, operation: int) -> Tensor {
    let grad_input = Tensor(saved.shape)
    if operation == 1 and saved.size % 4 == 0 {
        let kernel = cached_kernel("unary_backward_relu4")
        kernel.dispatch_async(
            [saved.buffer(), upstream.buffer(), grad_input.buffer()],
            saved.size // 4,
            [saved.size]
        )
        return grad_input
    }
    let kernel = cached_kernel("unary_backward_op")
    kernel.dispatch_async(
        [saved.buffer(), upstream.buffer(), grad_input.buffer()],
        saved.size,
        [saved.size, operation]
    )
    return grad_input
}
