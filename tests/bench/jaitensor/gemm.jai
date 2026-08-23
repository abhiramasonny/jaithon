from jaitensor import Tensor, matmul, set_mixed_precision
from jaitensor import is_available
from jaitensor.ops import matmul_backend
import std.gpu as gpu
import std.math
from std.os import env_or
from std.time import Instant

#: Dispatches to run before the clock starts.
#:
#: The GPU does not run at its working speed until it has been busy for a
#: while, and one warm-up call is nowhere near enough to get it there: the same
#: 512x784x8192 product measured 11.9 ms on the first eight dispatches and
#: 6.6 ms by the fortieth, all of it the clock coming up. Timing the first
#: eight measures the ramp rather than the kernel, and it measured jaitensor
#: 20 per cent behind PyTorch on shapes where it is in fact 20 per cent ahead.
#:
#: The peer benchmark warms up by the same rule, which is the only thing that
#: makes the two comparable.
let WARMUP = 24

fn gemm_batch(level: str) -> int {
    if level == "easy" { return 512 }
    if level == "medium" { return 2048 }
    return 4096
}

#: How long each shape should be timed for, in seconds.
#:
#: A fixed repeat count leaves the small shapes measured over a few
#: milliseconds and the large ones over hundreds, so their run-to-run spreads
#: are nothing alike -- several shapes here swung by 7 to 11% between runs of
#: the same binary, which is enough to move a row from a win to a loss. Timing
#: every shape for the same duration puts them on the same footing. The peer
#: calibrates against the same target, so each side measures itself for a
#: sixth of a second and the comparison is between per-call times.
let TARGET_SECONDS = 0.15
let MIN_REPEATS = 40
let MAX_REPEATS = 4096

#: How many repeats of this shape fill `TARGET_SECONDS`, from a short probe.
fn repeats_for(
    rows: int,
    inner: int,
    columns: int,
    floor: int,
    trans_a: bool = false,
    trans_b: bool = false
) -> int {
    let a = if trans_a { Tensor([inner, rows]) } else { Tensor([rows, inner]) }
    let b = if trans_b { Tensor([columns, inner]) } else { Tensor([inner, columns]) }
    let out = Tensor([rows, columns])
    defer { a.free() }
    defer { b.free() }
    defer { out.free() }
    for _warm in 0..WARMUP { matmul(a, b, trans_a: trans_a, trans_b: trans_b, out: out) }
    gpu.synchronize()
    let started = Instant.now()
    for _probe in 0..8 { matmul(a, b, trans_a: trans_a, trans_b: trans_b, out: out) }
    gpu.synchronize()
    let each = started.elapsed().to_seconds() / 8.0
    if each <= 0.0 { return MAX_REPEATS }
    let wanted = int(TARGET_SECONDS / each)
    if wanted < floor { return floor }
    if wanted > MAX_REPEATS { return MAX_REPEATS }
    return wanted
}

fn time_shape(
    rows: int,
    inner: int,
    columns: int,
    repeats: int,
    trans_a: bool = false,
    trans_b: bool = false
) -> tuple[float, bool] {
    let a = if trans_a { Tensor([inner, rows]) } else { Tensor([rows, inner]) }
    let b = if trans_b { Tensor([columns, inner]) } else { Tensor([inner, columns]) }
    defer { a.free() }
    defer { b.free() }
    a.buffer().fill_uniform(-1.0, 1.0, 1)
    b.buffer().fill_uniform(-1.0, 1.0, 2)
    let wave = 8
    var outs: list[Tensor] = []
    for _i in 0..wave {
        outs.push(Tensor([rows, columns]))
    }
    defer {
        for product in outs { product.free() }
    }
    for index in 0..max(repeats, WARMUP) {
        matmul(a, b, trans_a: trans_a, trans_b: trans_b, out: outs[index % wave])
    }
    gpu.synchronize()
    let started = Instant.now()
    for index in 0..repeats {
        matmul(a, b, trans_a: trans_a, trans_b: trans_b, out: outs[index % wave])
    }
    gpu.synchronize()
    let elapsed = started.elapsed().to_seconds()
    let product = outs[(repeats - 1) % wave]
    let offsets = [0, rows // 2 * columns + columns // 2, rows * columns - 1]
    var nonzero = false
    var valid = true
    for offset in offsets {
        let value = product.buffer().download(offset, 1)[0]
        if not math.is_finite(value) or math.fabs(value) > float(inner) * 1.05 {
            valid = false
        }
        if math.fabs(value) > 1e-6 { nonzero = true }
    }
    return (elapsed, valid and nonzero)
}

fn main() -> int {
    if not is_available() { throw RuntimeError("bench needs a GPU") }
    set_mixed_precision(true)
    let level = env_or("BENCH_LEVEL", "hard")
    let repeats = if level == "easy" { 8 } elif level == "medium" { 20 } else { 40 }
    let batch = gemm_batch(level)
    let warmup_a = Tensor([64, 1024])
    let warmup_b = Tensor([1024, 64])
    let warmup = matmul(warmup_a, warmup_b)
    gpu.synchronize()
    warmup.free()
    warmup_a.free()
    warmup_b.free()
    let shapes = [
        ("fashion-L1", batch, 784, 256, false, false),
        ("fashion-W-TN", 784, batch, 256, true, false),
        ("cifar-L1", batch, 3072, 512, false, false),
        ("cifar-W-TN", 3072, batch, 512, true, false),
        ("cifar-X-NT", batch, 512, 3072, false, true),
        ("cifar-wide-L1", batch, 3072, 1024, false, false),
        ("cifar-wide-W-TN", 3072, batch, 1024, true, false),
        ("cifar100-wide-L1", 1024, 3072, 2048, false, false),
        ("cifar100-wide-W-TN", 3072, 1024, 2048, true, false),
        ("fashion-wide-L1", batch, 784, 2048, false, false),
        ("fashion-wide-W-TN", 784, batch, 2048, true, false),
        ("fashion-xl-L1", batch, 784, 4096, false, false),
        ("fashion-xl-W-TN", 784, batch, 4096, true, false),
        ("mnist-xl-L1", batch, 784, 8192, false, false),
    ]
    for (name, rows, inner, columns, trans_a, trans_b) in shapes {
        let counted = repeats_for(
            rows,
            inner,
            columns,
            max(repeats, MIN_REPEATS),
            trans_a: trans_a,
            trans_b: trans_b
        )
        let (seconds, valid) = time_shape(
            rows,
            inner,
            columns,
            counted,
            trans_a: trans_a,
            trans_b: trans_b
        )
        #: Reported as though it had run `repeats` times, whatever it
        #: actually ran. The calibration above picks a different count per
        #: shape and per run, and the suite needs the work figure to be a
        #: property of the shape -- otherwise every run disagrees with the
        #: last about how much work a row stands for, and the samples cannot
        #: be pooled. Scaling keeps the column meaning what it used to.
        let each = seconds / float(counted)
        let scaled = each * float(repeats)
        let flops = 2 * rows * inner * columns * repeats
        let check = if valid { "ok" } else { "invalid" }
        let backend = matmul_backend(rows, inner, columns, trans_a: trans_a, trans_b: trans_b)
        print(f"jtb\t{name}\t{scaled:.9f}\t{flops}\t0\t{check}\tamp-f16-f32/{backend}")
    }
    return 0
}
