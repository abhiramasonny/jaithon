#: The tensor operations a model is built out of between its matmuls.
#:
#: The rest of this suite measures training steps, GEMM shapes, convolutions
#: and attention, which is where most of the arithmetic is. It is not where
#: all of the time is: a layout swap, a pad, a pooling window or a repeat
#: moves as many bytes as a small matmul and none of it was measured. Four of
#: the rows below were between a third and two thirds of the peer's speed when
#: they were first timed, and nothing in the suite would have said so.
#:
#: Every row is a single operation over a fixed shape, timed the way the rest
#: of the suite times: warm up, then a run long enough that the clock is not
#: the measurement. The work figure reported is bytes moved rather than
#: floating-point operations, because that is what these are limited by; the
#: suite prints it through the GFLOPS column all the same, so read that column
#: here as gigabytes a second.
import std.math
import std.gpu as gpu
from jaitensor import Tensor, is_available, set_mixed_precision
from jaitensor.ops import avg_pool2d, cat, gather_rows, gelu, max_pool2d, nchw, nhwc, pad2d, repeat, softmax, sum as reduce_sum, transpose, upsample_nearest2d
from std.os import env_or
from std.time import Instant

#: Long enough that a run says something about the kernel rather than about
#: the clock, short enough that twelve rows do not take a minute.
let TARGET_SECONDS = 0.08
let MIN_REPEATS = 30
let MAX_REPEATS = 2048
let WARMUP = 40

fn filled(shape: list[int], seed: int) -> Tensor {
    let held = Tensor(shape)
    held.buffer().fill_uniform(-1.0, 1.0, seed)
    return held
}

#: One reading, and whether the result looks like a result.
#:
#: The check reads three places rather than the whole thing: a kernel that
#: writes nothing leaves the buffer as it was allocated, and a kernel that
#: writes rubbish tends to write it everywhere.
fn once(work: fn() -> Tensor) -> bool {
    let product = work()
    var sane = product.size > 0
    let offsets = [0, product.size // 2, product.size - 1]
    for offset in offsets {
        let value = product.buffer().download(offset, 1)[0]
        if not math.is_finite(value) { sane = false }
    }
    product.free()
    return sane
}

fn repeats_for(work: fn() -> Tensor) -> int {
    for _warm in 0..8 {
        work().free()
    }
    gpu.synchronize()
    let started = Instant.now()
    for _probe in 0..8 {
        work().free()
    }
    gpu.synchronize()
    let each = started.elapsed().to_seconds() / 8.0
    if each <= 0.0 { return MAX_REPEATS }
    let wanted = int(TARGET_SECONDS / each)
    if wanted < MIN_REPEATS { return MIN_REPEATS }
    if wanted > MAX_REPEATS { return MAX_REPEATS }
    return wanted
}

#: `bytes_moved` is what one call reads plus what it writes, which is the
#: figure these are actually limited by.
fn run(name: str, bytes_moved: int, work: fn() -> Tensor) -> void {
    let sane = once(work)
    let counted = repeats_for(work)
    for _warm in 0..WARMUP {
        work().free()
    }
    gpu.synchronize()
    let started = Instant.now()
    for _r in 0..counted {
        work().free()
    }
    gpu.synchronize()
    let seconds = started.elapsed().to_seconds()
    #: Reported as though it had run `MIN_REPEATS` times, whatever it actually
    #: ran, so the work figure stays a property of the shape and the samples
    #: from different runs can be pooled. See the note in gemm.jai.
    let scaled = seconds / float(counted) * float(MIN_REPEATS)
    let moved = bytes_moved * MIN_REPEATS
    let check = if sane { "ok" } else { "invalid" }
    print(f"jtb\t{name}\t{scaled:.9f}\t{moved}\t0\t{check}\tbytes/elementwise")
}

fn main() -> int {
    if not is_available() { throw RuntimeError("bench needs a GPU") }
    set_mixed_precision(true)
    let level = env_or("BENCH_LEVEL", "hard")
    let scale = if level == "easy" { 4 } elif level == "medium" { 2 } else { 1 }

    let rows = 4096 // scale
    let wide = filled([rows, 4096], 1)
    let images = filled([64 // scale, 56, 56, 64], 2)
    let planar = nchw(images)
    let table = filled([50000, 512], 3)
    let ids = Tensor([rows])
    ids.buffer().fill_uniform(0.0, 49999.0, 4)
    let left = filled([rows, 2048], 5)
    let right = filled([rows, 2048], 6)
    let small = filled([64 // scale, 32, 32, 64], 7)

    let word = 4
    let flat = wide.size * word
    let picture = images.size * word

    run("op-softmax", flat * 2, || softmax(wide))
    run("op-transpose", flat * 2, || transpose(wide))
    run("op-row-sum", flat + rows * word, || reduce_sum(wide, axis: 1))
    run("op-gelu", flat * 2, || gelu(wide))
    run("op-to-nchw", picture * 2, || nchw(images))
    run("op-to-nhwc", picture * 2, || nhwc(planar))
    run("op-pad", picture + 64 // scale * 58 * 58 * 64 * word, || pad2d(images, [1, 1, 1, 1]))
    run("op-max-pool", picture + picture // 4, || max_pool2d(images, 2))
    run("op-avg-pool", picture + picture // 4, || avg_pool2d(images, 2))
    run("op-upsample", small.size * word * 5, || upsample_nearest2d(small, 2))
    run("op-gather", rows * 512 * word * 2, || gather_rows(table, ids))
    run("op-cat", left.size * word * 4, || cat([left, right], axis: 1))
    run("op-repeat", left.size * word * 5, || repeat(left, 4))
    return 0
}
