#: The array operations a numpy program spends its time in.
#:
#: `jainum` had 290 tests and no benchmark at all, so nothing said whether any
#: of it was faster than the library it copies. Every row here is one call on
#: one shape, chosen for how often numpy programs make it and how many bytes it
#: moves rather than for how interesting it is.
#:
#: numpy runs on the CPU and this runs on the GPU, so a row that loses is a
#: real loss and a row that wins by less than the dispatch floor -- about
#: 0.15 ms of command-buffer round trip -- is not really a win. Small shapes
#: are kept out for that reason: they measure the round trip, not the kernel.
#:
#: The work figure is bytes moved, not floating-point operations, because most
#: of these are bandwidth-bound; the suite prints it through the GFLOPS column,
#: so read that column here as gigabytes a second. `matmul` is the exception
#: and is reported the same way for consistency.
import std.math
import std.gpu as gpu
import jainum as nb
import jainum.random as nbr
from jainum import NDArray
from std.os import argv, env_or
from std.time import Instant

let TARGET_SECONDS = 0.08
let MIN_REPEATS = 30
let MAX_REPEATS = 2048
#: How long the device must have been working before a row is timed.
#:
#: A FLOOR, not a round count. A fixed number of warm-up calls does not climb
#: the GPU clock: forty calls of a row that takes a fifth of a millisecond is
#: eight milliseconds of work, and every row here runs in its own cold process.
#: The jaicv suite was measured both ways and its `blur-3` read 0.4476 ms warm
#: by count and 0.1854 ms warm by clock -- the first of those was reported as a
#: loss for months. The peer warms the same way; a CPU has no ramp, so it costs
#: it only time.
let WARM_SECONDS = 0.25

#: A stop, so a row that is somehow instant cannot spin in the warm-up.
let WARM_MAX_ROUNDS = 20000

#: Four bytes an element: every array here is float32.
let WIDTH = 4

#: One row a process, the two sides turn about. The jaicv suite learned this
#: the hard way -- a row read 1.32x as the last of twenty-one and 0.75x with
#: nine more rows in front of it, unchanged, because our side had held the
#: device flat out by the time it came up while the CPU peer reached it cool.
var _only = ""
var _listing = false

fn wanted(name: str) -> bool {
    if _listing {
        print(name)
        return false
    }
    return _only == "" or _only == name
}

fn once(work: fn() -> NDArray) -> bool {
    let product = work()
    var sane = product.size() > 0
    let offsets = [0, product.size() // 2, product.size() - 1]
    for offset in offsets {
        let value = product.storage.download(product.offset + offset, 1)[0]
        if not math.is_finite(value) { sane = false }
    }
    product.free()
    return sane
}

fn repeats_for(work: fn() -> NDArray) -> int {
    for _warm in 0..8 { work().free() }
    gpu.synchronize()
    let started = Instant.now()
    for _probe in 0..8 { work().free() }
    gpu.synchronize()
    let each = started.elapsed().to_seconds() / 8.0
    if each <= 0.0 { return MAX_REPEATS }
    let wanted = int(TARGET_SECONDS / each)
    if wanted < MIN_REPEATS { return MIN_REPEATS }
    if wanted > MAX_REPEATS { return MAX_REPEATS }
    return wanted
}

fn run(name: str, bytes_moved: int, work: fn() -> NDArray) -> void {
    if not wanted(name) { return }
    let sane = once(work)
    let counted = repeats_for(work)
    let opening = Instant.now()
    for round in 0..WARM_MAX_ROUNDS {
        work().free()
        if round % 4 == 3 {
            gpu.synchronize()
            if opening.elapsed().to_seconds() >= WARM_SECONDS { break }
        }
    }
    gpu.synchronize()
    let started = Instant.now()
    for _r in 0..counted { work().free() }
    gpu.synchronize()
    let seconds = started.elapsed().to_seconds()
    let scaled = seconds / float(counted) * float(MIN_REPEATS)
    let moved = bytes_moved * MIN_REPEATS
    let check = if sane { "ok" } else { "invalid" }
    print(f"jtb\t{name}\t{scaled:.9f}\t{moved}\t0\t{check}\tbytes/elementwise")
}

fn main() -> int {
    let given = argv()
    if given.len() > 1 {
        if given[1] == "list" { _listing = true } else { _only = given[1] }
    }
    if not _listing and not gpu.is_available() { throw RuntimeError("bench needs a GPU") }
    let level = env_or("BENCH_LEVEL", "hard")
    let scale = if level == "easy" { 4 } elif level == "medium" { 2 } else { 1 }

    let side = 4096 // scale
    let square = side * side
    let gemm_side = 2048 // scale
    let planes = 32 // scale
    let plane_side = 256

    nbr.seed(20260824)
    let wide = nbr.standard_normal([side, side])
    let other = nbr.standard_normal([side, side])
    let row = nbr.standard_normal([side])
    let left = nbr.standard_normal([gemm_side, gemm_side])
    let right = nbr.standard_normal([gemm_side, gemm_side])
    let stack_a = nbr.standard_normal([planes, plane_side, plane_side])
    let stack_b = nbr.standard_normal([planes, plane_side, plane_side])
    let table = nbr.standard_normal([50000, 512])
    let picks = nbr.integers(0, 50000, [side])
    let tall = nbr.standard_normal([1024 // scale, side])

    let gemm_bytes = gemm_side * gemm_side * 3 * WIDTH
    let batch_bytes = planes * plane_side * plane_side * 3 * WIDTH
    let pair_bytes = square * 3 * WIDTH
    let unary_bytes = square * 2 * WIDTH
    let reduce_bytes = square * WIDTH
    let sort_bytes = 1024 // scale * side * 2 * WIDTH

    run("matmul", gemm_bytes, || nb.matmul(left, right))
    run("matmul-batched", batch_bytes, || nb.matmul(stack_a, stack_b))
    run("add", pair_bytes, || nb.add(wide, other))
    run("add-broadcast", pair_bytes, || nb.add(wide, row))
    run("multiply", pair_bytes, || nb.multiply(wide, other))
    run("exp", unary_bytes, || nb.exp(wide))
    #: A row against every row of a matrix -- a bias, a normalisation -- which
    #: takes a different path from the full-size pair above it.
    run("maximum-broadcast", pair_bytes, || nb.maximum(wide, row))
    run("sum-axis0", reduce_bytes, || nb.sum(wide, axis: 0))
    run("sum-axis1", reduce_bytes, || nb.sum(wide, axis: 1))
    run("argmax-axis1", reduce_bytes, || nb.argmax(wide, axis: 1))
    run("transpose-copy", unary_bytes, || nb.transpose(wide).copy())
    run("cumsum-axis1", unary_bytes, || nb.cumsum(wide, axis: 1))
    run("sort-axis1", sort_bytes, || nb.sort(tall, axis: 1))
    run("take-rows", side * 512 * 2 * WIDTH, || nb.take(table, picks, axis: 0))
    #: Timed apart, because one row that runs two operations reports the bytes
    #: of one and reads as a loss that is not there.
    let mask = nb.greater(wide, other)
    run("greater", pair_bytes, || nb.greater(wide, other))
    run("where", pair_bytes + reduce_bytes, || nb.where(mask, wide, other))

    mask.free()
    for held in [wide, other, row, left, right, stack_a, stack_b, table, picks, tall] {
        held.free()
    }
    return 0
}
