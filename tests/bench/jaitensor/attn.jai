from jaitensor import Tensor, multi_head_attention, set_mixed_precision
from jaitensor import is_available
import std.gpu as gpu
import std.math
from std.os import env_or
from std.time import Instant

#: Dispatches to run before the clock starts. See gemm.jai.
let WARMUP = 24

#: How long each shape should be timed for, in seconds.
#:
#: A fixed repeat count times the small shapes for almost no time at all: 24
#: repeats of `attn-256` is two and a half milliseconds, which is launch and
#: synchronisation noise with a kernel somewhere inside it, and its run-to-run
#: spread reached 28% -- enough to move the shape from a win to a loss and
#: back between two runs of the same binary. Timing every shape for the same
#: DURATION instead puts them all on the same footing; the peer calibrates
#: against the same target, so each side measures itself for a sixth of a
#: second and the comparison is between per-call times.
let TARGET_SECONDS = 0.15
let MIN_REPEATS = 24
let MAX_REPEATS = 4096

#: How many repeats of this shape fill `TARGET_SECONDS`, from a short probe.
fn repeats_for(seq: int, dim: int, heads: int, floor: int) -> int {
    let q = Tensor([seq, dim])
    let k = Tensor([seq, dim])
    let v = Tensor([seq, dim])
    defer { q.free() }
    defer { k.free() }
    defer { v.free() }
    for _warm in 0..WARMUP {
        let throwaway = multi_head_attention(q, k, v, heads)
        throwaway.free()
    }
    gpu.synchronize()
    let started = Instant.now()
    for _probe in 0..8 {
        let throwaway = multi_head_attention(q, k, v, heads)
        throwaway.free()
    }
    gpu.synchronize()
    let each = started.elapsed().to_seconds() / 8.0
    if each <= 0.0 { return MAX_REPEATS }
    let wanted = int(TARGET_SECONDS / each)
    if wanted < floor { return floor }
    if wanted > MAX_REPEATS { return MAX_REPEATS }
    return wanted
}

fn time_attn(seq: int, dim: int, heads: int, repeats: int) -> tuple[float, bool] {
    let q = Tensor([seq, dim])
    let k = Tensor([seq, dim])
    let v = Tensor([seq, dim])
    defer { q.free() }
    defer { k.free() }
    defer { v.free() }
    q.buffer().fill_uniform(-1.0, 1.0, 1)
    k.buffer().fill_uniform(-1.0, 1.0, 2)
    v.buffer().fill_uniform(-1.0, 1.0, 3)
    #: The GPU does not run at its working speed until it has been busy for a
    #: while, and one call is nowhere near enough. See the note in gemm.jai:
    #: timing the first few dispatches measures the clock coming up rather than
    #: the kernel. The peer benchmark warms up by the same rule.
    for _warm in 0..max(repeats, WARMUP) {
        let throwaway = multi_head_attention(q, k, v, heads)
        throwaway.free()
    }
    gpu.synchronize()
    #: One result stays alive so the work cannot be elided, and each earlier
    #: one is handed straight back. The peer keeps exactly one too -- Python
    #: drops the previous on rebinding -- and holding every result instead put
    #: a working set of tens of megabytes between this side and its cache,
    #: which is a difference in the harness, not in the operation.
    let started = Instant.now()
    var output = multi_head_attention(q, k, v, heads)
    for _i in 1..repeats {
        let made = multi_head_attention(q, k, v, heads)
        output.free()
        output = made
    }
    gpu.synchronize()
    let elapsed = started.elapsed().to_seconds()
    let offsets = [0, output.size // 2, output.size - 1]
    var valid = true
    var nonzero = false
    for offset in offsets {
        let value = output.buffer().download(offset, 1)[0]
        if not math.is_finite(value) or math.fabs(value) > 1.01 { valid = false }
        if math.fabs(value) > 1e-6 { nonzero = true }
    }
    output.free()
    return (elapsed, valid and nonzero)
}

fn main() -> int {
    if not is_available() { throw RuntimeError("bench needs a GPU") }
    set_mixed_precision(true)
    let level = env_or("BENCH_LEVEL", "hard")
    let verbose = env_or("JAITENSOR_VERBOSE", "0") != "0"
    let repeats = if level == "easy" { 8 } elif level == "medium" { 16 } else { 24 }
    let warmup_q = Tensor([64, 256])
    let warmup_k = Tensor([64, 256])
    let warmup_v = Tensor([64, 256])
    let warmup = multi_head_attention(warmup_q, warmup_k, warmup_v, 4)
    gpu.synchronize()
    warmup.free()
    warmup_q.free()
    warmup_k.free()
    warmup_v.free()
    let shapes = if level == "easy" {
        [("attn-256", 256, 256, 8), ("attn-512", 512, 512, 8), ("attn-1024", 1024, 512, 8)]
    } else {
        [("attn-256", 256, 256, 8), ("attn-512", 512, 512, 8), ("attn-1024", 1024, 512, 8), ("attn-1024-wide", 1024, 1024, 16), ("attn-2048", 2048, 1024, 16), ("attn-4096", 4096, 1024, 16)]
    }
    for (name, seq, dim, heads) in shapes {
        let counted = repeats_for(seq, dim, heads, max(repeats, MIN_REPEATS))
        let (seconds, valid) = time_attn(seq, dim, heads, counted)
        if verbose { print(f"{name}: {seconds / float(counted):.6f}s") }
        #: Reported as though it had run `repeats` times; see gemm.jai.
        let scaled = seconds / float(counted) * float(repeats)
        let flops = 4 * seq * seq * dim * repeats
        let check = if valid { "ok" } else { "invalid" }
        print(f"jtb\t{name}\t{scaled:.9f}\t{flops}\t0\t{check}\tamp-f16-f32")
    }
    return 0
}
