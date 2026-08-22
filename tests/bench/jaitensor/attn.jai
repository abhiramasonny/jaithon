from jaitensor import Tensor, multi_head_attention, set_mixed_precision
from jaitensor import is_available
import std.gpu as gpu
import std.math
from std.os import env_or
from std.time import Instant

#: Dispatches to run before the clock starts. See gemm.jai.
let WARMUP = 24

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
    var live: list[Tensor] = []
    let started = Instant.now()
    for _i in 0..repeats {
        live.push(multi_head_attention(q, k, v, heads))
    }
    gpu.synchronize()
    let elapsed = started.elapsed().to_seconds()
    let output = live[live.len() - 1]
    let offsets = [0, output.size // 2, output.size - 1]
    var valid = true
    var nonzero = false
    for offset in offsets {
        let value = output.buffer().download(offset, 1)[0]
        if not math.is_finite(value) or math.fabs(value) > 1.01 { valid = false }
        if math.fabs(value) > 1e-6 { nonzero = true }
    }
    for y in live { y.free() }
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
        let (seconds, valid) = time_attn(seq, dim, heads, repeats)
        if verbose { print(f"{name}: {seconds / float(repeats):.6f}s") }
        let flops = 4 * seq * seq * dim * repeats
        let check = if valid { "ok" } else { "invalid" }
        print(f"jtb\t{name}\t{seconds:.9f}\t{flops}\t0\t{check}\tamp-f16-f32")
    }
    return 0
}
