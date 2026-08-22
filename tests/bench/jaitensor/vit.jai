#: One transformer block forward -- what a vision transformer is a stack of.
#:
#: The pieces are timed on their own elsewhere; this is here because a block is
#: not its pieces. It found a normalisation running fourteen times slower than
#: it needed to, which none of the other benchmarks touch and which cost more
#: than every matrix product in the block put together.
import jaitensor as jt
import std.gpu as gpu
import std.math
from jaitensor import Tensor, set_mixed_precision
from std.os import env_or
from std.time import Instant

fn block(
    x: Tensor,
    qw: Tensor,
    ow: Tensor,
    w1: Tensor,
    w2: Tensor,
    g1: Tensor,
    b1: Tensor,
    g2: Tensor,
    b2: Tensor,
    heads: int
) -> Tensor {
    let normed = jt.layer_norm(x, g1, b1)
    let q = jt.matmul(normed, qw)
    let attended = jt.multi_head_attention(q, q, q, heads)
    let projected = jt.matmul(attended, ow)
    let residual = jt.add(x, projected)
    let normed2 = jt.layer_norm(residual, g2, b2)
    let hidden = jt.gelu(jt.matmul(normed2, w1))
    return jt.add(residual, jt.matmul(hidden, w2))
}

fn main() -> int {
    set_mixed_precision(true)
    let level = env_or("BENCH_LEVEL", "hard")
    let tokens = if level == "easy" { 256 } elif level == "medium" { 512 } else { 1024 }
    let dim = 768
    let heads = 12
    let repeats = if level == "easy" { 8 } elif level == "medium" { 16 } else { 24 }
    #: The same warm-up rule the other microbenchmarks use; see gemm.jai.
    let warmup = 24

    let x = Tensor([tokens, dim])
    let qw = Tensor([dim, dim])
    let ow = Tensor([dim, dim])
    let w1 = Tensor([dim, dim * 4])
    let w2 = Tensor([dim * 4, dim])
    let g1 = Tensor.full([dim], 1.0)
    let b1 = Tensor.zeros([dim])
    let g2 = Tensor.full([dim], 1.0)
    let b2 = Tensor.zeros([dim])
    for t in [x, qw, ow, w1, w2] { t.buffer().fill_uniform(-0.05, 0.05, 3) }

    for _w in 0..warmup {
        let out = block(x, qw, ow, w1, w2, g1, b1, g2, b2, heads)
        out.free()
    }
    gpu.synchronize()
    let started = Instant.now()
    var live: list[Tensor] = []
    for _r in 0..repeats { live.push(block(x, qw, ow, w1, w2, g1, b1, g2, b2, heads)) }
    gpu.synchronize()
    let seconds = started.elapsed().to_seconds()
    let check = live[live.len() - 1].buffer().download(0, 1)[0]
    let flops = float(repeats) *
        (12.0 * float(tokens) * float(dim) * float(dim) + 4.0 * float(tokens) * float(tokens) * float(dim))
    let sane = math.is_finite(check) and math.fabs(check) < 1000.0
    print(
        f"jtb\tvit-block\t{seconds:.9f}\t{int(flops)}\t0\t{if sane { "ok" } else { "invalid" }}\tamp-f16-f32"
    )
    return 0
}
