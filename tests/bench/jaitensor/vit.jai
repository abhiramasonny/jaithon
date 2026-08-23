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
    inner_bias: Tensor,
    heads: int
) -> Tensor {
    let normed = jt.layer_norm(x, g1, b1)
    let q = jt.matmul(normed, qw)
    let attended = jt.multi_head_attention(q, q, q, heads)
    #: The projection and the residual it lands on, in one kernel.
    let residual = jt.linear_residual(attended, ow, x)
    let normed2 = jt.layer_norm(residual, g2, b2)
    #: One kernel for the product and the activation together, which is what
    #: the inner layer of a transformer is.
    let hidden = jt.linear_gelu(normed2, w1, inner_bias)
    let out = jt.linear_residual(hidden, w2, residual)
    #: Every intermediate handed back before the block returns.
    #:
    #: Device memory is not collected, so a `let` that goes out of scope here
    #: keeps its buffer for ever. Six of them a call, five of `[tokens, dim]`
    #: and one of `[tokens, 4 * dim]`, is twenty-eight megabytes a block that
    #: the pool never sees again -- so every call allocated fresh instead of
    #: recycling, and the timing was measuring the allocator.
    #:
    #: The peer frees these without being asked: its locals die at the end of
    #: the function and go straight back to torch's caching allocator. So
    #: this is what makes the two sides comparable, not a thumb on the scale
    #: -- it took the block from 74.4 ms to 58.5 ms and the comparison from
    #: 0.95x to about 1.3x. The gap it closes is real for anyone writing the
    #: same block by hand, though, and nothing in the language will free
    #: these for them.
    normed.free()
    q.free()
    attended.free()
    residual.free()
    normed2.free()
    hidden.free()
    return out
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
    let inner_bias = Tensor.zeros([dim * 4])
    for t in [x, qw, ow, w1, w2] { t.buffer().fill_uniform(-0.05, 0.05, 3) }

    for _w in 0..warmup {
        let out = block(x, qw, ow, w1, w2, g1, b1, g2, b2, inner_bias, heads)
        out.free()
    }
    gpu.synchronize()
    #: One result stays alive so the work cannot be elided, and each earlier
    #: one is handed straight back. The peer keeps exactly one too -- Python
    #: drops the previous on rebinding -- and holding every result instead put
    #: a working set of tens of megabytes between this side and its cache,
    #: which is a difference in the harness, not in the operation.
    let started = Instant.now()
    var last = block(x, qw, ow, w1, w2, g1, b1, g2, b2, inner_bias, heads)
    for _r in 1..repeats {
        let made = block(x, qw, ow, w1, w2, g1, b1, g2, b2, inner_bias, heads)
        last.free()
        last = made
    }
    gpu.synchronize()
    let seconds = started.elapsed().to_seconds()
    let check = last.buffer().download(0, 1)[0]
    #: Twenty products of `dim` by `dim` a token, plus the attention.
    #:
    #: Two for the query projection, two for the projection that lands on the
    #: residual, eight for the inner layer widening to four times `dim` and
    #: eight for the one bringing it back. Only one projection feeds the
    #: attention, not three -- the block passes the same tensor as query, key
    #: and value. This read twelve, which is what the block would cost if the
    #: inner layer were the same width as the outside, and made the rate
    #: column here and in the peer forty per cent low. Both sides said twelve,
    #: so the ratio between them was never wrong.
    let flops = float(repeats) *
        (20.0 * float(tokens) * float(dim) * float(dim) + 4.0 * float(tokens) * float(tokens) * float(dim))
    let sane = math.is_finite(check) and math.fabs(check) < 1000.0
    print(
        f"jtb\tvit-block\t{seconds:.9f}\t{int(flops)}\t0\t{if sane { "ok" } else { "invalid" }}\tamp-f16-f32"
    )
    return 0
}
