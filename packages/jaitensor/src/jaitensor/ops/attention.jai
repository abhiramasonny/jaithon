#: Scaled dot-product and multi-head attention: the fused kernels, and the
#: op compositions a backward pass needs.
import std.gpu as gpu
import std.math
from std.os import env
from ..tensor import GradNode, Tensor
from .elementwise import scale
from .kernels import cached_attn_kernel, f32_bits
from .matmul import matmul, raw_matmul, raw_matmul_batched
from .reduce import softmax
from .shape import cat, merge_heads, reshape, split_heads

#: Scaled dot-product attention for rank-two `[seq, dim]` tensors.
pub fn scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor {
    if query.rank() != 2 or key.rank() != 2 or value.rank() != 2 {
        throw ValueError("scaled_dot_product_attention needs rank-two Q, K, V")
    }
    if query.shape[1] != key.shape[1] {
        throw ValueError("query/key depth mismatch")
    }
    if key.shape[0] != value.shape[0] {
        throw ValueError("key/value sequence mismatch")
    }
    let factor = 1.0 / math.sqrt(float(query.shape[1]))
    if query.requires_grad or key.requires_grad or value.requires_grad {
        return matmul(softmax(scale(matmul(query, key, trans_b: true), factor)), value)
    }
    let scores = raw_matmul(query, key, trans_b: true)
    let scaled = scale(scores, factor)
    scores.free()
    let probs = softmax(scaled)
    scaled.free()
    let output = raw_matmul(probs, value)
    probs.free()
    return output
}

#: Per-head scaled-dot-product attention on packed `[seq, dim]` Q/K/V.
#:
#: `query` is `[tq, dim]` and `key`/`value` are `[tk, dim]`; the two lengths
#: may differ, which is what decoding against cached keys needs. Rank three
#: `[batch, t, dim]` folds the batch into the head axis, so every sequence in
#: the batch is attended separately in one dispatch.
#:
#: `causal` masks query row `i` to key rows `0..=i + tk - tq`: with equal
#: lengths that is the usual lower triangle, and a query block that continues
#: a longer key sequence sees its own prefix. It needs `tk >= tq`. That is the
#: bottom-right alignment; torch's `is_causal=True` aligns top left, and the
#: two differ whenever `tq < tk`.
#:
#: The fused device path is a Metal prefill kernel: packed `[seq, heads, hd]`
#: stays packed, Q/K/V stream through half SRAM, and QK/PV use simdgroup MMA.
#: A single query row takes a decode kernel that splits the keys across
#: threadgroups instead. MPSGraph SDPA is the fallback when neither can run.
#: Last resort is GEMM. Scale is `1 / sqrt(head_dim)`.
pub fn multi_head_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    heads: int,
    causal: bool = false
) -> Tensor {
    if query.rank() == 3 or key.rank() == 3 or value.rank() == 3 {
        return mha_batched(query, key, value, heads, causal)
    }
    if heads == 1 and not causal { return scaled_dot_product_attention(query, key, value) }
    if query.rank() != 2 or key.rank() != 2 or value.rank() != 2 {
        throw ValueError("multi_head_attention needs rank-two Q, K, V")
    }
    if heads <= 0 {
        throw ValueError(f"multi_head_attention needs a positive head count, got {heads}")
    }
    if key.shape != value.shape or query.shape[1] != key.shape[1] {
        throw ValueError(
            f"multi_head_attention Q/K/V shape mismatch: {query.shape} {key.shape} {value.shape}"
        )
    }
    let seq = query.shape[0]
    let kv_seq = key.shape[0]
    let dim = query.shape[1]
    if dim % heads != 0 {
        throw ValueError(f"multi_head_attention dim {dim} must divide heads {heads}")
    }
    if causal and kv_seq < seq {
        throw ValueError(
            f"causal multi_head_attention needs at least as many keys as queries, got {kv_seq} < {seq}"
        )
    }
    let hd = dim // heads
    let factor = 1.0 / math.sqrt(float(hd))
    if query.requires_grad or key.requires_grad or value.requires_grad {
        if seq == kv_seq and mha_mma_ok(seq, hd) {
            return mha_by_mma(query, key, value, heads, seq, hd, factor, causal)
        }
        return mha_by_ops(query, key, value, heads, seq, kv_seq, hd, factor, causal)
    }
    let output = Tensor(query.shape)
    if gpu.mha_async(
        query.buffer(),
        key.buffer(),
        value.buffer(),
        output.buffer(),
        seq,
        heads,
        hd,
        factor,
        causal: causal,
        kv_seq: kv_seq
    ) {
        return output
    }
    if causal or seq != kv_seq {
        output.free()
        return mha_by_ops(query, key, value, heads, seq, kv_seq, hd, factor, causal)
    }
    if seq >= 16 and hd >= 32 and hd <= 64 {
        let q_tiles = (seq + 15) // 16
        let kernel = cached_attn_kernel("flash_mha_wide")
        kernel.dispatch_async(
            [query.buffer(), key.buffer(), value.buffer(), output.buffer()],
            q_tiles * heads * 256,
            [seq, heads, hd, f32_bits(factor)],
            group_size: 256
        )
        return output
    }
    if seq >= 16 and hd <= 64 {
        let q_tiles = (seq + 15) // 16
        let kernel = cached_attn_kernel("flash_mha")
        kernel.dispatch_async(
            [query.buffer(), key.buffer(), value.buffer(), output.buffer()],
            q_tiles * heads * 256,
            [seq, heads, hd, f32_bits(factor)],
            group_size: 256
        )
        return output
    }
    output.free()
    return mha_by_gemm(query, key, value, heads, seq, hd, factor)
}

#: Rank-three attention: `[batch, t, dim]`, one sequence per batch row.
#:
#: The batch folds into the head axis. `[batch, t, dim]` read as
#: `[batch * t, dim]` is exactly `merge_heads`'s packed layout with `batch`
#: heads of width `dim`, so one `merge_heads` gives `[t, batch * dim]`, whose
#: columns are `batch * heads` heads of the original width. Attention runs
#: once over all of them and `split_heads` undoes the fold. Both copies carry
#: their own gradients, so this is differentiable whenever the inputs are.
fn mha_batched(query: Tensor, key: Tensor, value: Tensor, heads: int, causal: bool) -> Tensor {
    if query.rank() != 3 or key.rank() != 3 or value.rank() != 3 {
        throw ValueError(
            f"batched multi_head_attention needs rank-three Q, K, V, got {query.shape} {key.shape} {value.shape}"
        )
    }
    if key.shape != value.shape or query.shape[0] != key.shape[0] or query.shape[2] != key.shape[2] {
        throw ValueError(
            f"multi_head_attention Q/K/V shape mismatch: {query.shape} {key.shape} {value.shape}"
        )
    }
    let batch = query.shape[0]
    let seq = query.shape[1]
    let kv_seq = key.shape[1]
    let dim = query.shape[2]
    let qf = fold_batch(query)
    let kf = fold_batch(key)
    let vf = fold_batch(value)
    let folded = multi_head_attention(qf, kf, vf, heads * batch, causal: causal)
    hand_to_graph(folded, [qf, kf, vf])
    let split = split_heads(folded, batch)
    hand_to_graph(split, [folded])
    if split.creator is not null { return reshape(split, [batch, seq, dim]) }
    #: Unrecorded, `split` already holds the answer in `[batch, seq, dim]`
    #: order; the result adopts its buffer rather than viewing it, since a
    #: view would dangle once `split` was freed.
    return Tensor([batch, seq, dim], storage: split.buffer())
}

#: `[batch, t, dim]` to `[t, batch * dim]`.
fn fold_batch(input: Tensor) -> Tensor {
    let batch = input.shape[0]
    let seq = input.shape[1]
    let flat = reshape(input, [batch * seq, input.shape[2]])
    let folded = merge_heads(flat, batch, seq)
    release_unrecorded(flat)
    return folded
}

#: Free an intermediate unless an autograd graph recorded it; a recorded one
#: is released by whoever frees what `backward()` or `discard_graph()` returns.
fn release_unrecorded(tensor: Tensor) -> void {
    if tensor.creator is null { tensor.free() }
}

#: Settle the intermediates `consumer` was computed from, once nothing else
#: will read them in the forward pass.
#:
#: One a graph recorded already belongs to it. One no graph recorded may still
#: be read by `consumer`'s backward: an input with no gradient of its own
#: beside one that has one, such as a frozen projection, or the copy of a
#: plain key next to a query that trains. Freeing that made the backward
#: throw, and keeping it without telling the graph leaked it, since
#: `backward()` returns only what has a creator. So it rides on `consumer`'s
#: node as an extra and comes back with the rest. Only when `consumer`
#: recorded nothing is it freed here.
pub fn hand_to_graph(consumer: Tensor, tensors: list[Tensor]) -> void {
    let node = consumer.creator
    for tensor in tensors {
        if tensor.creator is not null { continue }
        if node is not null {
            node.extras.push(tensor)
        } else {
            tensor.free()
        }
    }
}

#: Whether the attention softmax kernels keep a row in registers between their
#: passes. Off by default: `JAITENSOR_ATTN_SOFTMAX_HOLD=1` takes the `_held`
#: kernels, for an A/B in one binary. Holding measured about 10% faster on
#: the plain softmax kernels at seq 1024 but neutral end to end, and 7-10%
#: slower end to end on a 512/512/8 training step. The answer is the same bit
#: for bit at head width 64, whose scale is exact, and within about an ulp at
#: other widths, where the compiler fuses the two spellings differently. Read
#: once.
var _softmax_hold: int = -1

fn softmax_entry(name: str) -> str {
    if _softmax_hold < 0 {
        let text = env("JAITENSOR_ATTN_SOFTMAX_HOLD")
        _softmax_hold = if text is not null and text == "1" { 1 } else { 0 }
    }
    return if _softmax_hold == 1 { name + "_held" } else { name }
}

#: Threadgroups `mha_scores_mma` needs per head: every 64x64 patch, or with a
#: causal mask only those on or below the diagonal.
fn score_patches(tiles: int, causal: bool) -> int {
    if causal { return tiles * (tiles + 1) // 2 }
    return tiles * tiles
}

#: Whether the batched simdgroup-matrix attention kernels can take this shape.
#:
#: They tile in 8x8 fragments with no partial-tile masking, so the sequence
#: has to fill whole 64-row patches, and the head dimension has to be exactly
#: the 64 the 2x2 of simdgroups covers. A causal mask is whole tiles plus the
#: diagonal one, which the softmax handles. Everything else keeps
#: `mha_by_ops`, which has no shape restriction at all.
fn mha_mma_ok(seq: int, hd: int) -> bool {
    return hd == 64 and seq % 64 == 0
}

#: Multi-head attention as six batched dispatches, forward and backward.
#:
#: `mha_by_ops` below is the same arithmetic written as ops, and it is what
#: runs whenever `mha_mma_ok` says no. What this saves over it is not a
#: different algorithm but a different shape of work. That version issues one
#: GEMM per head, and the route the GEMM router picks for the two attention
#: shapes -- [seq, 64] against [64, seq], and [seq, seq] against [seq, 64] --
#: reaches 2.3 and 2.8 TFLOP/s where a square 1024 product reaches 8.3 on the
#: same device. At seq 1024, dim 1024 and 16 heads the whole forward and
#: backward took 8.39 ms that way, of which the two forward loops alone were
#: 0.93 and 0.76. The six dispatches here measure 2.25 together, dropping
#: `split_heads` and `merge_heads` -- the kernels read the packed layout
#: directly -- takes another 0.31 off, and the pass now measures 3.15 against
#: torch's 4.36 for the same shape, both timed with the graph rebuilt each
#: iteration because backward consumes it.
#:
#: The score matrix is still written out, unlike the fused inference kernel,
#: because the backward pass has to read it. Nothing else is materialised.
#:
#: Causal, every kernel skips the 64x64 patches above the diagonal: the score
#: products are not computed, the softmax zeroes only up to the edge of the
#: diagonal patch, and the context and transposed products stop (or start) at
#: that patch. Nothing ever reads the skipped patches, so they are never
#: written either, and at seq 1024 that is 120 of 256 patches per head.
fn mha_by_mma(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    heads: int,
    seq: int,
    hd: int,
    factor: float,
    causal: bool
) -> Tensor {
    let tiles = seq // 64
    let mask = if causal { 1 } else { 0 }
    let causal_rows = if causal { seq } else { 0 }
    let probs = Tensor([heads * seq, seq])
    cached_attn_kernel("mha_scores_mma").dispatch_async(
        [query.buffer(), key.buffer(), probs.buffer()],
        score_patches(tiles, causal) * heads * 64,
        [seq, heads, hd, tiles, mask],
        group_size: 64
    )
    cached_attn_kernel(softmax_entry("mha_softmax_scaled")).dispatch_async(
        [probs.buffer()],
        heads * seq * 256,
        [heads * seq, seq, f32_bits(factor), causal_rows, 64],
        group_size: 256
    )
    let output = Tensor([seq, heads * hd])
    cached_attn_kernel("mha_context_mma").dispatch_async(
        [probs.buffer(), value.buffer(), output.buffer()],
        tiles * heads * 64,
        [seq, heads, hd, mask],
        group_size: 64
    )
    if query.requires_grad or key.requires_grad or value.requires_grad {
        output.attach_creator(GradNode([query, key, value], fn(upstream: Tensor) -> list[Tensor] {
            return mha_mma_grads(upstream, query, key, value, probs, heads, seq, hd, factor, causal)
        }, extras: [probs]))
    }
    return output
}

#: Gradients of `mha_by_mma`, in four dispatches around the softmax backward.
#:
#: The four products are the ones `matmul`'s backward would have chosen, in
#: the same operand order; only the batching and the layout differ. Each
#: gradient comes out packed like the input it belongs to, so nothing has to
#: be un-split afterwards. `grad_probs` carries d(probs) into the softmax
#: backward and comes back holding d(scores), which saves a second
#: sixty-seven-megabyte buffer at seq 1024.
fn mha_mma_grads(
    upstream: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    probs: Tensor,
    heads: int,
    seq: int,
    hd: int,
    factor: float,
    causal: bool
) -> list[Tensor] {
    let tiles = seq // 64
    let dim = heads * hd
    let mask = if causal { 1 } else { 0 }
    let causal_rows = if causal { seq } else { 0 }
    let grad_q = Tensor([seq, dim])
    let grad_k = Tensor([seq, dim])
    let grad_v = Tensor([seq, dim])
    let grad_probs = Tensor([heads * seq, seq])
    cached_attn_kernel("mha_scores_mma").dispatch_async(
        [upstream.buffer(), value.buffer(), grad_probs.buffer()],
        score_patches(tiles, causal) * heads * 64,
        [seq, heads, hd, tiles, mask],
        group_size: 64
    )
    cached_attn_kernel("mha_transposed_mma").dispatch_async(
        [probs.buffer(), upstream.buffer(), grad_v.buffer()],
        tiles * heads * 64,
        [seq, heads, hd, mask],
        group_size: 64
    )
    cached_attn_kernel(softmax_entry("mha_softmax_backward_scaled")).dispatch_async(
        [probs.buffer(), grad_probs.buffer()],
        heads * seq * 256,
        [heads * seq, seq, f32_bits(factor), causal_rows, 64],
        group_size: 256
    )
    cached_attn_kernel("mha_context_mma").dispatch_async(
        [grad_probs.buffer(), key.buffer(), grad_q.buffer()],
        tiles * heads * 64,
        [seq, heads, hd, mask],
        group_size: 64
    )
    cached_attn_kernel("mha_transposed_mma").dispatch_async(
        [grad_probs.buffer(), query.buffer(), grad_k.buffer()],
        tiles * heads * 64,
        [seq, heads, hd, mask],
        group_size: 64
    )
    grad_probs.free()
    return [grad_q, grad_k, grad_v]
}

#: Multi-head attention built from differentiable ops.
#:
#: The fused kernels above never materialise the attention probabilities, so
#: there is nothing for a backward pass to read; a graph that needs gradients
#: gets this composition instead. Every head's scores live in one
#: `[heads * seq, seq]` tensor and every head's context in one
#: `[heads * seq, hd]` tensor, so the only per-head work is the four GEMMs
#: that must be per-head; the softmax, the scale and the packing are one
#: dispatch each across all of them.
#:
#: The composition this replaced sliced Q, K and V per head with `slice_flat`
#: and reassembled the result with `cat`. Forward that is nearly free -- a
#: slice is a view -- but `slice_flat`'s backward allocates a zero tensor the
#: size of the whole packed activation and copies one head into it, and the
#: three slices per head over sixteen heads then arrive at `qs`, `ks` and `vs`
#: as forty-eight full-size gradients to be summed. At seq 1024 that pattern
#: alone measured 1.59 ms: forty-eight four-megabyte zero fills and
#: forty-eight adds, every one of which adds a head to forty-seven zeros. The
#: GEMMs below write into a view of the single gradient buffer instead, so
#: each element is written exactly once. Adding zero is exact, so the answer
#: is unchanged bit for bit.
#:
#: `seq` query rows attend `kv_seq` key rows. A causal mask is applied by the
#: softmax, which zeroes every masked entry so the GEMMs reading the whole
#: matrix see nothing past each row's limit; it also serves as the inference
#: route for a causal or uneven shape the device kernels cannot take.
fn mha_by_ops(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    heads: int,
    seq: int,
    kv_seq: int,
    hd: int,
    factor: float,
    causal: bool
) -> Tensor {
    let causal_rows = if causal { seq } else { 0 }
    let qs = split_heads(query, heads)
    let ks = split_heads(key, heads)
    let vs = split_heads(value, heads)
    let probs = Tensor([heads * seq, kv_seq])
    raw_matmul_batched(qs, ks, probs, heads, seq, hd, kv_seq, seq * hd, kv_seq * hd, trans_b: true)
    cached_attn_kernel(softmax_entry("mha_softmax_scaled")).dispatch_async(
        [probs.buffer()],
        heads * seq * 256,
        [heads * seq, kv_seq, f32_bits(factor), causal_rows, 0],
        group_size: 256
    )
    let context = Tensor([heads * seq, hd])
    raw_matmul_batched(probs, vs, context, heads, seq, kv_seq, hd, seq * kv_seq, kv_seq * hd)
    if qs.requires_grad or ks.requires_grad or vs.requires_grad {
        #: The backward reads all three split copies. One made from an input
        #: with no gradient has no creator, so `backward()` would never hand
        #: it back; it goes with `probs` instead.
        var kept = [probs]
        for split in [qs, ks, vs] { if split.creator is null { kept.push(split) } }
        context.attach_creator(GradNode([qs, ks, vs], fn(upstream: Tensor) -> list[Tensor] {
            return mha_grads(upstream, qs, ks, vs, probs, heads, seq, kv_seq, hd, factor, causal)
        }, extras: kept))
        return merge_heads(context, heads, seq)
    }
    let output = merge_heads(context, heads, seq)
    qs.free()
    ks.free()
    vs.free()
    probs.free()
    context.free()
    return output
}

#: Gradients of `mha_by_ops` with respect to the split-head Q, K and V.
#:
#: The four products per head are the ones `matmul`'s own backward would have
#: chosen for the same two products, in the same operand order, so the
#: arithmetic is identical; what differs is that each writes straight into its
#: slice of one gradient buffer. `grad_probs` carries d(probs) into the
#: softmax backward and comes back holding d(scores), which saves a second
#: sixty-seven-megabyte buffer at seq 1024.
fn mha_grads(
    upstream: Tensor,
    qs: Tensor,
    ks: Tensor,
    vs: Tensor,
    probs: Tensor,
    heads: int,
    seq: int,
    kv_seq: int,
    hd: int,
    factor: float,
    causal: bool
) -> list[Tensor] {
    let causal_rows = if causal { seq } else { 0 }
    let grad_q = Tensor([heads * seq, hd])
    let grad_k = Tensor([heads * kv_seq, hd])
    let grad_v = Tensor([heads * kv_seq, hd])
    let grad_probs = Tensor([heads * seq, kv_seq])
    raw_matmul_batched(
        upstream,
        vs,
        grad_probs,
        heads,
        seq,
        hd,
        kv_seq,
        seq * hd,
        kv_seq * hd,
        trans_b: true
    )
    raw_matmul_batched(
        probs,
        upstream,
        grad_v,
        heads,
        kv_seq,
        seq,
        hd,
        seq * kv_seq,
        seq * hd,
        trans_a: true
    )
    cached_attn_kernel(softmax_entry("mha_softmax_backward_scaled")).dispatch_async(
        [probs.buffer(), grad_probs.buffer()],
        heads * seq * 256,
        [heads * seq, kv_seq, f32_bits(factor), causal_rows, 0],
        group_size: 256
    )
    raw_matmul_batched(grad_probs, ks, grad_q, heads, seq, kv_seq, hd, seq * kv_seq, kv_seq * hd)
    raw_matmul_batched(
        grad_probs,
        qs,
        grad_k,
        heads,
        kv_seq,
        seq,
        hd,
        seq * kv_seq,
        seq * hd,
        trans_a: true
    )
    grad_probs.free()
    return [grad_q, grad_k, grad_v]
}

fn mha_by_gemm(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    heads: int,
    seq: int,
    hd: int,
    factor: float
) -> Tensor {
    let qs = split_heads(query, heads)
    let ks = split_heads(key, heads)
    let vs = split_heads(value, heads)
    var parts: list[Tensor] = []
    for h in 0..heads {
        let qh = qs.view(h * seq * hd, [seq, hd])
        let kh = ks.view(h * seq * hd, [seq, hd])
        let vh = vs.view(h * seq * hd, [seq, hd])
        let scores = raw_matmul(qh, kh, trans_b: true)
        let scaled = scale(scores, factor)
        scores.free()
        let probs = softmax(scaled)
        scaled.free()
        parts.push(raw_matmul(probs, vh))
        qh.free()
        kh.free()
        vh.free()
        probs.free()
    }
    qs.free()
    ks.free()
    vs.free()
    let packed = cat(parts)
    for part in parts { part.free() }
    let output = merge_heads(packed, heads, seq)
    packed.free()
    return output
}
