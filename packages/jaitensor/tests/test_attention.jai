#: Attention, fused and flash, against the same arithmetic written one head at a time.
import jaitensor as jt
import std.math
from jaitensor.tensor import Tensor
from std.random import Random
from std.test import assert_eq, assert_near, assert_true

fn assert_values_near(actual: list[float], expected: list[float], tolerance: float = 1e-5) -> void {
    assert_eq(actual.len(), expected.len())
    for (index, value) in actual.enumerate() {
        assert_near(value, expected[index], tolerance)
    }
}

fn test_attention_keeps_sequence_shape() -> void {
    if not jt.is_available() { return }
    let q = jt.Tensor([2, 4], [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    let k = jt.Tensor([2, 4], [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    let v = jt.Tensor([2, 4], [2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0])
    let y = jt.scaled_dot_product_attention(q, k, v)
    defer { q.free() }
    defer { k.free() }
    defer { v.free() }
    defer { y.free() }
    assert_eq(y.shape, [2, 4])
    let same = jt.multi_head_attention(q, k, v, 1)
    defer { same.free() }
    assert_values_near(y.download(), same.download(), 1e-5)
}

fn test_multi_head_attention_matches_per_head_sdpa() -> void {
    if not jt.is_available() { return }
    let q = jt.Tensor([2, 4], [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    let k = jt.Tensor([2, 4], [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    let v = jt.Tensor([2, 4], [2.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 5.0])
    defer { q.free() }
    defer { k.free() }
    defer { v.free() }
    let fused = jt.multi_head_attention(q, k, v, 2)
    defer { fused.free() }
    let seq = 2
    let heads = 2
    let hd = 2
    let qs = jt.split_heads(q, heads)
    let ks = jt.split_heads(k, heads)
    let vs = jt.split_heads(v, heads)
    defer { qs.free() }
    defer { ks.free() }
    defer { vs.free() }
    var parts: list[Tensor] = []
    for h in 0..heads {
        let qh = qs.view(h * seq * hd, [seq, hd])
        let kh = ks.view(h * seq * hd, [seq, hd])
        let vh = vs.view(h * seq * hd, [seq, hd])
        parts.push(jt.scaled_dot_product_attention(qh, kh, vh))
        qh.free()
        kh.free()
        vh.free()
    }
    let packed = jt.cat(parts)
    for part in parts { part.free() }
    defer { packed.free() }
    let looped = jt.merge_heads(packed, heads, seq)
    defer { looped.free() }
    assert_values_near(fused.download(), looped.download(), 1e-5)
    let rng = Random.with_seed(1)
    let layer = jt.MultiHeadAttention(4, heads: 2)
    defer { layer.free() }
    layer.build(4, rng)
    let y = layer.forward(q)
    defer { y.free() }
    assert_eq(y.shape, [2, 4])
}

fn test_flash_attention_matches_per_head_sdpa() -> void {
    if not jt.is_available() { return }
    let seq = 16
    let heads = 2
    let hd = 2
    var values: list[float] = []
    for i in 0..seq * heads * hd {
        values.push(float(i * 3 % 11) / 5.0 - 1.0)
    }
    let q = jt.Tensor([seq, heads * hd], values)
    let k = jt.Tensor([seq, heads * hd], values)
    var vvals: list[float] = []
    for i in 0..seq * heads * hd {
        vvals.push(float(i * 5 % 7) / 3.0)
    }
    let v = jt.Tensor([seq, heads * hd], vvals)
    defer { q.free() }
    defer { k.free() }
    defer { v.free() }
    let fused = jt.multi_head_attention(q, k, v, heads)
    defer { fused.free() }
    let qs = jt.split_heads(q, heads)
    let ks = jt.split_heads(k, heads)
    let vs = jt.split_heads(v, heads)
    defer { qs.free() }
    defer { ks.free() }
    defer { vs.free() }
    var parts: list[Tensor] = []
    for h in 0..heads {
        let qh = qs.view(h * seq * hd, [seq, hd])
        let kh = ks.view(h * seq * hd, [seq, hd])
        let vh = vs.view(h * seq * hd, [seq, hd])
        parts.push(jt.scaled_dot_product_attention(qh, kh, vh))
        qh.free()
        kh.free()
        vh.free()
    }
    let packed = jt.cat(parts)
    for part in parts { part.free() }
    defer { packed.free() }
    let looped = jt.merge_heads(packed, heads, seq)
    defer { looped.free() }
    assert_values_near(fused.download(), looped.download(), 2e-4)
}

fn test_wide_flash_attention_matches_per_head_sdpa() -> void {
    if not jt.is_available() { return }
    let seq = 96
    let heads = 2
    let hd = 32
    var values: list[float] = []
    for i in 0..seq * heads * hd {
        values.push(float(i * 3 % 11) / 5.0 - 1.0)
    }
    let q = jt.Tensor([seq, heads * hd], values)
    let k = jt.Tensor([seq, heads * hd], values)
    var vvals: list[float] = []
    for i in 0..seq * heads * hd {
        vvals.push(float(i * 5 % 7) / 3.0)
    }
    let v = jt.Tensor([seq, heads * hd], vvals)
    defer { q.free() }
    defer { k.free() }
    defer { v.free() }
    let fused = jt.multi_head_attention(q, k, v, heads)
    defer { fused.free() }
    let qs = jt.split_heads(q, heads)
    let ks = jt.split_heads(k, heads)
    let vs = jt.split_heads(v, heads)
    defer { qs.free() }
    defer { ks.free() }
    defer { vs.free() }
    var parts: list[Tensor] = []
    for h in 0..heads {
        let qh = qs.view(h * seq * hd, [seq, hd])
        let kh = ks.view(h * seq * hd, [seq, hd])
        let vh = vs.view(h * seq * hd, [seq, hd])
        parts.push(jt.scaled_dot_product_attention(qh, kh, vh))
        qh.free()
        kh.free()
        vh.free()
    }
    let packed = jt.cat(parts)
    for part in parts { part.free() }
    defer { packed.free() }
    let looped = jt.merge_heads(packed, heads, seq)
    defer { looped.free() }
    assert_values_near(fused.download(), looped.download(), 2e-3)
}

#: `multi_head_attention` against the same arithmetic written one head at a
#: time, at sizes the batched GEMM actually tiles.
#:
#: The tests above run at sequence two and head dimension two, where every
#: product is a single tile and split-K never engages, so they say nothing
#: about the batched dispatch that carries the per-head products. These sizes
#: do: 128 divides the tile shape exactly and takes the direct kernel, 100 does
#: not and takes the staged one, and both run enough heads that a mistake in
#: the per-matrix stride shows as one head holding another head's answer rather
#: than as rounding.
#:
#: The temporaries come back rather than being freed here because the caller
#: that asks for gradients needs them alive until backward has read them.
fn attention_by_head(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    heads: int,
    seq: int,
    hd: int
) -> (Tensor, list[Tensor]) {
    let factor = 1.0 / math.sqrt(float(hd))
    let qs = jt.split_heads(query, heads)
    let ks = jt.split_heads(key, heads)
    let vs = jt.split_heads(value, heads)
    var held: list[Tensor] = [qs, ks, vs]
    var parts: list[Tensor] = []
    for h in 0..heads {
        let qh = jt.slice_flat(qs, h * seq * hd, [seq, hd])
        let kh = jt.slice_flat(ks, h * seq * hd, [seq, hd])
        let vh = jt.slice_flat(vs, h * seq * hd, [seq, hd])
        let scores = jt.matmul(qh, kh, trans_b: true)
        let scaled = jt.scale(scores, factor)
        let probs = jt.softmax(scaled)
        parts.push(jt.matmul(probs, vh))
        for tensor in [qh, kh, vh, scores, scaled, probs] { held.push(tensor) }
    }
    let packed = jt.cat(parts)
    for part in parts { held.push(part) }
    held.push(packed)
    return (jt.merge_heads(packed, heads, seq), held)
}

#: Q, K and V carry gradients so that the differentiable composition runs.
#:
#: Without `requires_grad` this routes to the fused inference kernel, which
#: never materialises the scores and shares no code with the path under test:
#: the check then passes with the batched dispatch's per-matrix stride set to
#: zero. The gradient test below cannot stand in for this one either, because
#: the gradients of `sum(attention(q, k, v))` are recomputed from the
#: probabilities and never read the forward context product at all.
fn check_attention_against_per_head(seq: int, dim: int, heads: int) -> void {
    let hd = dim // heads
    let q = Tensor([seq, dim], requires_grad: true)
    let k = Tensor([seq, dim], requires_grad: true)
    let v = Tensor([seq, dim], requires_grad: true)
    defer { q.free() }
    defer { k.free() }
    defer { v.free() }
    q.buffer().fill_uniform(-1.0, 1.0, 11)
    k.buffer().fill_uniform(-1.0, 1.0, 22)
    v.buffer().fill_uniform(-1.0, 1.0, 33)
    let fused = jt.multi_head_attention(q, k, v, heads)
    defer { fused.free() }
    defer { for tensor in fused.discard_graph() { tensor.free() } }
    let (apart, held) = attention_by_head(q, k, v, heads, seq, hd)
    defer { apart.free() }
    defer { for tensor in held { if not tensor.is_freed() { tensor.free() } } }
    assert_eq(fused.shape, [seq, dim])
    assert_values_near(fused.download(), apart.download(), 1e-3)
}

fn test_multi_head_attention_tiled_shapes_match_per_head() -> void {
    if not jt.is_available() { return }
    check_attention_against_per_head(128, 192, 6)
    check_attention_against_per_head(100, 192, 6)
    check_attention_against_per_head(128, 384, 3)
}

#: The same comparison for the gradients, which is where the batched GEMM
#: carries both transposes and the split reduction.
fn test_multi_head_attention_tiled_gradients_match_per_head() -> void {
    if not jt.is_available() { return }
    let seq = 128
    let dim = 192
    let heads = 6
    let hd = dim // heads
    var collected: list[list[list[float]]] = []
    for fused_path in [true, false] {
        let q = Tensor([seq, dim], requires_grad: true)
        let k = Tensor([seq, dim], requires_grad: true)
        let v = Tensor([seq, dim], requires_grad: true)
        q.buffer().fill_uniform(-1.0, 1.0, 11)
        k.buffer().fill_uniform(-1.0, 1.0, 22)
        v.buffer().fill_uniform(-1.0, 1.0, 33)
        var held: list[Tensor] = []
        let y = if fused_path {
            jt.multi_head_attention(q, k, v, heads)
        } else {
            let (built, temporaries) = attention_by_head(q, k, v, heads, seq, hd)
            for tensor in temporaries {
                held.push(tensor)
            }
            built
        }
        let loss = jt.sum(y)
        for tensor in loss.backward() { held.push(tensor) }
        var gradients: list[list[float]] = []
        for parameter in [q, k, v] {
            let gradient = parameter.grad
            assert_true(gradient is not null)
            if gradient is not null { gradients.push(gradient.download()) }
        }
        collected.push(gradients)
        loss.free()
        y.free()
        for tensor in held {
            if not tensor.is_freed() { tensor.free() }
        }
        q.free()
        k.free()
        v.free()
    }
    assert_eq(collected.len(), 2)
    assert_eq(collected[0].len(), 3)
    for (index, values) in collected[0].enumerate() {
        assert_values_near(values, collected[1][index], 1e-3)
    }
}
