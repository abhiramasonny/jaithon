#: Attention, fused and flash, against the same arithmetic written one head at a time.
import jaitensor as jt
import std.gpu as gpu
import std.math
from jaitensor.layers import MultiHeadAttention
from jaitensor.tensor import Tensor
from std.io import Path
from std.random import Random
from std.test import assert_eq, assert_near, assert_throws, assert_true

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

#: Deterministic values in [-1, 1] that a host reference can rebuild exactly.
fn wave(count: int, phase: float) -> list[float] {
    var out: list[float] = []
    for index in 0..count { out.push(math.sin(float(index) * 0.731 + phase)) }
    return out
}

#: Attention computed on the host in float64, for `query` `[tq, heads * hd]`
#: against `key`/`value` `[tk, heads * hd]`. With `causal`, query row `i`
#: sees key rows up to `i + tk - tq`, which is the alignment a KV cache needs.
fn host_attention(
    query: list[float],
    key: list[float],
    value: list[float],
    tq: int,
    tk: int,
    heads: int,
    hd: int,
    causal: bool
) -> list[float] {
    let dim = heads * hd
    let factor = 1.0 / math.sqrt(float(hd))
    var out: list[float] = [0.0 for _ in 0..tq * dim]
    for h in 0..heads {
        for i in 0..tq {
            let limit = if causal { i + tk - tq + 1 } else { tk }
            var scores: list[float] = []
            var peak = -1.0e300
            for j in 0..limit {
                var dot = 0.0
                for d in 0..hd { dot += query[i * dim + h * hd + d] * key[j * dim + h * hd + d] }
                let s = dot * factor
                scores.push(s)
                if s > peak { peak = s }
            }
            var total = 0.0
            for j in 0..limit {
                scores[j] = math.exp(scores[j] - peak)
                total += scores[j]
            }
            for d in 0..hd {
                var acc = 0.0
                for j in 0..limit { acc += scores[j] * value[j * dim + h * hd + d] }
                out[i * dim + h * hd + d] = acc / total
            }
        }
    }
    return out
}

fn host_matmul(a: list[float], b: list[float], rows: int, inner: int, columns: int) -> list[float] {
    var out: list[float] = [0.0 for _ in 0..rows * columns]
    for r in 0..rows {
        for c in 0..columns {
            var acc = 0.0
            for i in 0..inner { acc += a[r * inner + i] * b[i * columns + c] }
            out[r * columns + c] = acc
        }
    }
    return out
}

fn free_all(tensors: list[Tensor]) -> void {
    for tensor in tensors { if not tensor.is_freed() { tensor.free() } }
}

#: The layer's forward value with gradients recorded, against the host.
#:
#: With `requires_grad` the layer takes the differentiable route, which shares
#: no code with the fused inference kernel; this is the value training sees.
fn test_attention_layer_forward_value_with_gradients_recorded() -> void {
    if not jt.is_available() { return }
    let seq = 6
    let dim = 8
    let heads = 2
    let layer = jt.MultiHeadAttention(dim, heads: heads)
    defer { layer.free() }
    layer.build(dim, Random.with_seed(3))
    let params = layer.parameters()
    let weights = [
        params[0].download(),
        params[1].download(),
        params[2].download(),
        params[3].download(),
    ]
    let xs = wave(seq * dim, 0.3)
    let x = Tensor([seq, dim], xs, requires_grad: true)
    defer { x.free() }
    let y = layer.forward(x)
    let got = y.download()
    free_all(y.discard_graph())
    y.free()
    let q = host_matmul(xs, weights[0], seq, dim, dim)
    let k = host_matmul(xs, weights[1], seq, dim, dim)
    let v = host_matmul(xs, weights[2], seq, dim, dim)
    let context = host_attention(q, k, v, seq, seq, heads, dim // heads, false)
    let expected = host_matmul(context, weights[3], seq, dim, dim)
    assert_values_near(got, expected, 1e-4)
}

#: `sum(layer(x) * w)` for the finite-difference checks below, freeing the
#: graph it built.
fn layer_objective(layer: MultiHeadAttention, x: Tensor, w: Tensor) -> float {
    let y = layer.forward(x)
    let product = jt.multiply(y, w)
    let total = jt.sum(product)
    let value = total.download()[0]
    free_all(total.discard_graph())
    total.free()
    return value
}

#: The layer's gradients for its query weights and its input, against a
#: central difference. This is the backward the layer could not run at all
#: while its forward freed the projections the graph reads.
fn test_attention_layer_gradients_match_finite_differences() -> void {
    if not jt.is_available() { return }
    let seq = 5
    let dim = 8
    let heads = 2
    let layer = jt.MultiHeadAttention(dim, heads: heads)
    defer { layer.free() }
    layer.build(dim, Random.with_seed(4))
    let wq = layer.parameters()[0]
    let wv = layer.parameters()[2]
    let xs = wave(seq * dim, 1.1)
    let w = Tensor([seq, dim], wave(seq * dim, 2.0))
    defer { w.free() }
    let x = Tensor([seq, dim], xs, requires_grad: true)
    defer { x.free() }
    let y = layer.forward(x)
    let product = jt.multiply(y, w)
    let total = jt.sum(product)
    let held = total.backward()
    let gx = x.grad
    let gq = wq.grad
    let gv = wv.grad
    assert_true(gx is not null and gq is not null and gv is not null)
    if gx is null or gq is null or gv is null { return }
    let grad_x = gx.download()
    let grad_q = gq.download()
    let grad_v = gv.download()
    free_all(held)
    total.free()
    let eps = 1e-3
    for (target, analytic) in [(wq, grad_q), (wv, grad_v)] {
        let base = target.download()
        for index in 0..base.len() {
            var plus = [value for value in base]
            plus[index] += eps
            target.upload(plus)
            let up = layer_objective(layer, x, w)
            var minus = [value for value in base]
            minus[index] -= eps
            target.upload(minus)
            let down = layer_objective(layer, x, w)
            target.upload(base)
            let expected = (up - down) / (2.0 * eps)
            assert_near(analytic[index], expected, max(1e-2, math.fabs(expected) * 5e-3))
        }
    }
    for index in 0..xs.len() {
        var plus = [value for value in xs]
        plus[index] += eps
        let xp = Tensor([seq, dim], plus)
        let up = layer_objective(layer, xp, w)
        xp.free()
        var minus = [value for value in xs]
        minus[index] -= eps
        let xm = Tensor([seq, dim], minus)
        let down = layer_objective(layer, xm, w)
        xm.free()
        let expected = (up - down) / (2.0 * eps)
        assert_near(grad_x[index], expected, max(1e-2, math.fabs(expected) * 5e-3))
    }
}

#: Three SGD steps on a hand-written model: the loss falls and nothing leaks.
fn test_attention_layer_trains_in_a_hand_written_model() -> void {
    if not jt.is_available() { return }
    let seq = 64
    let dim = 128
    let layer = jt.MultiHeadAttention(dim, heads: 2)
    defer { layer.free() }
    layer.build(dim, Random.with_seed(1))
    let x = Tensor([seq, dim], wave(seq * dim, 0.5))
    let target = Tensor([seq, dim], wave(seq * dim, 1.7))
    defer { x.free() }
    defer { target.free() }
    let optimizer = jt.SGD(learning_rate: 0.05)
    var losses: list[float] = []
    var live_after_first = 0
    for step in 0..6 {
        jt.zero_grad(layer.parameters())
        let y = layer.forward(x)
        let loss = jt.mse(y, target)
        losses.push(loss.download()[0])
        free_all(loss.backward())
        loss.free()
        optimizer.step(layer.parameters())
        gpu.synchronize()
        if step == 1 { live_after_first = gpu.live_buffers() }
        if step > 1 { assert_eq(gpu.live_buffers(), live_after_first) }
    }
    for value in losses { assert_true(math.is_finite(value)) }
    assert_true(losses[5] < losses[0])
}

#: The layer inside `Sequential`, where `fit` frees what `backward()` returns.
fn test_attention_layer_trains_inside_sequential() -> void {
    if not jt.is_available() { return }
    let samples = 32
    let width = 16
    let classes = 4
    var features: list[float] = []
    var labels: list[float] = []
    for sample in 0..samples {
        let label = sample % classes
        for column in 0..width {
            features.push(if column % classes == label {
                1.0
            } else {
                0.1 * math.sin(float(sample + column))
            })
        }
        labels.push(float(label))
    }
    let data = jt.Dataset(Tensor([samples, width], features), Tensor([samples], labels))
    defer { data.free() }
    let model = jt.Sequential(
        [
            jt.Dense(width),
            jt.MultiHeadAttention(width, heads: 2),
            jt.Dense(classes, activation: jt.Activation.Softmax),
        ]
    )
    defer { model.free() }
    model.compile(optimizer: jt.Adam(learning_rate: 0.01), loss: jt.SparseCategoricalCrossentropy())
    let history = model.fit(data, epochs: 3, batch_size: samples, verbose: false, plot_path: null)
    assert_eq(history.loss.len(), 3)
    for value in history.loss { assert_true(math.is_finite(value)) }
    assert_true(history.loss[2] < history.loss[0])
}

#: One recorded torch case: `[seq, heads * hd]` causal attention.
class TorchCase {
    pub let seq: int
    pub let hd: int
    pub let heads: int
    pub let loss: float
    pub let rows: list[list[float]]

    pub fn init(self, seq: int, hd: int, heads: int, loss: float, rows: list[list[float]]) {
        self.seq = seq
        self.hd = hd
        self.heads = heads
        self.loss = loss
        self.rows = rows
    }
}

fn torch_cases() -> list[TorchCase] {
    let lines = Path("packages/jaitensor/tests/oracle/causal_sdpa.txt").read_text().split("\n")
    var cases: list[TorchCase] = []
    var index = 0
    while index + 4 < lines.len() {
        let header = lines[index].split(" ")
        if header.len() != 5 or header[0] != "case" {
            index += 1
            continue
        }
        var rows: list[list[float]] = []
        for offset in 1..5 {
            rows.push([float(text) for text in lines[index + offset].split(" ")])
        }
        cases.push(
            TorchCase(int(header[1]), int(header[2]), int(header[3]), float(header[4]), rows)
        )
        index += 5
    }
    return cases
}

fn sample_positions(size: int) -> list[int] {
    return [(j * 7919 + 13) % size for j in 0..64]
}

fn assert_samples_near(
    values: list[float],
    expected: list[float],
    positions: list[int],
    absolute: float,
    relative: float
) -> void {
    for (j, position) in positions.enumerate() {
        let want = expected[j]
        assert_near(values[position], want, absolute + relative * math.fabs(want))
    }
}

#: Causal attention against PyTorch's `scaled_dot_product_attention` with
#: `is_causal=True`, run in float64 by tests/oracle/causal_sdpa.py.
#:
#: Each case runs twice: with gradients, which takes the MMA kernels at
#: sequences that fill 64-row patches with hd 64 and the op composition
#: otherwise, and without, which takes the flash prefill kernel (or the decode
#: kernel at one row). The lengths straddle the 64-row patch and the 32-key
#: tile on purpose: 63, 64 and 65 put the diagonal at every edge.
fn test_causal_attention_matches_torch() -> void {
    if not jt.is_available() { return }
    let cases = torch_cases()
    assert_eq(cases.len(), 12)
    for recorded in cases {
        let seq = recorded.seq
        let dim = recorded.heads * recorded.hd
        let positions = sample_positions(seq * dim)
        let w = Tensor([seq, dim], wave(seq * dim, 2.5))
        let q = Tensor([seq, dim], wave(seq * dim, 0.1), requires_grad: true)
        let k = Tensor([seq, dim], wave(seq * dim, 0.9), requires_grad: true)
        let v = Tensor([seq, dim], wave(seq * dim, 1.7), requires_grad: true)
        let y = jt.multi_head_attention(q, k, v, recorded.heads, causal: true)
        let product = jt.multiply(y, w)
        let total = jt.sum(product)
        let held = total.backward()
        assert_near(total.download()[0], recorded.loss, 1e-3 + 1e-4 * math.fabs(recorded.loss))
        assert_samples_near(y.download(), recorded.rows[0], positions, 1e-4, 1e-4)
        for (slot, tensor) in [q, k, v].enumerate() {
            let gradient = tensor.grad
            assert_true(gradient is not null)
            if gradient is not null {
                assert_samples_near(
                    gradient.download(),
                    recorded.rows[slot + 1],
                    positions,
                    2e-4,
                    1e-3
                )
            }
        }
        free_all(held)
        total.free()
        let qi = Tensor([seq, dim], wave(seq * dim, 0.1))
        let ki = Tensor([seq, dim], wave(seq * dim, 0.9))
        let vi = Tensor([seq, dim], wave(seq * dim, 1.7))
        let fast = jt.multi_head_attention(qi, ki, vi, recorded.heads, causal: true)
        #: The prefill kernel holds Q, K and V in half precision.
        assert_samples_near(fast.download(), recorded.rows[0], positions, 4e-3, 4e-3)
        free_all([w, q, k, v, qi, ki, vi, fast])
    }
}

#: Every inference route against the host, including the ones the torch cases
#: do not reach: fewer queries than keys (a continued sequence), the decode
#: kernel at one to seven rows, the MPSGraph fallback at a head width neither
#: device kernel takes, and one head.
fn check_route(tq: int, tk: int, heads: int, hd: int, causal: bool, tolerance: float) -> void {
    let dim = heads * hd
    let qs = wave(tq * dim, 0.2)
    let ks = wave(tk * dim, 1.3)
    let vs = wave(tk * dim, 2.4)
    let q = Tensor([tq, dim], qs)
    let k = Tensor([tk, dim], ks)
    let v = Tensor([tk, dim], vs)
    let y = jt.multi_head_attention(q, k, v, heads, causal: causal)
    let expected = host_attention(qs, ks, vs, tq, tk, heads, hd, causal)
    assert_values_near(y.download(), expected, tolerance)
    let qg = Tensor([tq, dim], qs, requires_grad: true)
    let yg = jt.multi_head_attention(qg, k, v, heads, causal: causal)
    assert_values_near(yg.download(), expected, 1e-4)
    free_all(yg.discard_graph())
    free_all([q, k, v, y, qg, yg])
}

fn test_attention_routes_match_the_host() -> void {
    if not jt.is_available() { return }
    for hd in [32, 64] {
        for tq in [1, 2, 7] {
            check_route(tq, 300, 3, hd, true, 1e-4)
            check_route(tq, 300, 3, hd, false, 1e-4)
        }
        check_route(65, 130, 2, hd, true, 4e-3)
        check_route(65, 130, 2, hd, false, 4e-3)
        check_route(100, 100, 2, hd, true, 4e-3)
    }
    check_route(40, 40, 2, 16, true, 2e-3)
    check_route(3, 50, 2, 16, true, 2e-3)
    check_route(70, 70, 1, 32, true, 4e-3)
    #: The decode kernel takes any head width that is a multiple of 32 up to
    #: 128; the prefill kernel only 32 and 64, so these longer queries go to
    #: MPSGraph.
    check_route(1, 200, 2, 128, true, 1e-4)
    check_route(3, 200, 2, 96, false, 1e-4)
    check_route(20, 40, 2, 128, true, 2e-3)
}

#: `[batch, t, dim]` folds the batch into the head axis: each sequence must
#: come out exactly as if it had been attended on its own, both routes.
fn test_batched_attention_matches_each_sequence_alone() -> void {
    if not jt.is_available() { return }
    let batch = 3
    let seq = 20
    let heads = 2
    let hd = 32
    let dim = heads * hd
    let qs = wave(batch * seq * dim, 0.4)
    let ks = wave(batch * seq * dim, 1.4)
    let vs = wave(batch * seq * dim, 2.2)
    for grad in [false, true] {
        let q = Tensor([batch, seq, dim], qs, requires_grad: grad)
        let k = Tensor([batch, seq, dim], ks)
        let v = Tensor([batch, seq, dim], vs)
        let y = jt.multi_head_attention(q, k, v, heads, causal: true)
        assert_eq(y.shape, [batch, seq, dim])
        let got = y.download()
        for b in 0..batch {
            let lo = b * seq * dim
            let hi = lo + seq * dim
            let expected = host_attention(
                qs[lo:hi],
                ks[lo:hi],
                vs[lo:hi],
                seq,
                seq,
                heads,
                hd,
                true
            )
            assert_values_near(got[lo:hi], expected, if grad { 1e-4 } else { 4e-3 })
        }
        if grad {
            let total = jt.sum(y)
            free_all(total.backward())
            total.free()
            assert_true(q.grad is not null)
        }
        free_all([q, k, v, y])
    }
}

#: A cached layer fed a prompt and then one row at a time gives, for every new
#: row, what the uncached causal layer gives for that row of the whole prefix,
#: and allocates nothing that outlives a step.
fn check_cache_matches_whole_prefix(dim: int, heads: int, prompt: int, total: int) -> void {
    let layer = jt.MultiHeadAttention(dim, heads: heads, causal: true)
    defer { layer.free() }
    layer.build(dim, Random.with_seed(7))
    for parameter in layer.parameters() { parameter.requires_grad = false }
    let xs = wave(total * dim, 0.6)
    let whole = Tensor([total, dim], xs)
    defer { whole.free() }
    let reference = layer.forward(whole)
    let expected = reference.download()
    reference.free()
    layer.start_cache(total)
    var got: list[float] = []
    var live = -1
    var start = 0
    while start < total {
        let count = if start == 0 { prompt } else { 1 }
        let rows = whole.view(start * dim, [count, dim])
        let y = layer.forward(rows)
        for value in y.download() { got.push(value) }
        y.free()
        rows.free()
        start += count
        gpu.synchronize()
        if start > prompt + 1 {
            if live < 0 { live = gpu.live_buffers() }
            assert_eq(gpu.live_buffers(), live)
        }
    }
    assert_eq(layer.cache_length(), total)
    #: The whole-prefix forward takes the half-precision prefill kernel; the
    #: decode kernel reads float32.
    assert_values_near(got, expected, 4e-3)
    layer.reset_cache()
    assert_eq(layer.cache_length(), 0)
    let again = whole.view(0, [prompt, dim])
    let first = layer.forward(again)
    assert_values_near(first.download(), expected[0:prompt * dim], 4e-3)
    first.free()
    again.free()
    layer.free_cache()
}

fn test_attention_layer_cache_matches_the_whole_prefix() -> void {
    if not jt.is_available() { return }
    check_cache_matches_whole_prefix(128, 2, 5, 40)
    check_cache_matches_whole_prefix(384, 6, 17, 150)
    check_cache_matches_whole_prefix(8, 2, 3, 12)
}

#: A causal layer trains: the mask is part of the differentiable path too.
fn test_causal_attention_layer_trains() -> void {
    if not jt.is_available() { return }
    let seq = 128
    let dim = 128
    let layer = jt.MultiHeadAttention(dim, heads: 2, causal: true)
    defer { layer.free() }
    layer.build(dim, Random.with_seed(2))
    let x = Tensor([seq, dim], wave(seq * dim, 0.25))
    let target = Tensor([seq, dim], wave(seq * dim, 1.25))
    defer { x.free() }
    defer { target.free() }
    let optimizer = jt.SGD(learning_rate: 0.05)
    var losses: list[float] = []
    for _step in 0..4 {
        jt.zero_grad(layer.parameters())
        let y = layer.forward(x)
        let loss = jt.mse(y, target)
        losses.push(loss.download()[0])
        free_all(loss.backward())
        loss.free()
        optimizer.step(layer.parameters())
    }
    for value in losses { assert_true(math.is_finite(value)) }
    assert_true(losses[3] < losses[0])
}

#: Gradients of `sum(attention(q, k, v) * w)` for whichever of Q, K and V
#: train, as host lists in Q, K, V order; an input that does not train gives
#: an empty list. Runs `steps` times and checks the last runs allocate
#: nothing that outlives a step.
fn batched_grads(
    shape: list[int],
    heads: int,
    causal: bool,
    trains: list[bool],
    steps: int
) -> list[list[float]] {
    let count = shape[0] * shape[1] * shape[2]
    let inputs = [
        Tensor(shape, wave(count, 0.3), requires_grad: trains[0]),
        Tensor(shape, wave(count, 1.1), requires_grad: trains[1]),
        Tensor(shape, wave(count, 2.3), requires_grad: trains[2]),
    ]
    let w = Tensor(shape, wave(count, 0.9))
    var grads: list[list[float]] = [[], [], []]
    var live = -1
    for step in 0..steps {
        let y = jt.multi_head_attention(inputs[0], inputs[1], inputs[2], heads, causal: causal)
        let total = jt.sum(jt.multiply(y, w))
        let held = total.backward()
        for (index, input) in inputs.enumerate() {
            let g = input.grad
            if trains[index] {
                assert_true(g is not null)
                if g is not null { grads[index] = g.download() }
            }
            if g is not null { g.free() }
            input.grad = null
        }
        free_all(held)
        free_all([total, y])
        gpu.synchronize()
        if step == 1 { live = gpu.live_buffers() }
        if step > 1 { assert_eq(gpu.live_buffers(), live) }
    }
    free_all(inputs)
    w.free()
    return grads
}

#: Rank-three attention where only some of Q, K and V train. The folded copy
#: of a plain input has no creator, but the backward still reads it: it used
#: to be freed (the MMA route threw on backward) or never returned (the ops
#: route leaked four buffers a step). Each gradient must equal the one the
#: same input gets when all three train.
fn test_batched_attention_trains_with_some_inputs_plain() -> void {
    if not jt.is_available() { return }
    #: MMA route causal, MMA route plain, and the ops route.
    let cases = [([2, 64, 64], 1, true), ([2, 64, 128], 2, false), ([2, 20, 64], 2, true)]
    for (shape, heads, causal) in cases {
        let all = batched_grads(shape, heads, causal, [true, true, true], 1)
        for trains in [[true, false, false], [false, true, true], [false, false, true]] {
            let some = batched_grads(shape, heads, causal, trains, 4)
            for index in 0..3 {
                if trains[index] { assert_values_near(some[index], all[index], 1e-5) }
            }
        }
    }
}

#: Rank-two attention through the ops route with only Q training: the split
#: copies of K and V are returned by `backward()` instead of leaking.
fn test_attention_ops_route_returns_plain_split_copies() -> void {
    if not jt.is_available() { return }
    let all = batched_grads([1, 20, 64], 2, false, [true, true, true], 1)
    let some = batched_grads([1, 20, 64], 2, false, [true, false, false], 4)
    assert_values_near(some[0], all[0], 1e-5)
}

#: Gradients of the layer's parameters with some of them frozen, as host
#: lists (empty for a frozen one), over `steps` steps that must not grow the
#: live buffer count.
fn layer_grads_with_frozen(frozen: list[int], causal: bool, steps: int) -> list[list[float]] {
    let seq = 64
    let dim = 64
    let layer = jt.MultiHeadAttention(dim, heads: 1, causal: causal)
    defer { layer.free() }
    layer.build(dim, Random.with_seed(3))
    let params = layer.parameters()
    for index in frozen { params[index].requires_grad = false }
    let x = Tensor([seq, dim], wave(seq * dim, 0.4))
    let w = Tensor([seq, dim], wave(seq * dim, 1.9))
    var grads: list[list[float]] = [[], [], [], []]
    var live = -1
    for step in 0..steps {
        jt.zero_grad(params)
        let y = layer.forward(x)
        let total = jt.sum(jt.multiply(y, w))
        let held = total.backward()
        #: A frozen parameter that feeds a recorded product may still be
        #: handed a gradient (matmul's backward fills both operands), so
        #: only the trained ones are checked; `zero_grad` frees the rest.
        for (index, param) in params.enumerate() {
            if index in frozen { continue }
            let g = param.grad
            assert_true(g is not null)
            if g is not null { grads[index] = g.download() }
        }
        free_all(held)
        free_all([total, y])
        gpu.synchronize()
        if step == 1 { live = gpu.live_buffers() }
        if step > 1 { assert_eq(gpu.live_buffers(), live) }
    }
    jt.zero_grad(params)
    free_all([x, w])
    return grads
}

#: The layer with some projections frozen and an input that does not train.
#: A frozen projection records nothing, but attention's backward reads it as
#: soon as another projection trains, and the output product's backward reads
#: the context when only `wo` trains; freeing either made backward throw.
#: Freezing one parameter must not change any other's gradient.
fn test_attention_layer_trains_with_frozen_projections() -> void {
    if not jt.is_available() { return }
    for causal in [false, true] {
        let all = layer_grads_with_frozen([], causal, 1)
        for frozen in [[0], [1], [2], [0, 1, 2], [3], [0, 2]] {
            let some = layer_grads_with_frozen(frozen, causal, 4)
            #: With all three projections frozen attention records nothing,
            #: so it takes the half-precision inference kernel, and `wo`'s
            #: gradient reads that context.
            let tolerance = if frozen.len() == 3 { 1e-2 } else { 1e-4 }
            for index in 0..4 {
                if not index in frozen { assert_values_near(some[index], all[index], tolerance) }
            }
        }
    }
}

#: Rows `lo..hi` of every sequence in a `[batch, total, dim]` host block, as
#: a `[batch, hi - lo, dim]` block.
fn sequence_rows(values: list[float], batch: int, total: int, dim: int, lo: int, hi: int) -> list[float] {
    var out: list[float] = []
    for b in 0..batch {
        let base = b * total * dim
        for value in values[base + lo * dim:base + hi * dim] { out.push(value) }
    }
    return out
}

#: The layer's output and gradients for `sum(layer(x) * w)`, in the order
#: output, dx, then each parameter's gradient. `x` is fed as one
#: `[batch, seq, dim]` block when `batched`, and otherwise one `[seq, dim]`
#: sequence at a time with the parameter gradients summed across them.
fn layer_batch_pass(
    layer: MultiHeadAttention,
    batch: int,
    seq: int,
    xs: list[float],
    ws: list[float],
    batched: bool,
    trains: bool
) -> list[list[float]] {
    let dim = layer.dim
    let params = layer.parameters()
    jt.zero_grad(params)
    var out: list[list[float]] = [[], []]
    let pieces = if batched { [[batch, seq, dim]] } else { [[seq, dim] for _ in 0..batch] }
    for (index, shape) in pieces.enumerate() {
        let lo = index * seq * dim
        let hi = if batched { batch * seq * dim } else { lo + seq * dim }
        let x = Tensor(shape, xs[lo:hi], requires_grad: trains)
        let w = Tensor(shape, ws[lo:hi])
        let y = layer.forward(x)
        assert_eq(y.shape, shape)
        for value in y.download() { out[0].push(value) }
        if trains {
            let total = jt.sum(jt.multiply(y, w))
            free_all(total.backward())
            total.free()
            let g = x.grad
            assert_true(g is not null)
            if g is not null { for value in g.download() { out[1].push(value) } }
        }
        free_all([x, w, y])
    }
    if trains {
        for param in params {
            let g = param.grad
            assert_true(g is not null)
            if g is not null { out.push(g.download()) }
        }
    }
    jt.zero_grad(params)
    return out
}

#: A batch through the layer must equal each sequence through it alone, with
#: gradients recorded and without, and leak nothing a step.
fn check_layer_takes_a_batch(batch: int, seq: int, dim: int, heads: int, causal: bool) -> void {
    let layer = jt.MultiHeadAttention(dim, heads: heads, causal: causal)
    defer { layer.free() }
    layer.build(dim, Random.with_seed(4))
    let xs = wave(batch * seq * dim, 0.7)
    let ws = wave(batch * seq * dim, 1.9)
    let together = layer_batch_pass(layer, batch, seq, xs, ws, true, true)
    let alone = layer_batch_pass(layer, batch, seq, xs, ws, false, true)
    assert_eq(together.len(), 6)
    for (index, values) in together.enumerate() {
        assert_values_near(values, alone[index], 1e-4)
    }
    #: Without a graph the batch takes the inference kernels, which hold Q, K
    #: and V in half precision, and must hand back everything it allocates.
    for param in layer.parameters() { param.requires_grad = false }
    let quick = layer_batch_pass(layer, batch, seq, xs, ws, true, false)
    assert_values_near(quick[0], alone[0], 4e-3)
    var live = -1
    for step in 0..3 {
        let x = Tensor([batch, seq, dim], xs)
        let y = layer.forward(x)
        free_all([x, y])
        gpu.synchronize()
        if step == 1 { live = gpu.live_buffers() }
        if step > 1 { assert_eq(gpu.live_buffers(), live) }
    }
    for param in layer.parameters() { param.requires_grad = true }
}

#: The layer used to refuse anything but `[seq, dim]`, so a model training on
#: batches had to pull the projections out of `parameters()` and write the
#: attention block itself. The cases take the MMA route (sequences of 64 at a
#: head width of 64), the op route, and a plain mask.
fn test_attention_layer_takes_a_batch_of_sequences() -> void {
    if not jt.is_available() { return }
    check_layer_takes_a_batch(3, 64, 128, 2, true)
    check_layer_takes_a_batch(2, 20, 64, 2, true)
    check_layer_takes_a_batch(2, 64, 128, 2, false)
    check_layer_takes_a_batch(1, 64, 64, 1, true)
}

#: A batch of sequences decoded through the cache, a prompt and then one row
#: at a time, equals the uncached batch for every sequence and position.
fn check_batched_cache(batch: int, dim: int, heads: int, prompt: int, total: int) -> void {
    let layer = jt.MultiHeadAttention(dim, heads: heads, causal: true)
    defer { layer.free() }
    layer.build(dim, Random.with_seed(11))
    for parameter in layer.parameters() { parameter.requires_grad = false }
    let xs = wave(batch * total * dim, 0.6)
    let whole = Tensor([batch, total, dim], xs)
    defer { whole.free() }
    let reference = layer.forward(whole)
    let expected = reference.download()
    reference.free()
    layer.start_cache(total, batch: batch)
    assert_eq(layer.cache_batch_size(), batch)
    var got: list[list[float]] = [[] for _ in 0..batch]
    var live = -1
    var start = 0
    while start < total {
        let count = if start == 0 { prompt } else { 1 }
        let rows = Tensor([batch, count, dim], sequence_rows(xs, batch, total, dim, start, start + count))
        let y = layer.forward(rows)
        assert_eq(y.shape, [batch, count, dim])
        let values = y.download()
        for b in 0..batch {
            for value in values[b * count * dim:(b + 1) * count * dim] { got[b].push(value) }
        }
        free_all([y, rows])
        start += count
        gpu.synchronize()
        if start > prompt + 1 {
            if live < 0 { live = gpu.live_buffers() }
            assert_eq(gpu.live_buffers(), live)
        }
    }
    assert_eq(layer.cache_length(), total)
    for b in 0..batch {
        assert_values_near(got[b], expected[b * total * dim:(b + 1) * total * dim], 4e-3)
    }
    #: A rank-two input is one sequence, which a cache of several is not.
    let single = Tensor([1, dim], xs[0:dim])
    defer { single.free() }
    layer.reset_cache()
    if batch > 1 { assert_throws(ValueError, || layer.forward(single)) }
    layer.free_cache()
    assert_eq(layer.cache_batch_size(), 1)
}

fn test_attention_layer_cache_takes_a_batch() -> void {
    if not jt.is_available() { return }
    check_batched_cache(3, 128, 2, 5, 24)
    check_batched_cache(2, 256, 4, 16, 40)
    check_batched_cache(1, 64, 1, 4, 12)
}

#: A leaf that went through `reshape` keeps a readable gradient once the
#: graph is freed. `reshape`'s backward handed the leaf a view of the reshaped
#: tensor's own gradient, which `backward()` returns for the caller to free,
#: so the leaf's `.grad` pointed at freed memory and reading it killed the
#: process. Batched attention reshapes every input it folds, so any leaf fed
#: to it as `[batch, seq, dim]` -- a layer's input, or Q itself -- hit this.
fn test_a_reshaped_leaf_keeps_its_gradient_after_the_graph_is_freed() -> void {
    if not jt.is_available() { return }
    let ws = wave(12, 0.5)
    let x = Tensor([3, 4], wave(12, 0.1), requires_grad: true)
    let w = Tensor([2, 6], ws)
    let total = jt.sum(jt.multiply(jt.reshape(jt.reshape(x, [12]), [2, 6]), w))
    free_all(total.backward())
    total.free()
    gpu.synchronize()
    let g = x.grad
    assert_true(g is not null)
    if g is not null {
        assert_eq(g.shape, [3, 4])
        assert_values_near(g.download(), ws, 1e-6)
    }
    free_all([x, w])
    #: The same through rank-three attention, whose batch fold reshapes Q.
    let batch = 2
    let seq = 8
    let dim = 64
    let q = Tensor([batch, seq, dim], wave(batch * seq * dim, 0.3), requires_grad: true)
    let k = Tensor([batch, seq, dim], wave(batch * seq * dim, 1.3))
    let v = Tensor([batch, seq, dim], wave(batch * seq * dim, 2.3))
    let y = jt.multi_head_attention(q, k, v, 2, causal: true)
    let loss = jt.sum(y)
    free_all(loss.backward())
    free_all([loss, y])
    gpu.synchronize()
    let gq = q.grad
    assert_true(gq is not null)
    if gq is not null {
        var finite = true
        for value in gq.download() { if not math.is_finite(value) { finite = false } }
        assert_true(finite)
    }
    free_all([q, k, v])
}
