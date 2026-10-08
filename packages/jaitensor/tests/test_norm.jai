#: Layer normalisation's parameter gradients at the sizes a transformer uses.
#:
#: test_gradients.jai checks `layer_norm` against a central difference, but at
#: three rows, where the column sums are one strip and the join never runs.
#: These sizes split the rows into strips, give a width that is not a whole
#: number of 32-column blocks, and are checked against the host in float64.
import jaitensor as jt
import std.math
from jaitensor.tensor import Tensor
from std.test import assert_eq, assert_near, assert_true

fn wave(count: int, phase: float, amplitude: float, offset: float) -> list[float] {
    var out: list[float] = []
    for index in 0..count { out.push(offset + amplitude * math.sin(float(index) * 0.377 + phase)) }
    return out
}

fn check_parameter_gradients(rows: int, width: int) -> void {
    let xs = wave(rows * width, 0.3, 2.0, 0.5)
    let ws = wave(rows * width, 1.9, 1.0, 0.0)
    let gs = wave(width, 0.7, 0.5, 1.0)
    let x = Tensor([rows, width], xs, requires_grad: true)
    let gamma = Tensor([width], gs, requires_grad: true)
    let beta = Tensor([width], wave(width, 2.2, 0.3, 0.0), requires_grad: true)
    let w = Tensor([rows, width], ws)
    let y = jt.layer_norm(x, gamma, beta)
    let product = jt.multiply(y, w)
    let total = jt.sum(product)
    let held = total.backward()
    let gg = gamma.grad
    let gb = beta.grad
    assert_true(gg is not null and gb is not null)
    if gg is null or gb is null { return }
    let grad_gamma = gg.download()
    let grad_beta = gb.download()
    var want_gamma: list[float] = [0.0 for _ in 0..width]
    var want_beta: list[float] = [0.0 for _ in 0..width]
    for r in 0..rows {
        var mean = 0.0
        for c in 0..width { mean += xs[r * width + c] }
        mean /= float(width)
        var spread = 0.0
        for c in 0..width {
            let d = xs[r * width + c] - mean
            spread += d * d
        }
        let inv = 1.0 / math.sqrt(spread / float(width) + 1e-5)
        for c in 0..width {
            let dy = ws[r * width + c]
            want_gamma[c] += dy * (xs[r * width + c] - mean) * inv
            want_beta[c] += dy
        }
    }
    assert_eq(grad_gamma.len(), width)
    for c in 0..width {
        assert_near(grad_gamma[c], want_gamma[c], 1e-3 + 1e-4 * math.fabs(want_gamma[c]))
        assert_near(grad_beta[c], want_beta[c], 1e-3 + 1e-4 * math.fabs(want_beta[c]))
    }
    for tensor in held { tensor.free() }
    total.free()
    x.free()
    gamma.free()
    beta.free()
    w.free()
}

#: `[batch, seq, dim]` is normalized as the `[batch * seq, dim]` it is laid out
#: as, and so is any deeper rank. `layer_norm` took rank two only, so every
#: transformer reshaped its activations around each norm by hand. The value
#: and all three gradients must equal the rank-two answer bit for bit: it is
#: the same kernel over the same rows.
fn test_layer_norm_folds_leading_axes_into_rows() -> void {
    if not jt.is_available() { return }
    let batch = 3
    let seq = 70
    let width = 100
    let count = batch * seq * width
    let xs = wave(count, 0.3, 2.0, 0.5)
    let ws = wave(count, 1.9, 1.0, 0.0)
    let gs = wave(width, 0.7, 0.5, 1.0)
    let bs = wave(width, 2.2, 0.3, 0.0)
    var answers: list[list[list[float]]] = []
    for shape in [[batch * seq, width], [batch, seq, width], [batch, seq, 1, width]] {
        let x = Tensor(shape, xs, requires_grad: true)
        let gamma = Tensor([width], gs, requires_grad: true)
        let beta = Tensor([width], bs, requires_grad: true)
        let w = Tensor(shape, ws)
        let y = jt.layer_norm(x, gamma, beta)
        assert_eq(y.shape, shape)
        var got = [y.download()]
        let total = jt.sum(jt.multiply(y, w))
        let held = total.backward()
        for tensor in [x, gamma, beta] {
            let g = tensor.grad
            assert_true(g is not null)
            if g is not null {
                assert_eq(g.shape, tensor.shape)
                got.push(g.download())
            }
        }
        answers.push(got)
        for tensor in held { tensor.free() }
        total.free()
        for tensor in [x, gamma, beta, w] { tensor.free() }
    }
    for answer in answers[1:] {
        assert_eq(answer.len(), 4)
        for (index, values) in answer.enumerate() { assert_eq(values, answers[0][index]) }
    }
}

fn test_layer_norm_parameter_gradients_at_transformer_sizes() -> void {
    if not jt.is_available() { return }
    check_parameter_gradients(3, 8)
    check_parameter_gradients(130, 100)
    check_parameter_gradients(1024, 768)
    check_parameter_gradients(4096, 64)
}
