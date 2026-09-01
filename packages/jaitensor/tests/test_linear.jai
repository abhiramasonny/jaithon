#: The fused products, each against the two steps it stands in for.
import jaitensor as jt
from jaitensor.tensor import Tensor
from std.test import assert_eq, assert_near, assert_throws, assert_true

#: The fused product-then-GELU gives what the two steps give.
fn test_linear_gelu_matches_the_two_steps() -> void {
    if not jt.is_available() { return }
    let rows = 40
    let inner = 64
    let columns = 96
    var xs: list[float] = []
    for index in 0..rows * inner { xs.push(float(index * 17 % 61 - 30) * 0.05) }
    var ws: list[float] = []
    for index in 0..inner * columns { ws.push(float(index * 23 % 47 - 23) * 0.03) }
    var bs: list[float] = []
    for index in 0..columns { bs.push(float(index % 9) * 0.1 - 0.4) }
    let x = Tensor([rows, inner], xs)
    let w = Tensor([inner, columns], ws)
    let b = Tensor([columns], bs)
    defer { x.free() }
    defer { w.free() }
    defer { b.free() }

    let fused = jt.linear_gelu(x, w, b)
    defer { fused.free() }
    let apart = jt.gelu(jt.add(jt.matmul(x, w), b))
    defer { apart.free() }
    let got = fused.download()
    let want = apart.download()
    assert_eq(got.len(), want.len())
    for index in 0..got.len() {
        assert_near(got[index], want[index], 1e-4, f"at {index}")
    }
}

#: Training takes the unfused path, because GELU's derivative needs what went
#: into it and the fused kernel keeps only what came out.
fn test_linear_gelu_still_differentiates() -> void {
    if not jt.is_available() { return }
    let x = Tensor([4, 6])
    let w = Tensor([6, 5])
    let b = Tensor.zeros([5])
    x.buffer().fill_uniform(-1.0, 1.0, 3)
    w.buffer().fill_uniform(-1.0, 1.0, 4)
    w.requires_grad = true
    let out = jt.linear_gelu(x, w, b)
    assert_eq(out.shape, [4, 5])
    jt.sum(out).backward()
    let grad = w.grad
    assert_true(grad is not null, "the weight got no gradient")
}

#: The fused residual-and-product gives what the two steps give.
fn test_linear_residual_matches_the_two_steps() -> void {
    if not jt.is_available() { return }
    let rows = 37
    let inner = 53
    let columns = 71
    var xs: list[float] = []
    for index in 0..rows * inner { xs.push(float(index * 19 % 71 - 35) * 0.04) }
    var ws: list[float] = []
    for index in 0..inner * columns { ws.push(float(index * 29 % 53 - 26) * 0.02) }
    var rs: list[float] = []
    for index in 0..rows * columns { rs.push(float(index * 13 % 37 - 18) * 0.07) }
    let x = Tensor([rows, inner], xs)
    let w = Tensor([inner, columns], ws)
    let residual = Tensor([rows, columns], rs)
    defer { x.free() }
    defer { w.free() }
    defer { residual.free() }

    let fused = jt.linear_residual(x, w, residual)
    defer { fused.free() }
    let apart = jt.add(residual, jt.matmul(x, w))
    defer { apart.free() }
    let got = fused.download()
    let want = apart.download()
    assert_eq(got.len(), want.len())
    for index in 0..got.len() { assert_near(got[index], want[index], 1e-4, f"at {index}") }
}

fn test_linear_residual_refuses_a_residual_of_the_wrong_shape() -> void {
    if not jt.is_available() { return }
    let x = Tensor([4, 6])
    let w = Tensor([6, 5])
    let wrong = Tensor([4, 6])
    defer { x.free() }
    defer { w.free() }
    defer { wrong.free() }
    assert_throws(ValueError, || jt.linear_residual(x, w, wrong))
}
