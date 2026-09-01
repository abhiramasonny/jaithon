#: The matmul routes: tile plans, the staged and MPS kernels, and the fused affine.
import jaitensor as jt
import std.gpu as gpu
from jaitensor.gemm import default_plan, plan_from_text
from jaitensor.ops import matmul_backend
from jaitensor.tensor import Tensor
from std.random import Random
from std.test import assert_eq, assert_near, assert_true

fn device_tensor(shape: list[int], seed: int) -> Tensor {
    let tensor = jt.Tensor(shape)
    tensor.buffer().fill_uniform(-1.0, 1.0, seed)
    return tensor
}

fn assert_gpu_matmul_probes(
    product: Tensor,
    a: Tensor,
    b: Tensor,
    rows: int,
    inner: int,
    columns: int,
    trans_a: bool = false,
    trans_b: bool = false,
    tol: float = 5e-3
) -> void {
    let got = product.download()
    let a_vals = a.download()
    let b_vals = b.download()
    let probes = [
        [0, 0],
        [0, columns - 1],
        [rows - 1, 0],
        [rows - 1, columns - 1],
        [rows // 2, columns // 2],
    ]
    for probe in probes {
        let row = probe[0]
        let column = probe[1]
        var want = 0.0
        for k in 0..inner {
            let av = if trans_a { a_vals[k * rows + row] } else { a_vals[row * inner + k] }
            let bv = if trans_b { b_vals[column * inner + k] } else { b_vals[k * columns + column] }
            want += av * bv
        }
        assert_near(got[row * columns + column], want, tol)
    }
}

fn assert_gpu_matmul(
    rows: int,
    inner: int,
    columns: int,
    trans_a: bool = false,
    trans_b: bool = false,
    tol: float = 5e-3
) -> void {
    let a = if trans_a { device_tensor([inner, rows], 1) } else { device_tensor([rows, inner], 1) }
    let b = if trans_b {
        device_tensor([columns, inner], 2)
    } else {
        device_tensor([inner, columns], 2)
    }
    defer { a.free() }
    defer { b.free() }
    gpu.flush()
    let product = jt.matmul(a, b, trans_a: trans_a, trans_b: trans_b)
    defer { product.free() }
    assert_gpu_matmul_probes(
        product,
        a,
        b,
        rows,
        inner,
        columns,
        trans_a: trans_a,
        trans_b: trans_b,
        tol: tol
    )
}

fn reduce_to_scalar_for_test(t: Tensor) -> Tensor {
    return jt.sum(t)
}

fn test_tiled_matmul_matches_cpu_on_ragged_tiles() -> void {
    if not jt.is_available() { return }
    let a = jt.Tensor([17, 19], [float(i * 3 % 11) for i in 0..17 * 19])
    let b = jt.Tensor([19, 18], [float(i * 5 % 7) for i in 0..19 * 18])
    defer { a.free() }
    defer { b.free() }
    let product = jt.matmul(a, b)
    defer { product.free() }
    let got = product.download()
    let a_vals = a.download()
    let b_vals = b.download()
    for row in 0..17 {
        for column in 0..18 {
            var want = 0.0
            for k in 0..19 {
                want += a_vals[row * 19 + k] * b_vals[k * 18 + column]
            }
            assert_near(got[row * 18 + column], want, 1e-3)
        }
    }
}

fn test_default_plan_prefers_large_tiles_then_splits() -> void {
    assert_eq(default_plan(512, 784, 8192).to_str(), "gemm128x128:1")
    assert_eq(default_plan(1024, 3072, 2048).to_str(), "gemm128x128:2")
    assert_eq(default_plan(512, 784, 256).to_str(), "gemm128x64:8")
    assert_eq(default_plan(8, 7, 5).to_str(), "gemm32x32:1")
}

fn test_matmul_backend_reports_a_known_plan() -> void {
    #: Whatever the tuner has decided for this machine, the reported route has
    #: to name a plan the dispatcher can read back.
    let reported = matmul_backend(512, 784, 256)
    let parsed = plan_from_text(reported)
    assert_true(parsed is not null)
    assert_eq(parsed.to_str(), reported)
}

fn test_staged_64_matmul_matches_cpu() -> void {
    if not jt.is_available() { return }
    assert_gpu_matmul(64, 1024, 64)
}

fn test_wide_nn2_matmul_matches_cpu() -> void {
    if not jt.is_available() { return }
    assert_gpu_matmul(128, 80, 2048)
}

fn test_mps_scale_matmul_matches_cpu() -> void {
    if not jt.is_available() { return }
    assert_gpu_matmul(512, 2048, 1024, tol: 2e-2)
}

fn test_mps_scale_transposed_matmul_matches_cpu() -> void {
    if not jt.is_available() { return }
    let rows = 784
    let inner = 512
    let columns = 4096
    let a = jt.Tensor([inner, rows], [float(i * 3 % 11) for i in 0..inner * rows])
    let b = jt.Tensor([inner, columns], [float(i * 5 % 7) for i in 0..inner * columns])
    defer { a.free() }
    defer { b.free() }
    let product = jt.matmul(a, b, trans_a: true)
    defer { product.free() }
    let got = product.download()
    let a_vals = a.download()
    let b_vals = b.download()
    for row in [0, 100, 783] {
        for column in [0, 1, 64, 2048, 4095] {
            var want = 0.0
            for k in 0..inner {
                want += a_vals[k * rows + row] * b_vals[k * columns + column]
            }
            assert_near(got[row * columns + column], want, 2e-2)
        }
    }
}

fn test_mps_fashion_weight_grad_matmul_matches_cpu() -> void {
    if not jt.is_available() { return }
    let rows = 784
    let inner = 512
    let columns = 256
    let a = jt.Tensor([inner, rows], [float(i * 3 % 11) for i in 0..inner * rows])
    let b = jt.Tensor([inner, columns], [float(i * 5 % 7) for i in 0..inner * columns])
    defer { a.free() }
    defer { b.free() }
    let product = jt.matmul(a, b, trans_a: true)
    defer { product.free() }
    let got = product.download()
    let a_vals = a.download()
    let b_vals = b.download()
    for row in [0, 100, 783] {
        for column in [0, 1, 64, 255] {
            var want = 0.0
            for k in 0..inner {
                want += a_vals[k * rows + row] * b_vals[k * columns + column]
            }
            assert_near(got[row * columns + column], want, 2e-2)
        }
    }
}

fn test_mps_cifar_layer_matmul_matches_cpu() -> void {
    if not jt.is_available() { return }
    assert_gpu_matmul(512, 3072, 512, tol: 2e-2)
}

fn test_staged_64_transposed_matmul_matches_cpu() -> void {
    if not jt.is_available() { return }
    let m = 64
    let k = 1024
    let n = 64
    let a_tn = jt.Tensor([k, m], [float(i * 3 % 11) for i in 0..k * m])
    let b_tn = jt.Tensor([k, n], [float(i * 5 % 7) for i in 0..k * n])
    defer { a_tn.free() }
    defer { b_tn.free() }
    let tn = jt.matmul(a_tn, b_tn, trans_a: true)
    defer { tn.free() }
    let a_nt = jt.Tensor([m, k], [float(i * 3 % 11) for i in 0..m * k])
    let b_nt = jt.Tensor([n, k], [float(i * 5 % 7) for i in 0..n * k])
    defer { a_nt.free() }
    defer { b_nt.free() }
    let nt = jt.matmul(a_nt, b_nt, trans_b: true)
    defer { nt.free() }
    let a_tn_vals = a_tn.download()
    let b_tn_vals = b_tn.download()
    let a_nt_vals = a_nt.download()
    let b_nt_vals = b_nt.download()
    let tn_got = tn.download()
    let nt_got = nt.download()
    for row in 0..m {
        for column in 0..n {
            var want_tn = 0.0
            var want_nt = 0.0
            for inner in 0..k {
                want_tn += a_tn_vals[inner * m + row] * b_tn_vals[inner * n + column]
                want_nt += a_nt_vals[row * k + inner] * b_nt_vals[column * k + inner]
            }
            assert_near(tn_got[row * n + column], want_tn, 5e-3)
            assert_near(nt_got[row * n + column], want_nt, 5e-3)
        }
    }
}

fn test_staged_64_matmul_covers_row_remainder() -> void {
    if not jt.is_available() { return }
    assert_gpu_matmul(80, 1024, 64)
}

fn test_staged_64_matmul_covers_l_remainder() -> void {
    if not jt.is_available() { return }
    assert_gpu_matmul(80, 1024, 100)
}

fn test_staged_64_matmul_ragged_k_matches_cpu() -> void {
    if not jt.is_available() { return }
    assert_gpu_matmul(64, 784, 1024)
}

fn test_mma_ragged_k_transposed_matmul_matches_cpu() -> void {
    if not jt.is_available() { return }
    let m = 64
    let k = 100
    let n = 64
    let a_nt = jt.Tensor([m, k], [float(i * 3 % 11) for i in 0..m * k])
    let b_nt = jt.Tensor([n, k], [float(i * 5 % 7) for i in 0..n * k])
    defer { a_nt.free() }
    defer { b_nt.free() }
    let nt = jt.matmul(a_nt, b_nt, trans_b: true)
    defer { nt.free() }
    let a_vals = a_nt.download()
    let b_vals = b_nt.download()
    let got = nt.download()
    for row in 0..m {
        for column in 0..n {
            var want = 0.0
            for inner in 0..k {
                want += a_vals[row * k + inner] * b_vals[column * k + inner]
            }
            assert_near(got[row * n + column], want, 5e-3)
        }
    }
}

fn test_aligned_256_matmul_matches_cpu() -> void {
    if not jt.is_available() { return }
    assert_gpu_matmul(64, 256, 256)
}

fn test_fused_64_relu_affine_backward_matches_cpu() -> void {
    if not jt.is_available() { return }
    let batch = 64
    let inputs = 1024
    let units = 64
    let x_vals = [float(i * 3 % 11) / 11.0 for i in 0..batch * inputs]
    let w_vals = [float(i * 5 % 7) / 7.0 - 0.5 for i in 0..inputs * units]
    let b_vals = [float(i % 5) / 5.0 - 0.4 for i in 0..units]
    let layer = jt.Dense(units, activation: jt.Activation.Relu)
    let rng = Random.with_seed(1)
    layer.build(inputs, rng)
    layer.set_weights(w_vals, b_vals)
    defer { layer.free() }
    let x = jt.Tensor([batch, inputs], x_vals, requires_grad: true)
    defer { x.free() }
    let y = layer.forward(x)
    defer { y.free() }
    let total = reduce_to_scalar_for_test(y)
    defer { total.free() }
    let intermediates = total.backward()
    defer {
        for tensor in intermediates { tensor.free() }
    }
    let params = layer.parameters()
    let grad_w = params[0].grad
    let grad_b = params[1].grad
    let grad_x = x.grad
    if grad_w is null or grad_b is null or grad_x is null {
        throw AssertionError("expected fused relu affine grads")
    }
    let gw = grad_w.download()
    let gb = grad_b.download()
    let gx = grad_x.download()
    var d_aff = [0.0 for _i in 0..batch * units]
    for row in 0..batch {
        for unit in 0..units {
            var aff = b_vals[unit]
            for k in 0..inputs {
                aff += x_vals[row * inputs + k] * w_vals[k * units + unit]
            }
            d_aff[row * units + unit] = if aff > 0.0 { 1.0 } else { 0.0 }
        }
    }
    for unit in 0..units {
        var want_b = 0.0
        for row in 0..batch { want_b += d_aff[row * units + unit] }
        assert_near(gb[unit], want_b, 5e-3)
    }
    for k in 0..inputs {
        for unit in 0..units {
            var want_w = 0.0
            for row in 0..batch {
                want_w += x_vals[row * inputs + k] * d_aff[row * units + unit]
            }
            assert_near(gw[k * units + unit], want_w, 5e-3)
        }
    }
    for row in 0..batch {
        for k in 0..inputs {
            var want_x = 0.0
            for unit in 0..units {
                want_x += d_aff[row * units + unit] * w_vals[k * units + unit]
            }
            assert_near(gx[row * inputs + k], want_x, 5e-3)
        }
    }
}

fn test_staged_128_matmul_matches_cpu() -> void {
    if not jt.is_available() { return }
    assert_gpu_matmul(128, 1024, 128)
}

fn test_staged_128_transposed_matmul_matches_cpu() -> void {
    if not jt.is_available() { return }
    let m = 128
    let k = 1024
    let n = 128
    let a_tn = jt.Tensor([k, m], [float(i * 3 % 11) for i in 0..k * m])
    let b_tn = jt.Tensor([k, n], [float(i * 5 % 7) for i in 0..k * n])
    defer { a_tn.free() }
    defer { b_tn.free() }
    let tn = jt.matmul(a_tn, b_tn, trans_a: true)
    defer { tn.free() }
    let a_nt = jt.Tensor([m, k], [float(i * 3 % 11) for i in 0..m * k])
    let b_nt = jt.Tensor([n, k], [float(i * 5 % 7) for i in 0..n * k])
    defer { a_nt.free() }
    defer { b_nt.free() }
    let nt = jt.matmul(a_nt, b_nt, trans_b: true)
    defer { nt.free() }
    let a_tn_vals = a_tn.download()
    let b_tn_vals = b_tn.download()
    let a_nt_vals = a_nt.download()
    let b_nt_vals = b_nt.download()
    let tn_got = tn.download()
    let nt_got = nt.download()
    for row in 0..m {
        for column in 0..n {
            var want_tn = 0.0
            var want_nt = 0.0
            for inner in 0..k {
                want_tn += a_tn_vals[inner * m + row] * b_tn_vals[inner * n + column]
                want_nt += a_nt_vals[row * k + inner] * b_nt_vals[column * k + inner]
            }
            assert_near(tn_got[row * n + column], want_tn, 5e-3)
            assert_near(nt_got[row * n + column], want_nt, 5e-3)
        }
    }
}

fn test_mixed_precision_matmul_stays_close() -> void {
    if not jt.is_available() { return }
    let a = jt.Tensor(
        [4, 4],
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    )
    let b = jt.Tensor(
        [4, 4],
        [2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0]
    )
    jt.set_mixed_precision(true)
    let y = jt.matmul(a, b)
    jt.set_mixed_precision(false)
    defer { a.free() }
    defer { b.free() }
    defer { y.free() }
    let values = y.download()
    assert_near(values[0], 2.0, 1e-3)
    assert_near(values[5], 3.0, 1e-3)
    assert_near(values[10], 4.0, 1e-3)
    assert_near(values[15], 5.0, 1e-3)
}
