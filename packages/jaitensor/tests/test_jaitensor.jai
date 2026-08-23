#: GPU tensor operations, generic datasets, optimizers, and model lifecycle.
import jaitensor as jt
import std.gpu as gpu
import std.math
import std.os
from jaitensor import Dataset
from jaitensor.gemm import default_plan, plan_from_text
from jaitensor.ops import dense_static_forward, matmul_backend
from jaitensor.tensor import GradNode, Tensor
from std.io import Path, read_file
from std.random import Random
from std.test import assert_eq, assert_near, assert_throws, assert_true
from std.trace import is_traced

fn assert_values_near(actual: list[float], expected: list[float], tolerance: float = 1e-5) -> void {
    assert_eq(actual.len(), expected.len())
    for (index, value) in actual.enumerate() {
        assert_near(value, expected[index], tolerance)
    }
}

fn numerical_grad(
    f: fn(list[float]) -> float,
    x: list[float],
    index: int,
    eps: float = 1e-3
) -> float {
    var plus = [value for value in x]
    var minus = [value for value in x]
    plus[index] += eps
    minus[index] -= eps
    return (f(plus) - f(minus)) / (2.0 * eps)
}

fn reduce_to_scalar_for_test(t: Tensor) -> Tensor {
    return jt.sum(t)
}

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

fn separable_data(repeats: int = 32) -> Dataset {
    var features: list[float] = []
    var labels: list[int] = []
    for _repeat in 0..repeats {
        features.extend([-2.0, -1.0, -1.0, -2.0, 2.0, 1.0, 1.0, 2.0])
        labels.extend([0, 0, 1, 1])
    }
    return jt.Dataset.from_flat(features, labels, 2)
}

fn test_tensor_constructors_shape_helpers_and_round_trip() -> void {
    if not jt.is_available() { return }
    let tensor = jt.Tensor([2, 2], [1.0, 2.0, 3.0, 4.0])
    let constant = jt.constant([5.0, 6.0], [2])
    let zeroes = jt.zeros([2])
    let ones = jt.ones([3])
    let full = jt.fill([2], 2.5)
    defer { tensor.free() }
    defer { constant.free() }
    defer { zeroes.free() }
    defer { ones.free() }
    defer { full.free() }
    assert_eq(tensor.shape, [2, 2])
    assert_eq(tensor.rank(), 2)
    assert_eq(tensor.dim(0), 2)
    assert_eq(tensor.dim(-1), 2)
    assert_eq(tensor.download(), [1.0, 2.0, 3.0, 4.0])
    assert_eq(constant.download(), [5.0, 6.0])
    assert_eq(zeroes.download(), [0.0, 0.0])
    assert_eq(ones.download(), [1.0, 1.0, 1.0])
    assert_eq(full.download(), [2.5, 2.5])
    let _error = assert_throws(IndexError, || tensor.dim(2))
}

fn test_backward_accumulates_gradients_through_a_hand_built_diamond_graph() -> void {
    if not jt.is_available() { return }
    let a = jt.Tensor([2], [3.0, 4.0], requires_grad: true)
    let left = jt.Tensor([2])
    let right = jt.Tensor([2])
    let out = jt.Tensor([1])
    left.creator = GradNode([a], |upstream| [jt.Tensor([2], upstream.download())])
    right.creator = GradNode([a], |upstream| [jt.Tensor([2], upstream.download())])
    out.creator = GradNode(
        [left, right],
        |upstream| [jt.Tensor([2], upstream.download()), jt.Tensor([2], upstream.download())]
    )
    out.grad = jt.Tensor.ones([2])
    let intermediates = out.backward()
    let grad = a.grad
    if grad is null { throw AssertionError("expected a.grad to be set after backward()") }
    assert_eq(grad.download(), [2.0, 2.0])
    assert_eq(intermediates.len(), 2)
    for tensor in intermediates {
        assert not tensor.is_freed()
        tensor.free()
    }
    a.free()
    out.free()
}

fn test_backward_seeds_the_root_gradient_when_the_caller_never_sets_it() -> void {
    if not jt.is_available() { return }
    let x = jt.Tensor([1], [5.0], requires_grad: true)
    let out = jt.Tensor([1])
    out.creator = GradNode([x], |upstream| [jt.Tensor([1], upstream.download())])
    let intermediates = out.backward()
    let grad = x.grad
    if grad is null { throw AssertionError("expected x.grad to be set after backward()") }
    assert_eq(grad.download(), [1.0])
    assert_eq(intermediates.len(), 0)
    x.free()
    out.free()
}

fn test_dataset_constructors_are_format_agnostic() -> void {
    if not jt.is_available() { return }
    let rows = jt.Dataset.from_tensor_slices([[1.0, 2.0], [3.0, 4.0]], [4, 7])
    let packed = jt.Dataset.from_u8(
        bytes([99, 2, 4, 6, 8]),
        bytes([99, 3, 5]),
        2,
        2,
        feature_offset: 1,
        label_offset: 1,
        feature_scale: 0.5
    )
    defer { rows.free() }
    defer { packed.free() }
    assert_eq(rows.samples, 2)
    assert_eq(rows.feature_count, 2)
    assert_eq(rows.batch_count(1), 2)
    assert_eq(rows.batch_count(3), 1)
    assert_eq(rows.labels.download(), [4.0, 7.0])
    assert_eq(packed.features.download(), [1.0, 2.0, 3.0, 4.0])
    assert_eq(packed.labels.download(), [3.0, 5.0])
    let _ragged = assert_throws(ValueError, || jt.Dataset.from_rows([[1.0], [2.0, 3.0]], [0, 1]))
}

fn test_elementwise_tensor_operations_run_on_device() -> void {
    if not jt.is_available() { return }
    let a = jt.Tensor([2, 2], [1.0, 2.0, 3.0, 4.0])
    let b = jt.Tensor([2, 2], [4.0, 3.0, 2.0, 1.0])
    let added = jt.add(a, b)
    let subtracted = jt.subtract(a, b)
    let multiplied = jt.multiply(a, b)
    let divided = jt.divide(a, b)
    let scaled = jt.scale(a, 0.5)
    let reshaped = jt.reshape(a, [4])
    defer { a.free() }
    defer { b.free() }
    defer { added.free() }
    defer { subtracted.free() }
    defer { multiplied.free() }
    defer { divided.free() }
    defer { scaled.free() }
    defer { reshaped.free() }
    assert_values_near(added.download(), [5.0, 5.0, 5.0, 5.0])
    assert_values_near(subtracted.download(), [-3.0, -1.0, 1.0, 3.0])
    assert_values_near(multiplied.download(), [4.0, 6.0, 6.0, 4.0])
    assert_values_near(divided.download(), [0.25, 2.0 / 3.0, 1.5, 4.0])
    assert_values_near(scaled.download(), [0.5, 1.0, 1.5, 2.0])
    assert_eq(reshaped.shape, [4])
}

fn test_matrix_activation_and_classification_operations() -> void {
    if not jt.is_available() { return }
    let a = jt.Tensor([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    let b = jt.Tensor([3, 2], [1.0, 2.0, 0.0, 1.0, 2.0, 0.0])
    let product = jt.matmul(a, b)
    let transposed = jt.transpose(a)
    let values = jt.Tensor([1, 3], [-1.0, 0.0, 1.0])
    let relu = jt.relu(values)
    let sigmoid = jt.sigmoid(values)
    let tanh = jt.tanh(values)
    let probabilities = jt.softmax(values)
    let classes = jt.argmax(probabilities)
    let hot = jt.one_hot(classes, 3)
    defer { a.free() }
    defer { b.free() }
    defer { product.free() }
    defer { transposed.free() }
    defer { values.free() }
    defer { relu.free() }
    defer { sigmoid.free() }
    defer { tanh.free() }
    defer { probabilities.free() }
    defer { classes.free() }
    defer { hot.free() }
    assert_values_near(product.download(), [7.0, 4.0, 16.0, 13.0])
    assert_values_near(transposed.download(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
    assert_values_near(relu.download(), [0.0, 0.0, 1.0])
    assert_values_near(sigmoid.download(), [0.268941, 0.5, 0.731059], 1e-5)
    assert_values_near(tanh.download(), [-0.761594, 0.0, 0.761594], 1e-5)
    let result = probabilities.download()
    assert_near(result[0] + result[1] + result[2], 1.0, 1e-5)
    assert_eq(classes.download(), [2.0])
    assert_eq(hot.download(), [0.0, 0.0, 1.0])
}

fn test_add_broadcasts_a_trailing_suffix_shape() -> void {
    if not jt.is_available() { return }
    let batched = jt.Tensor([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    let bias = jt.Tensor([3], [10.0, 20.0, 30.0])
    let result = jt.add(batched, bias)
    assert_eq(result.download(), [11.0, 22.0, 33.0, 14.0, 25.0, 36.0])
    batched.free()
    bias.free()
    result.free()
}

fn test_slice_flat_copies_a_device_range() -> void {
    if not jt.is_available() { return }
    let source = jt.Tensor([4, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    defer { source.free() }
    let sliced = jt.slice_flat(source, 3, [2, 3])
    defer { sliced.free() }
    assert_eq(sliced.shape, [2, 3])
    assert_eq(sliced.download(), [4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    assert_true(not sliced.requires_grad)
    assert_true(sliced.creator is null)
    let _error = assert_throws(ValueError, || jt.slice_flat(source, 10, [2, 3]))
}

fn test_tensor_view_reads_a_parent_range_without_copying() -> void {
    if not jt.is_available() { return }
    let source = jt.Tensor([4, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    defer { source.free() }
    let viewed = source.view(3, [2, 3])
    defer { viewed.free() }
    assert_eq(viewed.shape, [2, 3])
    assert_eq(viewed.download(), [4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    let _error = assert_throws(ValueError, || source.view(10, [2, 3]))
}

fn test_device_metric_kernels_accumulate_loss_and_correct_counts() -> void {
    if not jt.is_available() { return }
    let logits = jt.Tensor([2, 3], [0.0, 2.0, 1.0, 4.0, 0.0, 1.0])
    let labels = jt.Tensor([2], [1.0, 0.0])
    let correct = jt.zeros([1])
    let loss_total = jt.zeros([1])
    let mean = jt.Tensor([1], [0.5])
    defer { logits.free() }
    defer { labels.free() }
    defer { correct.free() }
    defer { loss_total.free() }
    defer { mean.free() }
    jt.add_correct_into(logits, labels, correct)
    jt.add_scaled_into(mean, loss_total, 4)
    assert_values_near(correct.download(), [2.0])
    assert_values_near(loss_total.download(), [2.0])
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

fn test_add_rejects_a_shape_that_is_not_a_trailing_suffix() -> void {
    if not jt.is_available() { return }
    let a = jt.Tensor([2, 3], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    let b = jt.Tensor([4], [0.0, 0.0, 0.0, 0.0])
    var raised = false
    try { jt.add(a, b) } catch e: ValueError { raised = true }
    assert raised
    a.free()
    b.free()
}

fn test_add_and_multiply_backward_match_numerical_gradients() -> void {
    if not jt.is_available() { return }
    fn loss_of(values: list[float]) -> float {
        let a = jt.Tensor([2], [values[0], values[1]])
        let b = jt.Tensor([2], [values[2], values[3]])
        let sum = jt.add(a, b)
        let product = jt.multiply(sum, a)
        let total = reduce_to_scalar_for_test(product)
        let result = total.download()[0]
        a.free()
        b.free()
        sum.free()
        product.free()
        total.free()
        return result
    }
    let x = [2.0, -1.0, 0.5, 3.0]
    let a = jt.Tensor([2], [x[0], x[1]], requires_grad: true)
    let b = jt.Tensor([2], [x[2], x[3]], requires_grad: true)
    let sum = jt.add(a, b)
    let product = jt.multiply(sum, a)
    let total = reduce_to_scalar_for_test(product)
    let intermediates = total.backward()
    let a_grad = a.grad
    let b_grad = b.grad
    if a_grad is null { throw AssertionError("expected a.grad to be set after backward()") }
    if b_grad is null { throw AssertionError("expected b.grad to be set after backward()") }
    let analytic = a_grad.download() + b_grad.download()
    for index in 0..4 {
        let expected = numerical_grad(loss_of, x, index)
        assert math.fabs(analytic[index] - expected) < 1e-2
    }
    for tensor in intermediates { tensor.free() }
    a.free()
    b.free()
    total.free()
}

fn test_subtract_and_divide_backward_match_numerical_gradients() -> void {
    if not jt.is_available() { return }
    fn loss_of(values: list[float]) -> float {
        let a = jt.Tensor([2], [values[0], values[1]])
        let b = jt.Tensor([2], [values[2], values[3]])
        let diff = jt.subtract(a, b)
        let quotient = jt.divide(diff, a)
        let total = reduce_to_scalar_for_test(quotient)
        let result = total.download()[0]
        a.free()
        b.free()
        diff.free()
        quotient.free()
        total.free()
        return result
    }
    let x = [2.0, -1.0, 0.5, 3.0]
    let a = jt.Tensor([2], [x[0], x[1]], requires_grad: true)
    let b = jt.Tensor([2], [x[2], x[3]], requires_grad: true)
    let diff = jt.subtract(a, b)
    let quotient = jt.divide(diff, a)
    let total = reduce_to_scalar_for_test(quotient)
    let intermediates = total.backward()
    let a_grad = a.grad
    let b_grad = b.grad
    if a_grad is null { throw AssertionError("expected a.grad to be set after backward()") }
    if b_grad is null { throw AssertionError("expected b.grad to be set after backward()") }
    let analytic = a_grad.download() + b_grad.download()
    for index in 0..4 {
        let expected = numerical_grad(loss_of, x, index)
        assert math.fabs(analytic[index] - expected) < 1e-2
    }
    for tensor in intermediates { tensor.free() }
    a.free()
    b.free()
    total.free()
}

fn test_broadcasting_add_backward_matches_numerical_gradients() -> void {
    if not jt.is_available() { return }
    fn loss_of(values: list[float]) -> float {
        let batched = jt.Tensor(
            [2, 3],
            [values[0], values[1], values[2], values[3], values[4], values[5]]
        )
        let bias = jt.Tensor([3], [values[6], values[7], values[8]])
        let result = jt.add(batched, bias)
        let total = reduce_to_scalar_for_test(result)
        let value = total.download()[0]
        batched.free()
        bias.free()
        result.free()
        total.free()
        return value
    }
    let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5, -1.0, 2.0]
    let batched = jt.Tensor([2, 3], [x[0], x[1], x[2], x[3], x[4], x[5]], requires_grad: true)
    let bias = jt.Tensor([3], [x[6], x[7], x[8]], requires_grad: true)
    let result = jt.add(batched, bias)
    let total = reduce_to_scalar_for_test(result)
    let intermediates = total.backward()
    let batched_grad = batched.grad
    let bias_grad = bias.grad
    if batched_grad is null {
        throw AssertionError("expected batched.grad to be set after backward()")
    }
    if bias_grad is null { throw AssertionError("expected bias.grad to be set after backward()") }
    let analytic = batched_grad.download() + bias_grad.download()
    for index in 0..9 {
        let expected = numerical_grad(loss_of, x, index)
        assert math.fabs(analytic[index] - expected) < 1e-2
    }
    for tensor in intermediates { tensor.free() }
    batched.free()
    bias.free()
    total.free()
}

fn test_matmul_backward_matches_numerical_gradients() -> void {
    if not jt.is_available() { return }
    fn loss_of(values: list[float]) -> float {
        let a = jt.Tensor([2, 2], [values[0], values[1], values[2], values[3]])
        let b = jt.Tensor([2, 2], [values[4], values[5], values[6], values[7]])
        let product = jt.matmul(a, b)
        let total = reduce_to_scalar_for_test(product)
        let result = total.download()[0]
        a.free()
        b.free()
        product.free()
        total.free()
        return result
    }
    let x = [1.0, 2.0, 3.0, 4.0, 0.5, -0.5, 1.5, 2.5]
    let a = jt.Tensor([2, 2], [x[0], x[1], x[2], x[3]], requires_grad: true)
    let b = jt.Tensor([2, 2], [x[4], x[5], x[6], x[7]], requires_grad: true)
    let product = jt.matmul(a, b)
    let total = reduce_to_scalar_for_test(product)
    let intermediates = total.backward()
    let a_grad = a.grad
    let b_grad = b.grad
    if a_grad is null { throw AssertionError("expected a.grad to be set after backward()") }
    if b_grad is null { throw AssertionError("expected b.grad to be set after backward()") }
    let analytic = a_grad.download() + b_grad.download()
    for index in 0..8 {
        let expected = numerical_grad(loss_of, x, index)
        assert math.fabs(analytic[index] - expected) < 1e-2
    }
    for tensor in intermediates { tensor.free() }
    a.free()
    b.free()
    total.free()
}

fn test_matmul_backward_succeeds_across_many_repeated_calls() -> void {
    if not jt.is_available() { return }
    for _index in 0..40 {
        let a = jt.Tensor([2, 2], [1.0, 2.0, 3.0, 4.0], requires_grad: true)
        let b = jt.Tensor([2, 2], [0.5, -0.5, 1.5, 2.5], requires_grad: true)
        let product = jt.matmul(a, b)
        let total = reduce_to_scalar_for_test(product)
        let intermediates = total.backward()
        let a_grad = a.grad
        let b_grad = b.grad
        if a_grad is null { throw AssertionError("expected a.grad to be set after backward()") }
        if b_grad is null { throw AssertionError("expected b.grad to be set after backward()") }
        for tensor in intermediates { tensor.free() }
        a.free()
        b.free()
        total.free()
    }
}

fn test_scale_transpose_reshape_backward_match_numerical_gradients() -> void {
    if not jt.is_available() { return }
    fn loss_of(values: list[float]) -> float {
        let a = jt.Tensor([2, 3], values)
        let scaled = jt.scale(a, 1.5)
        let transposed = jt.transpose(scaled)
        let reshaped = jt.reshape(transposed, [6])
        let total = reduce_to_scalar_for_test(reshaped)
        let result = total.download()[0]
        a.free()
        scaled.free()
        transposed.free()
        reshaped.free()
        total.free()
        return result
    }
    let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    let a = jt.Tensor([2, 3], x, requires_grad: true)
    let scaled = jt.scale(a, 1.5)
    let transposed = jt.transpose(scaled)
    let reshaped = jt.reshape(transposed, [6])
    let total = reduce_to_scalar_for_test(reshaped)
    let intermediates = total.backward()
    let grad = a.grad
    if grad is null { throw AssertionError("expected a.grad to be set after backward()") }
    let analytic = grad.download()
    for index in 0..6 {
        let expected = numerical_grad(loss_of, x, index)
        assert math.fabs(analytic[index] - expected) < 1e-2
    }
    for tensor in intermediates { tensor.free() }
    a.free()
    total.free()
}

fn test_relu_sigmoid_tanh_exp_log_backward_match_numerical_gradients() -> void {
    if not jt.is_available() { return }
    fn loss_with(name: str, x: list[float]) -> float {
        let a = jt.Tensor([3], x)
        let activated = match name {
            "relu" => jt.relu(a),
            "sigmoid" => jt.sigmoid(a),
            "tanh" => jt.tanh(a),
            "exp" => jt.exp(a),
            _ => jt.log(a),
        }
        let total = reduce_to_scalar_for_test(activated)
        let result = total.download()[0]
        a.free()
        activated.free()
        total.free()
        return result
    }
    let names = ["relu", "sigmoid", "tanh", "exp", "log"]
    let x = [0.6, 1.4, 2.2]
    for name in names {
        let a = jt.Tensor([3], x, requires_grad: true)
        let activated = match name {
            "relu" => jt.relu(a),
            "sigmoid" => jt.sigmoid(a),
            "tanh" => jt.tanh(a),
            "exp" => jt.exp(a),
            _ => jt.log(a),
        }
        let total = reduce_to_scalar_for_test(activated)
        let intermediates = total.backward()
        let grad = a.grad
        if grad is null { throw AssertionError("expected a.grad to be set after backward()") }
        let analytic = grad.download()
        for index in 0..3 {
            fn wrapped(values: list[float]) -> float { return loss_with(name, values) }
            let expected = numerical_grad(wrapped, x, index)
            assert math.fabs(analytic[index] - expected) < 1e-2
        }
        for tensor in intermediates { tensor.free() }
        a.free()
        total.free()
    }
}

fn test_softmax_backward_matches_numerical_gradients() -> void {
    if not jt.is_available() { return }
    fn loss_of(values: list[float]) -> float {
        let a = jt.Tensor([1, 3], values)
        let activated = jt.softmax(a)
        let total = reduce_to_scalar_for_test(activated)
        let result = total.download()[0]
        a.free()
        activated.free()
        total.free()
        return result
    }
    let x = [0.2, 1.5, -0.7]
    let a = jt.Tensor([1, 3], x, requires_grad: true)
    let activated = jt.softmax(a)
    let total = reduce_to_scalar_for_test(activated)
    let intermediates = total.backward()
    let grad = a.grad
    if grad is null { throw AssertionError("expected a.grad to be set after backward()") }
    let analytic = grad.download()
    for index in 0..3 {
        let expected = numerical_grad(loss_of, x, index)
        assert math.fabs(analytic[index] - expected) < 1e-2
    }
    for tensor in intermediates { tensor.free() }
    a.free()
    total.free()
}

fn test_softmax_backward_with_nonuniform_upstream_matches_numerical_gradients() -> void {
    if not jt.is_available() { return }
    fn loss_of(values: list[float]) -> float {
        let a = jt.Tensor([1, 3], values)
        let activated = jt.softmax(a)
        let weights = jt.Tensor([1, 3], [1.0, 2.0, 3.0])
        let weighted = jt.multiply(activated, weights)
        let total = reduce_to_scalar_for_test(weighted)
        let result = total.download()[0]
        a.free()
        activated.free()
        weights.free()
        weighted.free()
        total.free()
        return result
    }
    let x = [0.2, 1.5, -0.7]
    let a = jt.Tensor([1, 3], x, requires_grad: true)
    let activated = jt.softmax(a)
    let weights = jt.Tensor([1, 3], [1.0, 2.0, 3.0])
    let weighted = jt.multiply(activated, weights)
    let total = reduce_to_scalar_for_test(weighted)
    let intermediates = total.backward()
    let grad = a.grad
    if grad is null { throw AssertionError("expected a.grad to be set after backward()") }
    let analytic = grad.download()
    for index in 0..3 {
        let expected = numerical_grad(loss_of, x, index)
        assert math.fabs(analytic[index] - expected) < 1e-2
    }
    for tensor in intermediates { tensor.free() }
    a.free()
    weights.free()
    total.free()
}

fn test_sparse_categorical_crossentropy_forward_and_backward_match_hand_computation() -> void {
    if not jt.is_available() { return }
    let logits = jt.Tensor([2, 2], [1.0, 2.0, 0.5, 0.5], requires_grad: true)
    let labels = jt.Tensor([2], [1.0, 0.0])
    let loss_fn = jt.SparseCategoricalCrossentropy(from_logits: true)
    let loss = loss_fn.forward(logits, labels)
    let e0 = math.exp(1.0)
    let e1 = math.exp(2.0)
    let sample0 = math.log(e0 + e1) - 2.0
    let sample1 = math.log(math.exp(0.5) + math.exp(0.5)) - 0.5
    let expected_loss = (sample0 + sample1) / 2.0
    assert math.fabs(loss.download()[0] - expected_loss) < 1e-4
    let intermediates = loss.backward()
    let grad = logits.grad
    if grad is null { throw AssertionError("expected logits.grad to be set after backward()") }
    let p0 = e0 / (e0 + e1)
    let p1 = e1 / (e0 + e1)
    let expected_grad = [p0 / 2.0, (p1 - 1.0) / 2.0, -0.5 / 2.0, 0.5 / 2.0]
    let analytic = grad.download()
    for index in 0..4 {
        assert math.fabs(analytic[index] - expected_grad[index]) < 1e-4
    }
    for tensor in intermediates { tensor.free() }
    logits.free()
    labels.free()
    loss.free()
}

fn test_dense_forward_matches_a_hand_computed_affine_plus_relu() -> void {
    if not jt.is_available() { return }
    let layer = jt.Dense(2, activation: jt.Activation.Relu)
    let rng = Random.with_seed(1)
    layer.build(3, rng)
    layer.set_weights([1.0, 0.0, 0.0, 1.0, -1.0, 1.0], [0.5, -0.5])
    let x = jt.Tensor([1, 3], [1.0, 2.0, 3.0])
    let y = layer.forward(x)
    assert_eq(y.download(), [0.0, 4.5])
    let intermediates = y.discard_graph()
    for tensor in intermediates { tensor.free() }
    x.free()
    y.free()
    layer.free()
}

fn test_a_tiny_sequential_models_loss_decreases_over_training_steps() -> void {
    if not jt.is_available() { return }
    let features = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    let labels = [0, 1, 1, 0]
    let data = jt.Dataset.from_flat(features, labels, 2)
    defer { data.free() }
    let model = jt.Sequential(
        [
            jt.Dense(8, activation: jt.Activation.Tanh),
            jt.Dense(2, activation: jt.Activation.Softmax),
        ]
    )
    defer { model.free() }
    model.compile(jt.Adam(learning_rate: 0.05), loss: jt.SparseCategoricalCrossentropy())
    let history = model.fit(data, epochs: 30, batch_size: 4, verbose: false, plot_path: null)
    assert history.loss[history.loss.len() - 1] < history.loss[0]
}

fn test_fit_leaves_no_leftover_gradient_on_any_parameter_between_batches() -> void {
    if not jt.is_available() { return }
    let data = separable_data()
    defer { data.free() }
    let model = jt.Sequential([jt.Dense(2, activation: jt.Activation.Softmax)])
    defer { model.free() }
    model.compile(jt.SGD(learning_rate: 0.1))
    model.fit(data, epochs: 2, batch_size: 64, verbose: false, plot_path: null)
    for layer in model.layers {
        for parameter in layer.parameters() {
            assert_true(parameter.grad is null)
        }
    }
}

fn test_fit_rejects_a_label_outside_the_models_class_count() -> void {
    if not jt.is_available() { return }
    let data = jt.Dataset.from_flat([0.0, 0.0, 0.0, 1.0], [0, 5], 2)
    defer { data.free() }
    let model = jt.Sequential([jt.Dense(2, activation: jt.Activation.Softmax)])
    defer { model.free() }
    model.compile(jt.SGD())
    let _error = assert_throws(ValueError, fn() -> any {
        model.fit(data, epochs: 1, batch_size: 2, verbose: false, plot_path: null)
        return null
    })
}

fn test_sequential_sgd_predicts_and_reports_validation_metrics() -> void {
    if not jt.is_available() { return }
    let data = separable_data()
    let model = jt.Sequential([jt.Dense(2, activation: jt.Activation.Softmax)])
    defer { data.free() }
    defer { model.free() }
    let plot = Path(os.temp_dir()).join("jaithon-jaitensor-fit-test.svg")
    defer { if plot.exists() { plot.remove() } }
    model.compile(jt.SGD(learning_rate: 0.4, momentum: 0.2))
    let history = model.fit(
        data,
        epochs: 8,
        batch_size: 64,
        verbose: false,
        validation_data: data,
        test_data: data,
        plot_path: plot.to_str()
    )
    assert_eq(history.accuracy.len(), 8)
    assert_eq(history.val_accuracy.len(), 8)
    assert_true(history.test_accuracy is not null)
    assert_true((history.test_accuracy ?? 0.0) >= 99.0)
    assert_eq(history.plot_path, plot.to_str())
    assert_true(plot.is_file())
    assert_true(history.accuracy[7] >= 99.0, f"accuracy was {history.accuracy[7]}")
    let probabilities = model.predict(data.features, batch_size: 31)
    defer { probabilities.free() }
    assert_eq(probabilities.shape, [128, 2])
    assert_eq(model.predict_classes(data.features).len(), 128)
}

fn test_predict_returns_a_detached_tensor_with_no_creator() -> void {
    if not jt.is_available() { return }
    let data = separable_data()
    defer { data.free() }
    let model = jt.Sequential([jt.Dense(2, activation: jt.Activation.Softmax)])
    defer { model.free() }
    model.compile(jt.SGD())
    let probabilities = model.predict(data.features)
    defer { probabilities.free() }
    assert_true(not probabilities.requires_grad)
    assert_true(probabilities.creator is null)
}

fn test_adam_trains_sigmoid_and_tanh_layers() -> void {
    if not jt.is_available() { return }
    let data = separable_data()
    let model = jt.Sequential(
        [
            jt.Dense(4, activation: jt.Activation.Tanh),
            jt.Dense(3, activation: jt.Activation.Sigmoid),
            jt.Dense(2, activation: jt.Activation.Linear),
        ]
    )
    defer { data.free() }
    defer { model.free() }
    model.compile(
        jt.Adam(learning_rate: 0.03),
        loss: jt.SparseCategoricalCrossentropy(from_logits: true)
    )
    let history = model.fit(data, epochs: 10, batch_size: 64, verbose: false, plot_path: null)
    assert_true(history.accuracy[9] >= 99.0, f"Adam accuracy was {history.accuracy[9]}")
}

fn test_weights_round_trip_through_memory_and_json() -> void {
    if not jt.is_available() { return }
    let data = separable_data(2)
    let source = jt.Sequential([jt.Dense(2, activation: jt.Activation.Softmax)])
    let restored = jt.Sequential([jt.Dense(2, activation: jt.Activation.Softmax)])
    let path = Path(os.temp_dir()).join("jaithon-jaitensor-weights-test.json")
    defer { data.free() }
    defer { source.free() }
    defer { restored.free() }
    defer { if path.exists() { path.remove() } }
    source.compile(jt.SGD())
    source.build(2)
    let weights = source.get_weights()
    source.set_weights(weights)
    source.save_weights(path.to_str())
    restored.compile(jt.SGD())
    restored.load_weights(path.to_str())
    assert_eq(restored.get_weights(), weights)
    assert_eq(restored.predict_classes(data.features), source.predict_classes(data.features))
}

#: A model with a convolution in it, saved and loaded back.
#:
#: The file recorded how many numbers a sample holds and not what shape they
#: are in, and a convolution cannot be rebuilt from the product: the load
#: went down the flat path, every shape-dependent layer came out the wrong
#: size, and it threw on the first parameter that did not match. So no model
#: with a convolution in it could be reloaded at all -- which is most of the
#: reason to save one.
fn test_a_convolutional_model_survives_a_save_and_a_load() -> void {
    if not jt.is_available() { return }
    let path = Path(os.temp_dir()).join("jaithon-jaitensor-conv-weights.json")
    defer { if path.exists() { path.remove() } }
    var pixels: list[float] = []
    var labels: list[float] = []
    var seed = 9
    for _sample in 0..24 {
        seed = (seed * 1103515245 + 12345) % 2147483648
        for _slot in 0..8 * 8 { pixels.push(float(seed % 100) / 100.0) }
        labels.push(float(seed % 3))
    }
    let data = jt.Dataset(jt.Tensor([24, 8, 8, 1], pixels), jt.Tensor([24], labels))
    defer { data.free() }

    let source = jt.Sequential(
        [
            jt.Conv2d(4, 3, padding: 1, in_channels: 1, activation: jt.Activation.Relu),
            jt.MaxPool2d(2),
            jt.Flatten(),
            jt.Dense(3, activation: jt.Activation.Softmax),
        ]
    )
    defer { source.free() }
    source.compile(optimizer: jt.SGD(learning_rate: 0.05), loss: jt.SparseCategoricalCrossentropy())
    source.fit(data, epochs: 2, batch_size: 8, verbose: false, plot_path: null)
    let flat = jt.Tensor([24, 64], pixels)
    let before = source.predict(flat).download()
    source.save_weights(path.to_str())

    let restored = jt.Sequential(
        [
            jt.Conv2d(
                4,
                3,
                padding: 1,
                in_channels: 1,
                activation: jt.Activation.Relu
            ),
            jt.MaxPool2d(2),
            jt.Flatten(),
            jt.Dense(3, activation: jt.Activation.Softmax),
        ]
    )
    defer { restored.free() }
    restored.load_weights(path.to_str())
    let after = restored.predict(flat).download()
    assert_eq(after.len(), before.len())
    for (index, value) in before.enumerate() {
        assert math.fabs(value - after[index]) < 1e-6
    }
}

#: The same parameter, stepped by one kind of optimizer and then another.
#:
#: The optimizer's state is kept in a table keyed by the parameter, and a
#: parameter does not know what stepped it last -- so Adam picked up whatever
#: plain descent had left in `first_moment` and read it as its own first
#: moment. After five momentum steps at 0.9 that velocity is about 41, and
#: Adam's opening step moved the parameter by 1.93 where a clean one moves
#: 0.01. A hundred and ninety times, silently.
#: The accuracy `fit` reports has to be the accuracy the model has.
#:
#: The fused path that runs a whole run of epochs on the device pushed a flat
#: zero into the history, every epoch, because it never asked the kernel for
#: the tally it was perfectly willing to keep. `History.accuracy` is
#: documented as the percentage after each epoch and `plot()` draws it, so a
#: model learning perfectly well showed a flat line along the bottom.
#: A table whose row length is not a multiple of four.
#:
#: The fused kernels take each batch as a window into the dataset, and the
#: platform's array view pads its innermost dimension out to sixteen bytes --
#: so it refuses to build a window that starts part way in and whose rows are
#: five, six, seven or nine floats long. `fit` did not fall back, it threw:
#: "gpu_mlp_sgd_epoch(): the kernel failed", for most of the widths a table
#: of measurements actually has. The first batch starts at offset zero and is
#: exempt, so a run of fewer than two batches appeared to work while training
#: on the first batch alone.
#: A model built for a flat row, then fitted on pictures of the same size.
#:
#: Sixty-four values is sixty-four values either way, and the build guard
#: compared only the count -- so it returned happy with every convolution
#: still sized for a flat row of 64, and the graph handed to the platform
#: failed its own verification and took the process down with SIGABRT. Not a
#: throw: exit 134, no traceback, nothing a `try` could catch.
fn test_building_flat_then_fitting_pictures_is_refused() -> void {
    if not jt.is_available() { return }
    var pixels: list[float] = []
    var labels: list[float] = []
    for index in 0..32 {
        for _slot in 0..64 { pixels.push(float(index % 5) * 0.2) }
        labels.push(float(index % 3))
    }
    let data = jt.Dataset(jt.Tensor([32, 8, 8, 1], pixels), jt.Tensor([32], labels))
    defer { data.free() }
    let model = jt.Sequential(
        [
            jt.Conv2d(4, 3, padding: 1, in_channels: 1, activation: jt.Activation.Relu),
            jt.MaxPool2d(2),
            jt.Flatten(),
            jt.Dense(3, activation: jt.Activation.Softmax),
        ]
    )
    defer { model.free() }
    model.compile(optimizer: jt.SGD(learning_rate: 0.01), loss: jt.SparseCategoricalCrossentropy())
    model.build(64)
    assert_throws(
        ValueError,
        || model.fit(data, epochs: 1, batch_size: 8, verbose: false, plot_path: null)
    )
}

fn test_a_model_trains_on_any_number_of_features() -> void {
    if not jt.is_available() { return }
    for width in [4, 5, 6, 7, 9, 16] {
        var features: list[float] = []
        var labels: list[float] = []
        var seed = 4
        for _index in 0..256 {
            seed = (seed * 1103515245 + 12345) % 2147483648
            let kind = seed % 4
            for _slot in 0..width { features.push(float(kind) + float(seed % 50) / 200.0) }
            labels.push(float(kind))
        }
        let data = jt.Dataset(jt.Tensor([256, width], features), jt.Tensor([256], labels))
        let model = jt.Sequential(
            [
                jt.Dense(256, activation: jt.Activation.Relu),
                jt.Dense(4, activation: jt.Activation.Softmax),
            ]
        )
        model.compile(
            optimizer: jt.SGD(learning_rate: 0.01),
            loss: jt.SparseCategoricalCrossentropy()
        )
        #: More than two batches, so a window at a non-zero offset is really
        #: asked for.
        let history = model.fit(data, epochs: 1, batch_size: 64, verbose: false, plot_path: null)
        assert_eq(history.loss.len(), 1)
        assert math.is_finite(history.loss[0])
        model.free()
        data.free()
    }
}

fn test_the_fused_path_reports_the_accuracy_it_reached() -> void {
    if not jt.is_available() { return }
    var features: list[float] = []
    var labels: list[float] = []
    var seed = 4
    for _index in 0..256 {
        seed = (seed * 1103515245 + 12345) % 2147483648
        let kind = seed % 4
        for _slot in 0..16 { features.push(float(kind) + float(seed % 50) / 200.0) }
        labels.push(float(kind))
    }
    let data = jt.Dataset(jt.Tensor([256, 16], features), jt.Tensor([256], labels))
    defer { data.free() }
    let model = jt.Sequential(
        [
            jt.Dense(512, activation: jt.Activation.Relu),
            jt.Dense(4, activation: jt.Activation.Softmax),
        ]
    )
    defer { model.free() }
    model.compile(optimizer: jt.SGD(learning_rate: 0.05), loss: jt.SparseCategoricalCrossentropy())
    #: No validation data and not verbose, which is what selects the path
    #: that runs every epoch on the device without coming back between them.
    let history = model.fit(data, epochs: 3, batch_size: 64, verbose: false, plot_path: null)
    assert_eq(history.accuracy.len(), 3)
    for value in history.accuracy {
        assert math.is_finite(value)
        assert value > 0.0
        assert value <= 100.0
    }
    #: And it agrees with what the model actually scores, near enough that a
    #: number invented from nowhere would not pass.
    let measured = model.evaluate(data).accuracy
    assert math.fabs(history.accuracy[2] - measured) < 15.0
}

fn test_a_second_kind_of_optimizer_starts_from_nothing() -> void {
    if not jt.is_available() { return }
    let shared = jt.Tensor([1], [0.0], requires_grad: true)
    for _round in 0..5 {
        shared.grad = jt.Tensor([1], [10.0])
        jt.SGD(learning_rate: 1.0e-9, momentum: 0.9).step([shared])
    }
    let before = shared.download()[0]
    shared.grad = jt.Tensor([1], [1.0])
    jt.Adam(learning_rate: 0.01).step([shared])
    let moved = shared.download()[0] - before

    let clean = jt.Tensor([1], [0.0], requires_grad: true)
    clean.grad = jt.Tensor([1], [1.0])
    jt.Adam(learning_rate: 0.01).step([clean])
    let expected = clean.download()[0]

    assert math.fabs(moved - expected) < 1e-6
    #: And that step really is the learning rate, so the comparison is not
    #: two wrong numbers agreeing.
    assert math.fabs(expected + 0.01) < 1e-4
    shared.free()
    clean.free()
}

fn test_sgd_step_momentum_persists_and_affects_a_second_step() -> void {
    if not jt.is_available() { return }
    let start = 2.0
    let parameter = jt.Tensor([1], [start], requires_grad: true)
    let optimizer = jt.SGD(learning_rate: 0.1, momentum: 0.5)

    let first_grad = jt.Tensor([1], [1.0])
    parameter.grad = first_grad
    optimizer.step([parameter])
    let value_1 = parameter.download()[0]
    let velocity_1 = (start - value_1) / 0.1

    first_grad.free()
    let second_grad = jt.Tensor([1], [2.0])
    parameter.grad = second_grad
    optimizer.step([parameter])
    let velocity_2 = 0.5 * velocity_1 + 2.0
    let expected_value_2 = value_1 - 0.1 * velocity_2
    assert math.fabs(parameter.download()[0] - expected_value_2) < 1e-5

    second_grad.free()
}

fn test_sgd_step_matches_the_textbook_update_on_one_parameter() -> void {
    if not jt.is_available() { return }
    let parameter = jt.Tensor([2], [1.0, -1.0], requires_grad: true)
    parameter.grad = jt.Tensor([2], [0.5, 2.0])
    let optimizer = jt.SGD(learning_rate: 0.1)
    optimizer.step([parameter])
    let updated = parameter.download()
    assert math.fabs(updated[0] - (1.0 - 0.1 * 0.5)) < 1e-5
    assert math.fabs(updated[1] - (-1.0 - 0.1 * 2.0)) < 1e-5
    parameter.free()
}

#: A learning rate that changes between steps has to reach the device.
#:
#: The settings buffer was written when a parameter first saw an optimizer
#: and never again, so the card kept reading the opening rate: a schedule
#: decaying 1.0, 0.1, 0.01, 0.001 moved the parameter by a whole 1.0 four
#: times over. Nothing complains, because a loss that is falling too fast is
#: still falling.
fn test_a_changing_learning_rate_reaches_the_device() -> void {
    if not jt.is_available() { return }
    let parameter = jt.Tensor([1], [0.0], requires_grad: true)
    var expected = 0.0
    for rate in [1.0, 0.1, 0.01, 0.001] {
        let gradient = jt.Tensor([1], [1.0])
        parameter.grad = gradient
        jt.SGD(learning_rate: rate).step([parameter])
        expected -= rate
        assert math.fabs(parameter.download()[0] - expected) < 1e-6
        gradient.free()
    }
    parameter.free()
}

#: The same parameter handed to a different kind of optimizer.
#:
#: Plain descent keeps three numbers in its settings and Adam six, so a
#: buffer sized for the first is too short for the second and Adam would read
#: past the end of it.
fn test_switching_optimizer_kind_resizes_the_settings() -> void {
    if not jt.is_available() { return }
    let parameter = jt.Tensor([1], [0.0], requires_grad: true)
    let first = jt.Tensor([1], [1.0])
    parameter.grad = first
    jt.SGD(learning_rate: 0.1).step([parameter])
    first.free()
    let second = jt.Tensor([1], [1.0])
    parameter.grad = second
    jt.Adam(learning_rate: 0.1).step([parameter])
    let landed = parameter.download()[0]
    #: Adam's first step is the learning rate, near enough, whatever came
    #: before it.
    assert math.is_finite(landed)
    assert math.fabs(landed - (-0.1 - 0.1)) < 1e-3
    second.free()
    parameter.free()
}

fn test_adam_step_moments_persist_and_affect_a_second_step() -> void {
    if not jt.is_available() { return }
    let parameter = jt.Tensor([1], [0.0], requires_grad: true)
    let beta1 = 0.5
    let beta2 = 0.5
    let epsilon = 1e-7
    let optimizer = jt.Adam(learning_rate: 0.001, beta1: beta1, beta2: beta2, epsilon: epsilon)

    let warmup_grad = jt.Tensor([1], [1.0])
    parameter.grad = warmup_grad
    for _index in 0..40 { optimizer.step([parameter]) }
    warmup_grad.free()
    let value_before = parameter.download()[0]
    let converged_first_moment = 1.0
    let converged_second_moment = 1.0

    let next_grad = jt.Tensor([1], [3.0])
    parameter.grad = next_grad
    optimizer.step([parameter])
    let first_moment_next = beta1 * converged_first_moment + (1.0 - beta1) * 3.0
    let second_moment_next = beta2 * converged_second_moment + (1.0 - beta2) * 3.0 * 3.0
    let expected_value = value_before -
        0.001 * first_moment_next / (math.sqrt(second_moment_next) + epsilon)
    assert math.fabs(parameter.download()[0] - expected_value) < 1e-6

    next_grad.free()
}

fn test_adam_step_matches_the_textbook_first_step_update_on_one_parameter() -> void {
    if not jt.is_available() { return }
    let parameter = jt.Tensor([1], [0.0], requires_grad: true)
    parameter.grad = jt.Tensor([1], [1.0])
    let optimizer = jt.Adam(learning_rate: 0.001, beta1: 0.9, beta2: 0.999, epsilon: 1e-7)
    optimizer.step([parameter])
    let first_moment = 0.1
    let second_moment = 0.001
    let corrected_first = first_moment / (1.0 - 0.9)
    let corrected_second = second_moment / (1.0 - 0.999)
    let expected = 0.0 - 0.001 * corrected_first / (math.sqrt(corrected_second) + 1e-7)
    assert math.fabs(parameter.download()[0] - expected) < 1e-6
    parameter.free()
}

fn test_optimizer_step_requires_every_parameter_to_have_a_gradient() -> void {
    if not jt.is_available() { return }
    let parameter = jt.Tensor([1], [1.0], requires_grad: true)
    let optimizer = jt.SGD()
    let _error = assert_throws(RuntimeError, fn() -> any {
        optimizer.step([parameter])
        return null
    })
    parameter.free()
}

fn test_optimizer_step_state_does_not_leak_to_a_new_parameter_reusing_a_freed_handle() -> void {
    if not jt.is_available() { return }
    let optimizer = jt.SGD(learning_rate: 0.1, momentum: 0.9)

    let stale_parameter = jt.Tensor([1], [5.0], requires_grad: true)
    stale_parameter.grad = jt.Tensor([1], [2.0])
    optimizer.step([stale_parameter])
    stale_parameter.free()

    let fresh_parameter = jt.Tensor([1], [1.0], requires_grad: true)
    fresh_parameter.grad = jt.Tensor([1], [0.5])
    optimizer.step([fresh_parameter])
    let expected = 1.0 - 0.1 * 0.5
    assert math.fabs(fresh_parameter.download()[0] - expected) < 1e-5

    fresh_parameter.free()
}

fn test_optimizer_momentum_starts_at_zero_on_a_recycled_device_buffer() -> void {
    if not jt.is_available() { return }
    #: A freed buffer goes back to a pool for the next allocation of the same
    #: size, so a new parameter's momentum is very likely to land on memory a
    #: previous parameter's momentum used. Zeroing it is a host write, and the
    #: update that filled it may be encoded and not yet run -- so without a
    #: wait, the old update lands after the zeroing and the new parameter
    #: starts with someone else's velocity.
    #:
    #: The churn is what makes this deterministic: it leaves buffers of the
    #: right size in the pool before the parameter under test asks for one.
    for _round in 0..8 {
        let scratch = jt.Tensor([1], [3.0], requires_grad: true)
        scratch.grad = jt.Tensor([1], [1.0])
        jt.SGD(learning_rate: 0.1, momentum: 0.9).step([scratch])
        scratch.free()
    }

    let optimizer = jt.SGD(learning_rate: 0.1, momentum: 0.9)
    let parameter = jt.Tensor([1], [1.0], requires_grad: true)
    parameter.grad = jt.Tensor([1], [0.5])
    optimizer.step([parameter])
    #: One step from rest moves by the learning rate times the gradient, with
    #: no velocity term yet.
    assert math.fabs(parameter.download()[0] - (1.0 - 0.1 * 0.5)) < 1e-5
    parameter.free()
}

fn test_optimizer_configuration_rejects_invalid_values() -> void {
    let _rate = assert_throws(ValueError, || jt.SGD(learning_rate: 0.0))
    let _momentum = assert_throws(ValueError, || jt.SGD(momentum: 1.0))
    let _nesterov = assert_throws(ValueError, || jt.SGD(nesterov: true))
    let _beta = assert_throws(ValueError, || jt.Adam(beta1: 1.0))
    let _epsilon = assert_throws(ValueError, || jt.Adam(epsilon: 0.0))
}

fn test_history_plot_uses_the_jaiplot_package_dependency() -> void {
    let history = jt.History()
    history.loss.extend([1.2, 0.7, 0.4])
    history.accuracy.extend([42.0, 71.0, 89.0])
    history.val_loss.extend([1.3, 0.8, 0.5])
    history.val_accuracy.extend([39.0, 68.0, 85.0])
    history.test_loss = 0.45
    history.test_accuracy = 87.0
    let path = Path(os.temp_dir()).join("jaithon-jaitensor-history-test.svg")
    defer { if path.exists() { path.remove() } }
    history.plot(path.to_str())
    let document = read_file(path.to_str())
    assert_true(document.starts_with("<svg"))
    assert_true(document.contains("validation"))
    assert_true(document.contains("test"))
    assert_eq(history.plot_path, path.to_str())
}

fn test_fused_two_layer_relu_sgd_reduces_loss() -> void {
    if not jt.is_available() { return }
    let data = separable_data(8)
    defer { data.free() }
    let model = jt.Sequential(
        [
            jt.Dense(512, activation: jt.Activation.Relu),
            jt.Dense(2, activation: jt.Activation.Softmax),
        ]
    )
    defer { model.free() }
    model.compile(jt.SGD(learning_rate: 0.2), loss: jt.SparseCategoricalCrossentropy())
    let history = model.fit(data, epochs: 6, batch_size: 32, verbose: false, plot_path: null)
    assert history.loss[history.loss.len() - 1] < history.loss[0]
}

fn test_fused_mlp_step_with_momentum_reduces_loss() -> void {
    if not jt.is_available() { return }
    let data = separable_data(8)
    defer { data.free() }
    let model = jt.Sequential(
        [
            jt.Dense(512, activation: jt.Activation.Relu),
            jt.Dense(2, activation: jt.Activation.Softmax),
        ]
    )
    defer { model.free() }
    model.compile(
        jt.SGD(learning_rate: 0.2, momentum: 0.5),
        loss: jt.SparseCategoricalCrossentropy()
    )
    let history = model.fit(data, epochs: 6, batch_size: 32, verbose: false, plot_path: null)
    assert history.loss[history.loss.len() - 1] < history.loss[0]
}

fn test_fused_three_layer_relu_sgd_reduces_loss() -> void {
    if not jt.is_available() { return }
    let data = separable_data(8)
    defer { data.free() }
    let model = jt.Sequential(
        [
            jt.Dense(256, activation: jt.Activation.Relu),
            jt.Dense(256, activation: jt.Activation.Relu),
            jt.Dense(256, activation: jt.Activation.Relu),
            jt.Dense(2, activation: jt.Activation.Softmax),
        ]
    )
    defer { model.free() }
    model.compile(jt.SGD(learning_rate: 0.2), loss: jt.SparseCategoricalCrossentropy())
    let history = model.fit(data, epochs: 10, batch_size: 32, verbose: false, plot_path: null)
    assert history.loss[history.loss.len() - 1] < history.loss[0]
}

fn test_fused_mlp_step_adam_reduces_loss() -> void {
    if not jt.is_available() { return }
    let data = separable_data(8)
    defer { data.free() }
    let model = jt.Sequential(
        [
            jt.Dense(512, activation: jt.Activation.Relu),
            jt.Dense(2, activation: jt.Activation.Softmax),
        ]
    )
    defer { model.free() }
    model.compile(jt.Adam(learning_rate: 0.05), loss: jt.SparseCategoricalCrossentropy())
    let history = model.fit(data, epochs: 6, batch_size: 32, verbose: false, plot_path: null)
    assert history.loss[history.loss.len() - 1] < history.loss[0]
}

fn test_execution_plan_built_on_compile() -> void {
    if not jt.is_available() { return }
    let model = jt.Sequential(
        [
            jt.Dense(16, activation: jt.Activation.Relu),
            jt.Dense(4, activation: jt.Activation.Softmax),
        ]
    )
    defer { model.free() }
    model.compile(jt.SGD(), input_units: 2, batch_size: 32)
    let plan = model.execution_plan()
    assert plan is not null
    assert_eq(plan.input_units, 2)
    assert_eq(plan.batch_size, 32)
    assert_eq(plan.layer_widths, [16, 4])
    assert plan.static_step
}

fn test_static_step_reduces_loss() -> void {
    if not jt.is_available() { return }
    let data = separable_data(8)
    defer { data.free() }
    let model = jt.Sequential(
        [
            jt.Dense(32, activation: jt.Activation.Relu),
            jt.Dense(2, activation: jt.Activation.Softmax),
        ]
    )
    defer { model.free() }
    model.compile(
        jt.SGD(learning_rate: 0.2),
        loss: jt.SparseCategoricalCrossentropy(),
        input_units: 2,
        batch_size: 32
    )
    let history = model.fit(data, epochs: 8, batch_size: 32, verbose: false, plot_path: null)
    assert history.loss[history.loss.len() - 1] < history.loss[0]
}

fn test_static_step_matches_autograd_weights() -> void {
    if not jt.is_available() { return }
    let data = separable_data(8)
    defer { data.free() }
    let static_model = jt.Sequential(
        [
            jt.Dense(16, activation: jt.Activation.Relu),
            jt.Dense(2, activation: jt.Activation.Softmax),
        ]
    )
    defer { static_model.free() }
    let autograd_model = jt.Sequential(
        [
            jt.Dense(16, activation: jt.Activation.Relu),
            jt.Dense(2, activation: jt.Activation.Softmax),
        ]
    )
    defer { autograd_model.free() }
    static_model.build(2)
    autograd_model.build(2)
    let (weights, bias) = static_model.layers[0].get_weights()
    let (head_weights, head_bias) = static_model.layers[1].get_weights()
    autograd_model.layers[0].set_weights(weights, bias)
    autograd_model.layers[1].set_weights(head_weights, head_bias)
    static_model.compile(
        jt.SGD(learning_rate: 0.1),
        loss: jt.SparseCategoricalCrossentropy(),
        input_units: 2,
        batch_size: 32
    )
    autograd_model.compile(jt.SGD(learning_rate: 0.1), loss: jt.SparseCategoricalCrossentropy())
    let static_history = static_model.fit(
        data,
        epochs: 3,
        batch_size: 32,
        verbose: false,
        plot_path: null
    )
    let autograd_history = autograd_model.fit(
        data,
        epochs: 3,
        batch_size: 32,
        verbose: false,
        plot_path: null
    )
    assert abs(static_history.loss[2] - autograd_history.loss[2]) < 0.05
}

fn test_static_kernels_are_traced() -> void {
    assert is_traced(dense_static_forward)
}

fn test_conv2d_matches_spatial_output_and_channel_count() -> void {
    if not jt.is_available() { return }
    let x = jt.Tensor(
        [1, 4, 4, 1],
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
        ]
    )
    let weights = jt.Tensor([2, 2, 1, 1], [1.0, 0.0, 0.0, 0.0])
    let y = jt.conv2d(x, weights, stride: 1, padding: 0)
    defer { x.free() }
    defer { weights.free() }
    defer { y.free() }
    assert_eq(y.shape, [1, 3, 3, 1])
    let values = y.download()
    assert_near(values[0], 1.0)
    assert_near(values[1], 2.0)
    assert_near(values[3], 5.0)
}

fn test_layer_norm_centers_each_row() -> void {
    if not jt.is_available() { return }
    let x = jt.Tensor([2, 4], [1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0])
    let gamma = jt.ones([4])
    let beta = jt.zeros([4])
    let y = jt.layer_norm(x, gamma, beta)
    defer { x.free() }
    defer { gamma.free() }
    defer { beta.free() }
    defer { y.free() }
    let values = y.download()
    var mean = 0.0
    for index in 0..4 { mean += values[index] }
    assert_near(mean / 4.0, 0.0, 1e-4)
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

#: Two epochs of the same run have to see the samples in different orders.
#:
#: Each of the three epoch bodies built its own loader with the default seed,
#: so every epoch of a run drew the same permutation -- the samples were
#: reordered once when training began and then frozen in that order, which is
#: most of what shuffling is for. Nothing observable went wrong; the model
#: simply saw the same batches over and over.
fn test_a_loader_per_epoch_orders_the_samples_differently() -> void {
    if not jt.is_available() { return }
    var features: list[float] = []
    var labels: list[int] = []
    for index in 0..24 {
        features.push(float(index))
        labels.push(index)
    }
    let data = jt.Dataset.from_flat(features, labels, 1)
    defer { data.free() }
    var seen: list[str] = []
    for epoch in 1..4 {
        let loader = jt.DataLoader(data, 6, shuffle: true, seed: epoch)
        let (batch_features, batch_labels) = loader.next_batch()
        var order = ""
        for value in batch_labels.download() { order += f"{int(value)}," }
        assert not seen.contains(order)
        seen.push(order)
        batch_features.free()
        batch_labels.free()
        #: And the order tensor goes back, which is the other half of what
        #: building one of these per epoch used to cost.
        loader.free()
        loader.free()
    }
}

#: An embedding fed a batch of index rows, which is how a model in this
#: library is trained.
#:
#: The layer only ever accepted a flat list of indices, so putting one in a
#: `Sequential` and calling `fit` threw on the first forward: a batch of four
#: contexts arrived as `[4, context]` and the gather refused anything but
#: rank one. There was no way to use the layer in a batched model at all.
fn test_an_embedding_takes_a_batch_of_index_rows() -> void {
    if not jt.is_available() { return }
    let vocab = 11
    let context = 3
    let width = 5
    var indices: list[float] = []
    var labels: list[int] = []
    var seed = 5
    for _sample in 0..32 {
        for _slot in 0..context {
            seed = (seed * 1103515245 + 12345) % 2147483648
            indices.push(float(seed % vocab))
        }
        labels.push(seed % vocab)
    }
    let data = jt.Dataset(
        jt.Tensor([32, context], indices),
        jt.Tensor([32], [float(value) for value in labels])
    )
    defer { data.free() }
    let embedding = jt.Embedding(vocab, width)
    let model = jt.Sequential(
        [
            embedding,
            jt.Dense(16, activation: jt.Activation.Relu),
            jt.Dense(vocab, activation: jt.Activation.Softmax),
        ]
    )
    defer { model.free() }
    model.compile(optimizer: jt.Adam(learning_rate: 0.02), loss: jt.SparseCategoricalCrossentropy())
    let history = model.fit(data, epochs: 4, batch_size: 8, verbose: false, plot_path: null)
    #: One row of indices becomes `context` lookups laid end to end, so the
    #: layer after it sees that much width. The layer learns this when it is
    #: built, which is the first time the model is fitted.
    assert_eq(embedding.output_width(), context * width)
    assert_eq(history.loss.len(), 4)
    for value in history.loss { assert math.is_finite(value) }
    assert history.loss[3] < history.loss[0]
}

fn test_dataloader_shuffle_covers_every_sample() -> void {
    if not jt.is_available() { return }
    let data = jt.Dataset.from_flat([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [0, 1, 2], 2)
    defer { data.free() }
    let loader = jt.DataLoader(data, 2, shuffle: true, seed: 3)
    var seen = 0
    while loader.has_next() {
        let (features, labels) = loader.next_batch()
        seen += labels.size
        features.free()
        labels.free()
    }
    assert_eq(seen, 3)
}

fn test_dataloader_shuffle_keeps_labels_with_their_rows() -> void {
    if not jt.is_available() { return }
    #: Row `i` carries its own answer in its first column, so a batch that
    #: pairs a row with somebody else's label cannot go unnoticed.
    let data = jt.Dataset.from_flat(
        [10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0],
        [1, 2, 3, 4],
        2
    )
    defer { data.free() }
    let loader = jt.DataLoader(data, 2, shuffle: true, seed: 3)
    var seen = 0
    while loader.has_next() {
        let (features, labels) = loader.next_batch()
        let rows = features.download()
        let answers = labels.download()
        for index in 0..answers.len() {
            assert_near(
                answers[index],
                rows[index * 2] / 10.0,
                1e-6,
                f"row {index} was handed the wrong label"
            )
            seen += 1
        }
        features.free()
        labels.free()
    }
    assert_eq(seen, 4)
}

fn test_dataloader_shuffle_keeps_images_shaped() -> void {
    if not jt.is_available() { return }
    var pixels: list[float] = []
    var answers: list[float] = []
    for sample in 0..4 {
        for _slot in 0..2 * 2 * 3 { pixels.push(float(sample)) }
        answers.push(float(sample))
    }
    let data = jt.Dataset(jt.Tensor([4, 2, 2, 3], pixels), jt.Tensor([4], answers))
    defer { data.free() }
    let loader = jt.DataLoader(data, 2, shuffle: true, seed: 7)
    while loader.has_next() {
        let (features, labels) = loader.next_batch()
        assert_eq(features.shape, [2, 2, 2, 3], "a shuffled image batch lost its shape")
        let rows = features.download()
        let picked = labels.download()
        for index in 0..picked.len() {
            assert_near(
                picked[index],
                rows[index * 12],
                1e-6,
                f"image {index} was handed the wrong label"
            )
        }
        features.free()
        labels.free()
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

fn test_conv2d_layer_builds_from_image_shape() -> void {
    if not jt.is_available() { return }
    var pixels: list[float] = []
    var labels: list[int] = []
    for sample in 0..4 {
        for _pixel in 0..16 { pixels.push(float(sample)) }
        labels.push(sample % 2)
    }
    let features = jt.Tensor([4, 4, 4, 1], pixels)
    let data = jt.Dataset(features, jt.Tensor([4], [float(label) for label in labels]))
    defer { data.free() }
    let model = jt.Sequential(
        [
            jt.Conv2d(2, 3, padding: 1, in_channels: 1),
            jt.Flatten(),
            jt.Dense(2, activation: jt.Activation.Softmax),
        ]
    )
    defer { model.free() }
    model.compile(jt.SGD(learning_rate: 0.05))
    let history = model.fit(data, epochs: 1, batch_size: 2, verbose: false, plot_path: null)
    assert_true(history.loss.len() == 1)
}

fn test_pooling_reports_its_pooled_shape_to_the_next_layer() -> void {
    if not jt.is_available() { return }
    #: `Sequential` sizes each layer from the shape the one before it reports.
    #: A pool that reported its input shape sized the Dense head as if nothing
    #: had been pooled, and the model failed on its first forward with a matmul
    #: shape error instead of at build time.
    var pixels: list[float] = []
    var labels: list[float] = []
    for sample in 0..4 {
        for _pixel in 0..64 { pixels.push(float(sample) * 0.1) }
        labels.push(float(sample % 2))
    }
    let features = jt.Tensor([4, 8, 8, 1], pixels)
    let data = jt.Dataset(features, jt.Tensor([4], labels))
    defer { data.free() }
    let model = jt.Sequential(
        [
            jt.Conv2d(2, 3, padding: 1, in_channels: 1),
            jt.MaxPool2d(2),
            jt.Flatten(),
            jt.Dense(2, activation: jt.Activation.Softmax),
        ]
    )
    defer { model.free() }
    model.compile(jt.SGD(learning_rate: 0.05))
    let history = model.fit(data, epochs: 1, batch_size: 2, verbose: false, plot_path: null)
    assert_true(history.loss.len() == 1)
}

fn test_average_pooling_reports_its_pooled_shape_too() -> void {
    if not jt.is_available() { return }
    var pixels: list[float] = []
    var labels: list[float] = []
    for sample in 0..4 {
        for _pixel in 0..64 { pixels.push(float(sample) * 0.1) }
        labels.push(float(sample % 2))
    }
    let features = jt.Tensor([4, 8, 8, 1], pixels)
    let data = jt.Dataset(features, jt.Tensor([4], labels))
    defer { data.free() }
    let model = jt.Sequential(
        [
            jt.Conv2d(2, 3, padding: 1, in_channels: 1),
            jt.AvgPool2d(2),
            jt.Flatten(),
            jt.Dense(2, activation: jt.Activation.Softmax),
        ]
    )
    defer { model.free() }
    model.compile(jt.SGD(learning_rate: 0.05))
    let history = model.fit(data, epochs: 1, batch_size: 2, verbose: false, plot_path: null)
    assert_true(history.loss.len() == 1)
}

fn test_conv2d_on_a_batch_slice_reads_that_slice() -> void {
    if not jt.is_available() { return }
    #: An `MPSNDArray` pads its innermost dimension to a sixteen-byte row, so a
    #: windowed feed misreads a packed buffer whenever that dimension is not a
    #: multiple of four floats. Three-channel NHWC images are exactly that
    #: shape, and every batch after the first convolved the wrong pixels — a
    #: model that trained on batch zero forever and looked merely untrainable.
    #: Two channels here so the row stays unaligned; the copy is the reference.
    #: Every operand has to clear the 512-byte floor the native path declines
    #: below, or the fallback runs and the window is never exercised.
    let batch = 16
    let side = 8
    let channels = 2
    let stride = side * side * channels
    var values: list[float] = []
    for index in 0..3 * batch * stride {
        values.push(float(index * 7 % 13) * 0.05 - 0.3)
    }
    let all = jt.Tensor([3 * batch, side, side, channels], values)
    defer { all.free() }
    var weight_values: list[float] = []
    for index in 0..3 * 3 * channels * 8 {
        weight_values.push(float(index * 5 % 9) * 0.05 - 0.2)
    }
    let weights = jt.Tensor([3, 3, channels, 8], weight_values)
    defer { weights.free() }
    for slice in 1..3 {
        let window = all.view(slice * batch * stride, [batch, side, side, channels])
        let copied = jt.Tensor([batch, side, side, channels], window.download())
        let from_window = jt.conv2d(window, weights, padding: 1).download()
        let from_copy = jt.conv2d(copied, weights, padding: 1).download()
        for index in 0..from_copy.len() {
            assert_near(from_window[index], from_copy[index], 1e-4)
        }
        window.free()
        copied.free()
    }
}

fn test_a_convolution_actually_trains() -> void {
    if not jt.is_available() { return }
    #: Conv2d recorded no gradient at all until its backward pass existed, and
    #: `Flatten` handed the head a view with no creator, so the graph stopped
    #: at the first dense layer. Both are invisible from the outside: the model
    #: runs, the loss moves a little as the head fits, and the filters never
    #: change. This asserts the filters move.
    let batch = 16
    var pixels: list[float] = []
    var labels: list[float] = []
    for sample in 0..batch {
        let kind = sample % 2
        for row in 0..8 {
            for column in 0..8 {
                pixels.push(if kind == 0 { float(row) * 0.1 } else { float(column) * 0.1 })
            }
        }
        labels.push(float(kind))
    }
    let data = jt.Dataset(jt.Tensor([batch, 8, 8, 1], pixels), jt.Tensor([batch], labels))
    defer { data.free() }
    let conv = jt.Conv2d(4, 3, padding: 1, in_channels: 1, activation: jt.Activation.Relu)
    let model = jt.Sequential(
        [
            conv,
            jt.MaxPool2d(2),
            jt.Flatten(),
            jt.Dense(2, activation: jt.Activation.Softmax),
        ]
    )
    defer { model.free() }
    model.compile(jt.SGD(learning_rate: 0.2))
    model.fit(data, epochs: 1, batch_size: 8, verbose: false, plot_path: null)
    let before = conv.parameters()[0].download()
    let history = model.fit(data, epochs: 6, batch_size: 8, verbose: false, plot_path: null)
    let after = conv.parameters()[0].download()
    var moved = 0.0
    for index in 0..before.len() {
        let step = if after[index] > before[index] {
            after[index] - before[index]
        } else {
            before[index] - after[index]
        }
        if step > moved { moved = step }
    }
    assert_true(moved > 1e-4)
    assert_true(history.loss[5] < history.loss[0])
}

fn conv_drift_after(samples: int, batch_size: int) -> float {
    var pixels: list[float] = []
    var labels: list[float] = []
    for _sample in 0..samples {
        for index in 0..64 { pixels.push(float(index * 7 % 11) * 0.1 - 0.5) }
        labels.push(0.0)
    }
    let data = jt.Dataset(jt.Tensor([samples, 8, 8, 1], pixels), jt.Tensor([samples], labels))
    defer { data.free() }
    let conv = jt.Conv2d(4, 3, padding: 1, in_channels: 1, activation: jt.Activation.Relu)
    let model = jt.Sequential([conv, jt.Flatten(), jt.Dense(2, activation: jt.Activation.Softmax)])
    defer { model.free() }
    #: Small enough that the parameters barely move, so every batch sees the
    #: same gradient and the totals are comparable.
    model.compile(jt.SGD(learning_rate: 1e-6))
    #: One whole-batch pass first, so the layers are built and both variants
    #: start from the same weights.
    model.fit(data, epochs: 1, batch_size: samples, verbose: false, plot_path: null)
    let before = conv.parameters()[0].download()
    model.fit(data, epochs: 1, batch_size: batch_size, verbose: false, plot_path: null)
    let after = conv.parameters()[0].download()
    var total = 0.0
    for index in 0..before.len() {
        let step = after[index] - before[index]
        total += step * step
    }
    return math.sqrt(total)
}

fn test_gradients_do_not_carry_between_batches() -> void {
    if not jt.is_available() { return }
    #: `Optimizer.step` reads `.grad` and leaves it alone; the training loop is
    #: what resets it. Without that reset, an op whose backward allocates a
    #: fresh gradient summed its batches, so four identical batches moved the
    #: filters by one, two, three and four gradients instead of one each —
    #: ten steps of drift against four. The two dense ops that write their
    #: gradient in place were immune, so every existing model hid it.
    let one_batch = conv_drift_after(4, 4)
    let four_batches = conv_drift_after(4, 1)
    assert_true(one_batch > 0.0)
    let ratio = four_batches / one_batch
    assert_true(ratio > 3.5)
    assert_true(ratio < 4.5)
}

fn test_device_count_is_non_negative() -> void {
    if not jt.is_available() { return }
    assert_true(jt.device_count() >= 1)
}

fn shaped_identity(x: Tensor[float, 2, 3]) -> Tensor[float, 2, 3] {
    return x
}

fn test_shaped_tensor_annotation_accepts_matching_runtime_value() -> void {
    if not jt.is_available() { return }
    let tensor = jt.Tensor([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    defer { tensor.free() }
    let got = shaped_identity(tensor)
    assert_eq(got.size, 6)
}

fn test_unary_abs_neg_square_and_gelu() -> void {
    if not jt.is_available() { return }
    let x = jt.Tensor([4], [-2.0, -0.0, 0.5, 2.0])
    defer { x.free() }
    let a = jt.abs(x)
    let n = jt.neg(x)
    let s = jt.square(x)
    let g = jt.gelu(jt.Tensor([1], [0.0]))
    defer { a.free() }
    defer { n.free() }
    defer { s.free() }
    defer { g.free() }
    assert_values_near(a.download(), [2.0, 0.0, 0.5, 2.0], 1e-5)
    assert_values_near(n.download(), [2.0, 0.0, -0.5, -2.0], 1e-5)
    assert_values_near(s.download(), [4.0, 0.0, 0.25, 4.0], 1e-5)
    assert_values_near(g.download(), [0.0], 1e-5)
}

fn test_sum_mean_clamp_and_cat() -> void {
    if not jt.is_available() { return }
    let x = jt.Tensor([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    defer { x.free() }
    let all = jt.sum(x)
    let rows = jt.sum(x, axis: -1)
    let avg = jt.mean(x)
    let clipped = jt.clamp(x, 2.0, 5.0)
    let stacked = jt.cat([x, x], axis: 0)
    defer { all.free() }
    defer { rows.free() }
    defer { avg.free() }
    defer { clipped.free() }
    defer { stacked.free() }
    assert_values_near(all.download(), [21.0], 1e-4)
    assert_values_near(rows.download(), [6.0, 15.0], 1e-4)
    assert_values_near(avg.download(), [3.5], 1e-4)
    assert_values_near(clipped.download(), [2.0, 2.0, 3.0, 4.0, 5.0, 5.0], 1e-5)
    assert_eq(stacked.shape, [4, 3])
}

fn test_max_pool2d_and_mse() -> void {
    if not jt.is_available() { return }
    let images = jt.Tensor([1, 2, 2, 1], [1.0, 2.0, 3.0, 4.0])
    defer { images.free() }
    let pooled = jt.max_pool2d(images, 2)
    defer { pooled.free() }
    assert_eq(pooled.shape, [1, 1, 1, 1])
    assert_values_near(pooled.download(), [4.0], 1e-5)
    let pred = jt.Tensor([2], [1.0, 3.0])
    let target = jt.Tensor([2], [1.0, 1.0])
    defer { pred.free() }
    defer { target.free() }
    let loss = jt.mse(pred, target)
    defer { loss.free() }
    assert_values_near(loss.download(), [2.0], 1e-4)
}

fn test_dropout_eval_is_identity_and_reshape_is_a_view() -> void {
    if not jt.is_available() { return }
    let x = jt.Tensor([3], [1.0, 2.0, 3.0])
    defer { x.free() }
    let kept = jt.dropout(x, 0.5, training: false)
    defer { kept.free() }
    assert_eq(kept.download(), [1.0, 2.0, 3.0])
    let viewed = jt.reshape(x, [1, 3])
    defer { viewed.free() }
    assert_eq(viewed.shape, [1, 3])
    assert_eq(viewed.download(), [1.0, 2.0, 3.0])
}

fn test_cat_last_axis_and_pad2d() -> void {
    if not jt.is_available() { return }
    let a = jt.Tensor([2, 2], [1.0, 2.0, 3.0, 4.0])
    let b = jt.Tensor([2, 1], [5.0, 6.0])
    defer { a.free() }
    defer { b.free() }
    let joined = jt.cat([a, b], axis: 1)
    defer { joined.free() }
    assert_eq(joined.shape, [2, 3])
    assert_values_near(joined.download(), [1.0, 2.0, 5.0, 3.0, 4.0, 6.0], 1e-5)
    let image = jt.Tensor([1, 1, 1, 1], [9.0])
    defer { image.free() }
    let padded = jt.pad2d(image, [1, 0, 0, 1])
    defer { padded.free() }
    assert_eq(padded.shape, [1, 2, 2, 1])
    assert_values_near(padded.download(), [0.0, 0.0, 9.0, 0.0], 1e-5)
}

fn test_where_squeeze_stack_and_pow() -> void {
    if not jt.is_available() { return }
    let cond = jt.Tensor([3], [1.0, 0.0, 1.0])
    let left = jt.Tensor([3], [2.0, 3.0, 4.0])
    let right = jt.Tensor([3], [8.0, 9.0, 10.0])
    defer { cond.free() }
    defer { left.free() }
    defer { right.free() }
    let picked = jt.where(cond, left, right)
    defer { picked.free() }
    assert_values_near(picked.download(), [2.0, 9.0, 4.0], 1e-5)
    let unit = jt.unsqueeze(left, 0)
    defer { unit.free() }
    assert_eq(unit.shape, [1, 3])
    let back = jt.squeeze(unit, axis: 0)
    defer { back.free() }
    assert_eq(back.shape, [3])
    let stacked = jt.stack([left, right])
    defer { stacked.free() }
    assert_eq(stacked.shape, [2, 3])
    let squared = jt.pow(left, 2.0)
    defer { squared.free() }
    assert_values_near(squared.download(), [4.0, 9.0, 16.0], 1e-4)
}

fn test_sequential_predict_skips_dropout() -> void {
    if not jt.is_available() { return }
    let data = separable_data(8)
    defer { data.free() }
    let model = jt.Sequential(
        [
            jt.Dense(4, activation: jt.Activation.Relu),
            jt.Dropout(0.9),
            jt.Dense(2, activation: jt.Activation.Softmax),
        ]
    )
    defer { model.free() }
    model.compile(jt.SGD(learning_rate: 0.05))
    model.fit(data, epochs: 1, batch_size: 8, verbose: false, plot_path: null)
    let batch = data.features.view(0, [8, 2])
    defer { batch.free() }
    let first = model.predict(batch)
    defer { first.free() }
    let second = model.predict(batch)
    defer { second.free() }
    assert_values_near(first.download(), second.download(), 1e-5)
}

fn test_ones_full_skip_host_lists_and_layouts() -> void {
    if not jt.is_available() { return }
    let ones = jt.ones([3])
    let filled = jt.fill([2], 7.0)
    defer { ones.free() }
    defer { filled.free() }
    assert_eq(ones.download(), [1.0, 1.0, 1.0])
    assert_eq(filled.download(), [7.0, 7.0])
    let nhwc = jt.Tensor([1, 2, 2, 1], [1.0, 2.0, 3.0, 4.0])
    defer { nhwc.free() }
    let nchw = jt.nchw(nhwc)
    defer { nchw.free() }
    assert_eq(nchw.shape, [1, 1, 2, 2])
    assert_values_near(nchw.download(), [1.0, 2.0, 3.0, 4.0], 1e-5)
    let back = jt.nhwc(nchw)
    defer { back.free() }
    assert_values_near(back.download(), [1.0, 2.0, 3.0, 4.0], 1e-5)
    let src = jt.Tensor([2, 1], [1.0, 2.0])
    defer { src.free() }
    let tiled = jt.repeat(src, 3)
    defer { tiled.free() }
    assert_eq(tiled.shape, [2, 3])
    assert_values_near(tiled.download(), [1.0, 1.0, 1.0, 2.0, 2.0, 2.0], 1e-5)
    let pred = jt.Tensor([2], [0.9, 0.1])
    let target = jt.Tensor([2], [1.0, 0.0])
    defer { pred.free() }
    defer { target.free() }
    let loss = jt.binary_cross_entropy(pred, target)
    defer { loss.free() }
    assert_true(loss.download()[0] > 0.0)
    assert_true(loss.download()[0] < 0.5)
}

fn test_eye_flip_upsample_and_split_heads() -> void {
    if not jt.is_available() { return }
    let ident = jt.eye(3)
    defer { ident.free() }
    assert_eq(ident.download(), [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    let src = jt.Tensor([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    defer { src.free() }
    let flipped = jt.flip(src)
    defer { flipped.free() }
    assert_eq(flipped.download(), [3.0, 2.0, 1.0, 6.0, 5.0, 4.0])
    let image = jt.Tensor([1, 1, 2, 1], [1.0, 2.0])
    defer { image.free() }
    let up = jt.upsample_nearest2d(image, 2)
    defer { up.free() }
    assert_eq(up.shape, [1, 2, 4, 1])
    assert_eq(up.download(), [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0])
    let packed_in = jt.Tensor([2, 4], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    defer { packed_in.free() }
    let split = jt.split_heads(packed_in, 2)
    defer { split.free() }
    assert_eq(split.shape, [4, 2])
    assert_eq(split.download(), [1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0])
    let merged = jt.merge_heads(split, 2, 2)
    defer { merged.free() }
    assert_eq(merged.download(), packed_in.download())
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

fn test_sum_mean_and_mse_backward_stay_on_device() -> void {
    if not jt.is_available() { return }
    let x = jt.Tensor([2, 2], [1.0, 2.0, 3.0, 4.0], requires_grad: true)
    defer { x.free() }
    let total = jt.sum(x)
    defer { total.free() }
    let extras = total.backward()
    for tensor in extras { tensor.free() }
    let sum_grad = x.grad
    if sum_grad is null { throw AssertionError("expected x.grad after sum.backward()") }
    assert_values_near(sum_grad.download(), [1.0, 1.0, 1.0, 1.0], 1e-5)

    let y = jt.Tensor([2, 2], [1.0, 2.0, 3.0, 4.0], requires_grad: true)
    defer { y.free() }
    let avg = jt.mean(y)
    defer { avg.free() }
    let mean_extras = avg.backward()
    for tensor in mean_extras { tensor.free() }
    let mean_grad = y.grad
    if mean_grad is null { throw AssertionError("expected y.grad after mean.backward()") }
    assert_values_near(mean_grad.download(), [0.25, 0.25, 0.25, 0.25], 1e-5)

    let z = jt.Tensor([2, 2], [1.0, 2.0, 3.0, 4.0], requires_grad: true)
    defer { z.free() }
    let rows = jt.sum(z, axis: -1)
    let row_total = jt.sum(rows)
    defer { row_total.free() }
    let row_extras = row_total.backward()
    for tensor in row_extras { tensor.free() }
    let row_grad = z.grad
    if row_grad is null { throw AssertionError("expected z.grad after last-axis sum") }
    assert_values_near(row_grad.download(), [1.0, 1.0, 1.0, 1.0], 1e-5)

    let pred = jt.Tensor([2], [1.0, 3.0], requires_grad: true)
    let target = jt.Tensor([2], [1.0, 1.0])
    defer { pred.free() }
    defer { target.free() }
    let loss = jt.mse(pred, target)
    defer { loss.free() }
    let mse_extras = loss.backward()
    for tensor in mse_extras { tensor.free() }
    let pred_grad = pred.grad
    if pred_grad is null { throw AssertionError("expected pred.grad after mse.backward()") }
    assert_values_near(pred_grad.download(), [0.0, 2.0], 1e-4)
}

fn test_gather_rows_scatters_embedding_grads() -> void {
    if not jt.is_available() { return }
    let table = jt.Tensor([3, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], requires_grad: true)
    defer { table.free() }
    let ids = jt.Tensor([3], [0.0, 2.0, 0.0])
    defer { ids.free() }
    let rows = jt.gather_rows(table, ids)
    let total = jt.sum(rows)
    defer { total.free() }
    let extras = total.backward()
    for tensor in extras { tensor.free() }
    let g = table.grad
    if g is null { throw AssertionError("expected table.grad after embedding gather") }
    assert_values_near(g.download(), [2.0, 2.0, 0.0, 0.0, 1.0, 1.0], 1e-5)
}

fn test_group_norm_centers_each_group() -> void {
    if not jt.is_available() { return }
    let x = jt.Tensor([1, 1, 2, 2], [1.0, 3.0, 3.0, 5.0])
    defer { x.free() }
    let gamma = jt.ones([2])
    let beta = jt.zeros([2])
    defer { gamma.free() }
    defer { beta.free() }
    let y = jt.group_norm(x, gamma, beta, 2)
    defer { y.free() }
    let values = y.download()
    assert_near((values[0] + values[2]) / 2.0, 0.0, 1e-4)
    assert_near((values[1] + values[3]) / 2.0, 0.0, 1e-4)
}

fn test_tril_and_bce_with_logits() -> void {
    if not jt.is_available() { return }
    let square = jt.Tensor([2, 2], [1.0, 2.0, 3.0, 4.0])
    defer { square.free() }
    let lower = jt.tril(square)
    defer { lower.free() }
    assert_eq(lower.download(), [1.0, 0.0, 3.0, 4.0])
    let logits = jt.Tensor([2], [0.0, 10.0], requires_grad: true)
    let target = jt.Tensor([2], [0.0, 1.0])
    defer { logits.free() }
    defer { target.free() }
    let loss = jt.binary_cross_entropy_with_logits(logits, target)
    defer { loss.free() }
    assert_near(loss.download()[0], math.log(2.0) / 2.0, 1e-4)
    let extras = loss.backward()
    for tensor in extras { tensor.free() }
    let g = logits.grad
    if g is null { throw AssertionError("expected logits.grad after bce_with_logits") }
    assert_near(g.download()[0], 0.25, 1e-4)
    assert_near(g.download()[1], 0.0, 1e-4)
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

#: `amax` is the companion to `argmax`: where `argmax` says which element of a
#: row is largest, `amax` says what it is. A detector wants both, and working
#: the value out on the host afterwards means downloading everything the
#: reduction was there to avoid.
fn test_amax_reduces_the_last_axis_and_the_whole_tensor() -> void {
    if not jt.is_available() { return }
    let values = [3.0, -1.0, 2.5, 0.0, -4.0, -0.5, 7.25, 1.0, 6.0, 6.0, -2.0, 5.5]
    let x = Tensor([3, 4], values)
    let along = jt.amax(x, axis: -1).download()
    assert_eq(along.len(), 3)
    assert_near(along[0], 3.0, 1e-6)
    assert_near(along[1], 7.25, 1e-6)
    assert_near(along[2], 6.0, 1e-6)

    #: A tie keeps the first, the same rule `argmax` follows.
    let where = jt.argmax(x).download()
    assert_eq(int(where[2]), 0)

    let whole = jt.amax(x).download()
    assert_eq(whole.len(), 1)
    assert_near(whole[0], 7.25, 1e-6)

    #: A single column is still a reduction, of one element each.
    let column = Tensor([4, 1], [1.0, 2.0, 3.0, 4.0])
    let each = jt.amax(column, axis: -1).download()
    assert_eq(each, [1.0, 2.0, 3.0, 4.0])

    let _refused = assert_throws(ValueError, || jt.amax(x, axis: 0))
    x.free()
    column.free()
}

#: `max_with_index` agrees with asking for the two separately.
fn test_max_with_index_matches_amax_and_argmax() -> void {
    if not jt.is_available() { return }
    var values: list[float] = []
    for index in 0..64 * 40 {
        #: Every row has its peak somewhere different, and negatives so that a
        #: zero-initialised answer would show.
        values.push(float(index * 37 % 79 - 40) * 0.25)
    }
    let input = Tensor([64, 40], values)
    defer { input.free() }
    let (peaks, wheres) = jt.max_with_index(input)
    defer { peaks.free() }
    defer { wheres.free() }
    let apart_values = jt.amax(input, axis: -1)
    defer { apart_values.free() }
    let apart_indices = jt.argmax(input)
    defer { apart_indices.free() }
    assert_eq(peaks.download(), apart_values.download())
    assert_eq(wheres.download(), apart_indices.download())
}

fn test_max_with_index_refuses_anything_but_rank_two() -> void {
    if not jt.is_available() { return }
    let flat = Tensor([8])
    defer { flat.free() }
    assert_throws(ValueError, || jt.max_with_index(flat))
}

#: Reducing down the columns gives what transposing and reducing along the rows
#: gives, which is the pass it exists to avoid.
fn test_max_with_index_down_the_columns_matches_a_transpose() -> void {
    if not jt.is_available() { return }
    var values: list[float] = []
    for index in 0..24 * 57 { values.push(float(index * 41 % 91 - 45) * 0.125) }
    let input = Tensor([24, 57], values)
    defer { input.free() }
    let (peaks, wheres) = jt.max_with_index(input, axis: 0)
    defer { peaks.free() }
    defer { wheres.free() }
    let flipped = jt.transpose(input)
    defer { flipped.free() }
    let (want_peaks, want_wheres) = jt.max_with_index(flipped, axis: -1)
    defer { want_peaks.free() }
    defer { want_wheres.free() }
    assert_eq(peaks.shape, [57])
    assert_eq(peaks.download(), want_peaks.download())
    assert_eq(wheres.download(), want_wheres.download())
}

fn test_max_with_index_refuses_an_axis_it_has_no_kernel_for() -> void {
    if not jt.is_available() { return }
    let input = Tensor([4, 5])
    defer { input.free() }
    assert_throws(ValueError, || jt.max_with_index(input, axis: 2))
}

#: A row wider than the threadgroup, checked against arithmetic done here.
#:
#: The kernel spreads a row across 256 threads and brings the two sums back
#: through a simd reduction, so a row narrower than that never exercises the
#: striding or the second stage of the reduction. Widths either side of 256 and
#: of a warp are the ones that catch a reduction wired up wrongly.
fn test_layer_norm_matches_arithmetic_over_wide_rows() -> void {
    if not jt.is_available() { return }
    for width in [17, 32, 33, 256, 257, 768] {
        let rows = 3
        var values: list[float] = []
        for index in 0..rows * width {
            values.push(float(index * 31 % 97 - 48) * 0.125)
        }
        var scales: list[float] = []
        var shifts: list[float] = []
        for index in 0..width {
            scales.push(1.0 + float(index % 5) * 0.25)
            shifts.push(float(index % 7) * 0.5 - 1.5)
        }
        let x = jt.Tensor([rows, width], values)
        let gamma = jt.Tensor([width], scales)
        let beta = jt.Tensor([width], shifts)
        let y = jt.layer_norm(x, gamma, beta)
        let got = y.download()
        for row in 0..rows {
            var total = 0.0
            for column in 0..width { total += values[row * width + column] }
            let mean = total / float(width)
            var square = 0.0
            for column in 0..width {
                let gap = values[row * width + column] - mean
                square += gap * gap
            }
            let inverse = 1.0 / math.sqrt(square / float(width) + 1e-5)
            for column in 0..width {
                let centred = values[row * width + column] - mean
                let want = scales[column] * centred * inverse + shifts[column]
                assert_near(
                    got[row * width + column],
                    want,
                    1e-3,
                    f"width {width} at {row},{column}"
                )
            }
        }
        x.free()
        gamma.free()
        beta.free()
        y.free()
    }
}

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

#: A row wider than the threadgroup that reduces it.
#:
#: These reductions used to run one thread to a row, so a row of any width was
#: one thread's serial walk and the width never mattered. Now the group strides
#: the row, and a row of 1000 takes four passes with a partial per thread to
#: combine -- which is the case a row of four never reaches.
fn test_wide_rows_reduce_the_same_as_narrow_ones() -> void {
    if not jt.is_available() { return }
    let rows = 3
    let columns = 1000
    var values: list[float] = []
    for row in 0..rows {
        for column in 0..columns {
            values.push(float((row * 7 + column * 13) % 101) * 0.01)
        }
    }
    #: One unmistakable peak per row, past where a single group would stop.
    values[0 * columns + 700] = 9.5
    values[1 * columns + 999] = 9.5
    values[2 * columns + 0] = 9.5
    let x = jt.Tensor([rows, columns], values)
    defer { x.free() }

    let totals = jt.sum(x, axis: -1).download()
    let averages = jt.mean(x, axis: -1).download()
    let peaks = jt.amax(x, axis: -1).download()
    let where = jt.argmax(x).download()
    let (also_peaks, also_where) = jt.max_with_index(x)
    defer { also_peaks.free() }
    defer { also_where.free() }
    let peak_values = also_peaks.download()
    let peak_places = also_where.download()

    let wanted = [700.0, 999.0, 0.0]
    for row in 0..rows {
        var total = 0.0
        for column in 0..columns { total += values[row * columns + column] }
        assert_near(totals[row], total, total * 1e-5 + 1e-4, f"row {row} total")
        assert_near(averages[row], total / float(columns), 1e-5, f"row {row} mean")
        assert_near(peaks[row], 9.5, 1e-6, f"row {row} peak")
        assert_near(where[row], wanted[row], 1e-6, f"row {row} peak position")
        assert_near(peak_values[row], 9.5, 1e-6, f"row {row} peak, together")
        assert_near(peak_places[row], wanted[row], 1e-6, f"row {row} position, together")
    }
}

#: Every element equal: the first one wins, whichever thread happened to see it.
fn test_a_wide_row_of_ties_keeps_the_first() -> void {
    if not jt.is_available() { return }
    let columns = 700
    var values: list[float] = []
    for _column in 0..columns { values.push(2.5) }
    let x = jt.Tensor([1, columns], values)
    defer { x.free() }
    assert_near(jt.argmax(x).download()[0], 0.0, 1e-6, "a row of ties should keep the first")
    let (peak, place) = jt.max_with_index(x)
    defer { peak.free() }
    defer { place.free() }
    assert_near(peak.download()[0], 2.5, 1e-6)
    assert_near(place.download()[0], 0.0, 1e-6, "a row of ties should keep the first")
}

#: The four exports no test called directly.
#:
#: Three of them are reached through something else -- the activations through
#: `Dense`, the NCHW convolution through jaicv's imported graphs -- so the gap
#: was in this suite rather than in the code. `rsqrt` had no caller anywhere,
#: which is the one worth checking: an unused export is an unverified one.
fn test_the_exports_no_test_reached() -> void {
    if not jt.is_available() { return }
    let x = jt.Tensor([2, 3], [4.0, 1.0, 0.25, 16.0, 100.0, 0.0625])
    defer { x.free() }
    let roots = jt.rsqrt(x).download()
    let wanted = [0.5, 1.0, 2.0, 0.25, 0.1, 4.0]
    for index in 0..wanted.len() {
        assert_near(roots[index], wanted[index], 1e-5, f"rsqrt at {index}")
    }

    let mixed = jt.Tensor([2, 3], [-2.0, -0.5, 0.0, 0.5, 2.0, 3.0])
    defer { mixed.free() }
    let curved = jt.silu(mixed).download()
    for (index, value) in [-2.0, -0.5, 0.0, 0.5, 2.0, 3.0].enumerate() {
        #: x * sigmoid(x), which is what the layer applies.
        let want = value / (1.0 + math.exp(-value))
        assert_near(curved[index], want, 1e-5, f"silu at {index}")
    }
    #: A hundredth below zero, which is the slope the kernel is written with.
    let leaky = jt.leaky_relu(mixed).download()
    for (index, value) in [-2.0, -0.5, 0.0, 0.5, 2.0, 3.0].enumerate() {
        let want = if value > 0.0 { value } else { value * 0.01 }
        assert_near(leaky[index], want, 1e-5, f"leaky_relu at {index}")
    }

    #: One channel, one 2x2 kernel of ones over a 3x3 ramp: each output is the
    #: sum of the square it covers.
    let image = jt.Tensor([1, 1, 3, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    defer { image.free() }
    let weights = jt.Tensor([1, 1, 2, 2], [1.0, 1.0, 1.0, 1.0])
    defer { weights.free() }
    let convolved = jt.conv2d_nchw(image, weights)
    defer { convolved.free() }
    assert_eq(convolved.shape, [1, 1, 2, 2])
    let sums = convolved.download()
    for (index, want) in [12.0, 16.0, 24.0, 28.0].enumerate() {
        assert_near(sums[index], want, 1e-5, f"conv2d_nchw at {index}")
    }
}

#: Transpose at sizes that do not land on the tile it stages through.
#:
#: The kernel reads and writes 32 by 32 squares through threadgroup memory so
#: that neither pass strides. Every edge of that -- a matrix narrower than one
#: tile, one taller, one that ends part way through -- is a bounds test the
#: naive version never had.
fn test_transpose_handles_every_edge_of_its_tile() -> void {
    if not jt.is_available() { return }
    for shape in [[1, 1], [1, 37], [37, 1], [31, 33], [32, 32], [33, 31], [64, 65], [129, 257]] {
        let rows = shape[0]
        let cols = shape[1]
        var values: list[float] = []
        for index in 0..rows * cols { values.push(float(index) * 0.5 - 3.0) }
        let x = jt.Tensor([rows, cols], values)
        let flipped = jt.transpose(x)
        assert_eq(flipped.shape, [cols, rows], f"{rows}x{cols} came back the wrong shape")
        let got = flipped.download()
        for r in 0..rows {
            for c in 0..cols {
                assert_near(
                    got[c * rows + r],
                    values[r * cols + c],
                    1e-6,
                    f"{rows}x{cols} at {r},{c}"
                )
            }
        }
        x.free()
        flipped.free()
    }
}
