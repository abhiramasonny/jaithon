#: Backward passes against a central difference, and against arithmetic done here.
import jaitensor as jt
import std.gpu as gpu
import std.math
import std.os
from jaitensor.losses import sparse_ce_with_delta
from jaitensor.tensor import GradNode, Tensor
from std.test import assert_eq, assert_near, assert_true

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

fn assert_values_near(actual: list[float], expected: list[float], tolerance: float = 1e-5) -> void {
    assert_eq(actual.len(), expected.len())
    for (index, value) in actual.enumerate() {
        assert_near(value, expected[index], tolerance)
    }
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

#: The group-norm backward worked out on the host from the definition, with
#: nothing in common with the kernels but the shape.
#:
#: Returns the input gradient, then gamma's, then beta's.
fn group_norm_backward_by_hand(
    values: list[float],
    up: list[float],
    gamma: list[float],
    n: int,
    spatial: int,
    channels: int,
    groups: int
) -> tuple[list[float], list[float], list[float]] {
    let gs = channels // groups
    let members = spatial * gs
    var grad_input = [0.0 for _slot in 0..values.len()]
    var grad_gamma = [0.0 for _channel in 0..channels]
    var grad_beta = [0.0 for _channel in 0..channels]
    for sample in 0..n {
        for group in 0..groups {
            var total = 0.0
            for hw in 0..spatial {
                for k in 0..gs {
                    total += values[(sample * spatial + hw) * channels + group * gs + k]
                }
            }
            let centre = total / float(members)
            var spread = 0.0
            for hw in 0..spatial {
                for k in 0..gs {
                    let d = values[(sample * spatial + hw) * channels + group * gs + k] - centre
                    spread += d * d
                }
            }
            let inv = 1.0 / math.sqrt(spread / float(members) + 1e-5)
            var sum_dy = 0.0
            var sum_dy_xhat = 0.0
            for hw in 0..spatial {
                for k in 0..gs {
                    let index = (sample * spatial + hw) * channels + group * gs + k
                    let xhat = (values[index] - centre) * inv
                    let dy = up[index] * gamma[group * gs + k]
                    sum_dy += dy
                    sum_dy_xhat += dy * xhat
                }
            }
            for hw in 0..spatial {
                for k in 0..gs {
                    let index = (sample * spatial + hw) * channels + group * gs + k
                    let xhat = (values[index] - centre) * inv
                    let dy = up[index] * gamma[group * gs + k]
                    let pull = (sum_dy + xhat * sum_dy_xhat) / float(members)
                    grad_input[index] = inv * (dy - pull)
                    grad_gamma[group * gs + k] += up[index] * xhat
                    grad_beta[group * gs + k] += up[index]
                }
            }
        }
    }
    return (grad_input, grad_gamma, grad_beta)
}

#: Both arms of the group-norm backward against that hand computation.
#:
#: The reduction was rewritten as a blocked two-pass split that reads the
#: tensor once instead of twice, and it only applies when the channel count
#: divides a threadgroup and the feature map is long enough to cut up. The
#: sixteen-square shape takes the split -- four blocks a sample, so the join
#: has more than one partial to add -- and the four-square shape is too short
#: for it and takes the older kernels, so one call each covers both.
fn test_group_norm_backward_matches_a_hand_computation() -> void {
    if not jt.is_available() { return }
    let channels = 4
    let groups = 2
    var gamma_values: list[float] = []
    var beta_values: list[float] = []
    for index in 0..channels {
        gamma_values.push(0.7 + float(index) * 0.3)
        beta_values.push(float(index) * 0.1 - 0.2)
    }
    let gamma = jt.Tensor([channels], gamma_values)
    defer { gamma.free() }
    let beta = jt.Tensor([channels], beta_values)
    defer { beta.free() }
    for side in [16, 4] {
        let n = 3
        let spatial = side * side
        let count = n * spatial * channels
        var values: list[float] = []
        var up_values: list[float] = []
        for index in 0..count {
            values.push(float(index * 37 % 101) * 0.05 - 2.5)
            up_values.push(float(index * 53 % 61) * 0.03 - 0.9)
        }
        let x = jt.Tensor([n, side, side, channels], values, requires_grad: true)
        defer { x.free() }
        let up = jt.Tensor([n, side, side, channels], up_values)
        defer { up.free() }
        let y = jt.group_norm(x, gamma, beta, groups)
        defer { y.free() }
        let node = y.creator
        assert_true(node is not null)
        let grads = node.apply(up)
        let (want_input, want_gamma, want_beta) = group_norm_backward_by_hand(
            values,
            up_values,
            gamma_values,
            n,
            spatial,
            channels,
            groups
        )
        assert_values_near(grads[0].download(), want_input, 1e-5)
        #: The parameter gradients each sum three thousand terms that cancel
        #: heavily, so they carry more of the accumulation's error than the
        #: input gradient does: the worst of them lands 3e-6 out.
        assert_values_near(grads[1].download(), want_gamma, 1e-4)
        assert_values_near(grads[2].download(), want_beta, 1e-4)
        for grad in grads { grad.free() }
    }
}

fn free_graph(held: list[Tensor]) -> void {
    for tensor in held { tensor.free() }
}

fn clear_grad(tensor: Tensor) -> void {
    let gradient = tensor.grad
    if gradient is not null {
        gradient.free()
        tensor.grad = null
    }
}

#: `op` applied with the constant on the left (`swap` false) or the right.
fn binary_by_name(name: str, constant: Tensor, x: Tensor, swap: bool) -> Tensor {
    let a = if swap { x } else { constant }
    let b = if swap { constant } else { x }
    if name == "add" { return jt.add(a, b) }
    if name == "subtract" { return jt.subtract(a, b) }
    if name == "multiply" { return jt.multiply(a, b) }
    return jt.divide(a, b)
}

#: d/dx of sum(op(c, x)) or sum(op(x, c)), elementwise, done here.
fn binary_grad_by_hand(name: str, c: float, x: float, swap: bool) -> float {
    if name == "add" { return 1.0 }
    if name == "subtract" { return if swap { 1.0 } else { -1.0 } }
    if name == "multiply" { return c }
    return if swap { 1.0 / c } else { -c / (x * x) }
}

fn test_backward_gives_no_gradient_to_a_constant_on_either_side_of_a_binary_op() -> void {
    if not jt.is_available() { return }
    let x_values = [0.5, -1.25, 2.0, 1.5]
    let c_values = [2.0, 0.75, -1.5, 3.0]
    let x = jt.Tensor([4], x_values, requires_grad: true)
    defer { x.free() }
    let c = jt.Tensor([4], c_values)
    defer { c.free() }
    for name in ["add", "subtract", "multiply", "divide"] {
        for swap in [false, true] {
            gpu.synchronize()
            let before = gpu.live_buffers()
            for _step in 0..3 {
                let out = binary_by_name(name, c, x, swap)
                let loss = jt.sum(out)
                free_graph(loss.backward())
                loss.free()
                if c.grad is not null {
                    throw AssertionError(f"{name} (swap {swap}) gave the constant a gradient")
                }
                let gradient = x.grad
                if gradient is null { throw AssertionError(f"{name} (swap {swap}) gave x no gradient") }
                var want: list[float] = []
                for index in 0..4 {
                    want.push(binary_grad_by_hand(name, c_values[index], x_values[index], swap))
                }
                assert_values_near(gradient.download(), want, 1e-5)
                clear_grad(x)
            }
            gpu.synchronize()
            assert_eq(gpu.live_buffers(), before)
        }
    }
}

fn test_matmul_backward_gives_no_gradient_to_a_data_operand_on_either_side() -> void {
    if not jt.is_available() { return }
    let data = jt.Tensor([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    defer { data.free() }
    let right = jt.Tensor([3, 2], [0.5, -1.0, 2.0, 0.25, -0.5, 1.5], requires_grad: true)
    defer { right.free() }
    let left = jt.Tensor([3, 2], [1.0, -2.0, 0.5, 0.75, 3.0, -1.0], requires_grad: true)
    defer { left.free() }
    for _step in 0..3 {
        #: data @ right: d/d(right) of the sum is data^T @ ones, data's column sums.
        let product = jt.matmul(data, right)
        let loss = jt.sum(product)
        free_graph(loss.backward())
        loss.free()
        assert_true(data.grad is null)
        let gradient = right.grad
        if gradient is null { throw AssertionError("matmul gave the weight no gradient") }
        assert_values_near(gradient.download(), [5.0, 5.0, 7.0, 7.0, 9.0, 9.0], 1e-5)
        clear_grad(right)

        #: left^T-shaped use: left [3,2] times data [2,3]; d/d(left) of the sum
        #: is ones @ data^T, data's row sums in every row.
        let reversed = jt.matmul(left, data)
        let reversed_loss = jt.sum(reversed)
        free_graph(reversed_loss.backward())
        reversed_loss.free()
        assert_true(data.grad is null)
        let left_grad = left.grad
        if left_grad is null { throw AssertionError("matmul gave the left weight no gradient") }
        assert_values_near(left_grad.download(), [6.0, 15.0, 6.0, 15.0, 6.0, 15.0], 1e-5)
        clear_grad(left)
    }
}

#: The custom linear from probes5/eager/nograd_inputs.jai: neither the batch
#: nor the target asked for a gradient, and over three steps neither grows one,
#: while the weights' gradients come out as computed here.
fn test_custom_linear_leaves_the_batch_and_target_without_gradients() -> void {
    if not jt.is_available() { return }
    let rows = 4
    let inner = 3
    let cols = 2
    var x_values: list[float] = []
    for index in 0..rows * inner { x_values.push(float(index % 5) * 0.5 - 1.0) }
    var t_values: list[float] = []
    for index in 0..rows * cols { t_values.push(float(index % 3) - 1.0) }
    let x = jt.Tensor([rows, inner], x_values)
    defer { x.free() }
    let target = jt.Tensor([rows, cols], t_values)
    defer { target.free() }
    let w = jt.Tensor([inner, cols], [0.5, -0.25, 1.0, 0.75, -1.5, 0.125], requires_grad: true)
    defer { w.free() }
    let bias = jt.Tensor([cols], [0.1, -0.2], requires_grad: true)
    defer { bias.free() }
    gpu.synchronize()
    let before = gpu.live_buffers()
    for _step in 0..3 {
        let y = jt.add(jt.matmul(x, w), bias)
        let loss = jt.sum(jt.subtract(y, target))
        free_graph(loss.backward())
        loss.free()
        assert_true(x.grad is null)
        assert_true(target.grad is null)
        let w_grad = w.grad
        let b_grad = bias.grad
        if w_grad is null or b_grad is null { throw AssertionError("weights got no gradient") }
        #: d/dW sum(xW + b - t) = x^T @ ones; d/db = rows.
        var want_w: list[float] = []
        for k in 0..inner {
            var column = 0.0
            for r in 0..rows { column += x_values[r * inner + k] }
            for _c in 0..cols { want_w.push(column) }
        }
        assert_values_near(w_grad.download(), want_w, 1e-5)
        assert_values_near(b_grad.download(), [float(rows), float(rows)], 1e-5)
        clear_grad(w)
        clear_grad(bias)
    }
    gpu.synchronize()
    assert_eq(gpu.live_buffers(), before)
}

fn test_backward_releases_an_intermediate_gradient_unless_it_is_retained() -> void {
    if not jt.is_available() { return }
    if os.env_or("JAITENSOR_BACKWARD_FREE", "1") == "0" { return }
    let x = jt.Tensor([3], [1.0, -2.0, 0.5], requires_grad: true)
    defer { x.free() }
    let squared = jt.multiply(x, x)
    let loss = jt.sum(squared)
    let held = loss.backward()
    assert_true(squared.grad is null)
    let gradient = x.grad
    if gradient is null { throw AssertionError("expected x.grad") }
    assert_values_near(gradient.download(), [2.0, -4.0, 1.0], 1e-5)
    free_graph(held)
    loss.free()
    clear_grad(x)

    let kept = jt.multiply(x, x).retain_grad()
    let kept_loss = jt.sum(jt.scale(kept, 3.0))
    let kept_held = kept_loss.backward()
    let kept_grad = kept.grad
    if kept_grad is null { throw AssertionError("a retained gradient was released") }
    assert_values_near(kept_grad.download(), [3.0, 3.0, 3.0], 1e-5)
    let x_grad = x.grad
    if x_grad is null { throw AssertionError("expected x.grad") }
    assert_values_near(x_grad.download(), [6.0, -12.0, 3.0], 1e-5)
    free_graph(kept_held)
    kept_loss.free()
}

#: A retained gradient that the op below passes straight through (either side
#: of a sum) must not change when that input's gradient is added to later.
fn test_a_retained_gradient_is_not_shared_with_the_input_below_it() -> void {
    if not jt.is_available() { return }
    let x = jt.Tensor([2], [1.0, 2.0], requires_grad: true)
    defer { x.free() }
    let summed = jt.add(x, x).retain_grad()
    let loss = jt.sum(jt.add(summed, x))
    let held = loss.backward()
    let kept = summed.grad
    if kept is null { throw AssertionError("a retained gradient was released") }
    assert_values_near(kept.download(), [1.0, 1.0], 1e-6)
    let x_grad = x.grad
    if x_grad is null { throw AssertionError("expected x.grad") }
    assert_values_near(x_grad.download(), [3.0, 3.0], 1e-6)
    free_graph(held)
    loss.free()
}

#: `reshape`'s backward hands its input a view of the upstream gradient. For
#: a parameter that view pointed into the reshaped tensor's gradient, which
#: freeing the graph released, so the parameter's `.grad` read whatever the
#: pool handed out next.
fn test_a_reshaped_parameter_keeps_its_gradient_after_the_graph_is_freed() -> void {
    if not jt.is_available() { return }
    let w = jt.Tensor([2, 3], [1.0, 2.0, 3.0, -1.0, -2.0, 0.5], requires_grad: true)
    defer { w.free() }
    for step in 0..3 {
        let flat = jt.reshape(w, [6])
        let loss = jt.sum(jt.multiply(flat, flat))
        free_graph(loss.backward())
        loss.free()
        #: Take the freed memory back out of the pool and scribble on it.
        var scribbles: list[Tensor] = []
        for _i in 0..4 { scribbles.push(jt.fill([6], 7.0)) }
        gpu.synchronize()
        let gradient = w.grad
        if gradient is null { throw AssertionError("expected w.grad") }
        assert_true(not gradient.is_freed())
        let times = float(step + 1)
        assert_values_near(
            gradient.download(),
            [2.0 * times, 4.0 * times, 6.0 * times, -2.0 * times, -4.0 * times, 1.0 * times],
            1e-5
        )
        for scribble in scribbles { scribble.free() }
    }
}

fn test_backward_releases_what_only_one_op_saved_for_its_backward() -> void {
    if not jt.is_available() { return }
    if os.env_or("JAITENSOR_BACKWARD_FREE", "1") == "0" { return }
    let logits = jt.Tensor([2, 3], [1.0, 2.0, 0.5, 0.5, -1.0, 2.0], requires_grad: true)
    defer { logits.free() }
    let labels = jt.Tensor([2], [1.0, 2.0])
    defer { labels.free() }
    let loss = jt.SparseCategoricalCrossentropy(from_logits: true).forward(logits, labels)
    let node = loss.creator
    if node is null { throw AssertionError("expected a recorded loss") }
    let saved = node.extras
    assert_true(saved.len() > 0)
    let held = loss.backward()
    for tensor in saved { assert_true(tensor.is_freed()) }
    assert_true(logits.grad is not null)
    free_graph(held)
    loss.free()
}

fn test_no_grad_records_nothing_and_restores_the_previous_state() -> void {
    if not jt.is_available() { return }
    let x = jt.Tensor([2, 2], [1.0, 2.0, 3.0, 4.0])
    defer { x.free() }
    let w = jt.Tensor([2, 2], [0.5, -1.0, 1.5, 2.0], requires_grad: true)
    defer { w.free() }
    let gamma = jt.Tensor([2], [1.0, 1.0], requires_grad: true)
    defer { gamma.free() }
    let beta = jt.Tensor([2], [0.0, 0.0], requires_grad: true)
    defer { beta.free() }
    let labels = jt.Tensor([2], [0.0, 1.0])
    defer { labels.free() }
    assert_true(jt.is_grad_enabled())
    gpu.synchronize()
    let before = gpu.live_buffers()
    var seen: list[bool] = []
    jt.no_grad(fn() -> void {
        seen.push(jt.is_grad_enabled())
        let y = jt.matmul(x, w)
        assert_true(y.creator is null)
        assert_true(not y.requires_grad)
        let z = jt.add(y, w)
        assert_true(z.creator is null)
        let n = jt.layer_norm(z, gamma, beta)
        assert_true(n.creator is null)
        let loss = jt.SparseCategoricalCrossentropy(from_logits: true).forward(n, labels)
        assert_true(loss.creator is null)
        #: Fused while nothing records; the two-step form would leave its
        #: middle tensor to a graph that is never built.
        let gelu_out = jt.linear_gelu(x, w, beta)
        assert_true(gelu_out.creator is null)
        let residual_out = jt.linear_residual(x, w, x)
        assert_true(residual_out.creator is null)
        jt.no_grad(fn() -> void { seen.push(jt.is_grad_enabled()) })
        seen.push(jt.is_grad_enabled())
        for tensor in [y, z, n, loss, gelu_out, residual_out] { tensor.free() }
    })
    assert_eq(seen, [false, false, false])
    assert_true(jt.is_grad_enabled())
    gpu.synchronize()
    assert_eq(gpu.live_buffers(), before)
    assert_true(w.grad is null)

    var raised = false
    try {
        jt.no_grad(fn() -> void { throw ValueError("inside") })
    } catch _failure: ValueError {
        raised = true
    }
    assert_true(raised)
    assert_true(jt.is_grad_enabled())

    let previous = jt.set_grad_enabled(false)
    assert_true(previous)
    let off = jt.matmul(x, w)
    assert_true(off.creator is null)
    off.free()
    assert_true(not jt.set_grad_enabled(true))
    let on = jt.matmul(x, w)
    assert_true(on.creator is not null)
    free_graph(on.discard_graph())
    on.free()
}

#: Softmax cross-entropy of one row and its gradient, in double precision here.
fn ce_row_by_hand(row: list[float], label: int, scale: float) -> (float, list[float]) {
    var peak = row[0]
    for value in row { if value > peak { peak = value } }
    var total = 0.0
    for value in row { total += math.exp(value - peak) }
    let loss = peak + math.log(total) - row[label]
    var grad: list[float] = []
    for (index, value) in row.enumerate() {
        let hit = if index == label { 1.0 } else { 0.0 }
        grad.push((math.exp(value - peak) / total - hit) * scale)
    }
    return (loss, grad)
}

#: Both kernels (a simdgroup a row up to 1024 classes, a threadgroup above),
#: one class, and the labels at both ends, against arithmetic done here; and
#: the autograd gradient against the static path's delta, which must agree
#: bit for bit since the two share the arithmetic.
fn test_sparse_cross_entropy_matches_a_hand_computation_at_every_width() -> void {
    if not jt.is_available() { return }
    let ce = jt.SparseCategoricalCrossentropy(from_logits: true)
    for classes in [1, 7, 256, 1500, 4096] {
        for rows in [1, 3] {
            var values: list[float] = []
            for index in 0..rows * classes {
                values.push(float((index * 37) % 1009) / 256.0 - 2.0)
            }
            var labels: list[float] = []
            for row in 0..rows { labels.push(float(if row % 2 == 0 { classes - 1 } else { 0 })) }
            let logits = jt.Tensor([rows, classes], values, requires_grad: true)
            let target = jt.Tensor([rows], labels)
            let loss = ce.forward(logits, target)
            let held = loss.backward()
            var want_loss = 0.0
            var want_grad: list[float] = []
            for row in 0..rows {
                let slice = [values[row * classes + column] for column in 0..classes]
                let (row_loss, row_grad) = ce_row_by_hand(slice, int(labels[row]), 1.0 / float(rows))
                want_loss += row_loss / float(rows)
                for value in row_grad { want_grad.push(value) }
            }
            let got_loss = loss.download()[0]
            assert_near(got_loss, want_loss, 1e-5 * max(1.0, math.fabs(want_loss)))
            let gradient = logits.grad
            if gradient is null { throw AssertionError("expected logits.grad") }
            let got_grad = gradient.download()
            assert_values_near(got_grad, want_grad, 1e-6)
            let (static_loss, delta) = sparse_ce_with_delta(logits, target)
            assert_eq(static_loss.download()[0], got_loss)
            assert_eq(delta.download(), got_grad)
            for tensor in held { tensor.free() }
            for tensor in [static_loss, delta, loss, target, logits] { tensor.free() }
        }
    }
}

#: A weighted loss carries the weight into the gradient on the new path too.
fn test_sparse_cross_entropy_gradient_carries_the_upstream() -> void {
    if not jt.is_available() { return }
    let logits = jt.Tensor([2, 3], [1.0, 2.0, 0.5, 0.5, -1.0, 2.0], requires_grad: true)
    defer { logits.free() }
    let labels = jt.Tensor([2], [1.0, 2.0])
    defer { labels.free() }
    let ce = jt.SparseCategoricalCrossentropy(from_logits: true)
    let plain = ce.forward(logits, labels)
    free_graph(plain.backward())
    let once = logits.grad
    if once is null { throw AssertionError("expected logits.grad") }
    let single = once.download()
    clear_grad(logits)
    plain.free()
    let weighted = jt.scale(ce.forward(logits, labels), 3.0)
    free_graph(weighted.backward())
    weighted.free()
    let thrice = logits.grad
    if thrice is null { throw AssertionError("expected logits.grad") }
    assert_values_near(thrice.download(), [value * 3.0 for value in single], 1e-6)
}
