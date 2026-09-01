#: Backward passes against a central difference, and against arithmetic done here.
import jaitensor as jt
import std.math
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
