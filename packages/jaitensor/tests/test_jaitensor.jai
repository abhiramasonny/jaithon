#: GPU tensor operations, generic datasets, optimizers, and model lifecycle.
import jaitensor as jt
import std.math
import std.os
from jaitensor import Dataset
from jaitensor.tensor import GradNode, Tensor
from std.io import Path, read_file
from std.test import assert_eq, assert_near, assert_throws, assert_true

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
    let values = t.download()
    var total = 0.0
    for value in values { total += value }
    let scalar = jt.Tensor([1], [total], requires_grad: t.requires_grad)
    if t.requires_grad {
        scalar.creator = GradNode([t], fn(upstream: Tensor) -> list[Tensor] {
            let value = upstream.download()[0]
            return [jt.Tensor(t.shape, [value for _index in 0..t.size])]
        })
    }
    return scalar
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
