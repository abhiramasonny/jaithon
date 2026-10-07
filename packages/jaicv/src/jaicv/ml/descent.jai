#: Logistic regression's gradient descent on the device.
#:
#: Kept apart from `ml.models` so that importing the models does not load
#: jaitensor; `LogisticRegression.train` imports this when a training set is
#: big enough to send.
import jaitensor as jt
from jaitensor import Tensor
from ..core.device import device_view
from ..core.mat import Mat

#: The most values, counting the intercept's column, the resident descent
#: takes on. It holds four tensors the size of the samples -- the transposed
#: samples, the spread targets, the errors and their products -- which for a
#: million samples of a hundred features is 1.6 GB; above this the stepwise
#: descent, which holds one, runs instead.
pub let RESIDENT_VALUES = 1 << 25

#: `iterations` full-batch steps from `weights` -- the features' weights, then
#: the intercept's -- or null when the samples are not a plain run of floats
#: on the device.
#:
#: A column of ones is appended to the samples once, so the intercept is one
#: more weight and its gradient comes out of the same product. `wanted` holds
#: 1 for a positive sample and 0 for a negative one.
pub fn descend(
    samples: Mat,
    wanted: list[float],
    weights: list[float],
    learning_rate: float,
    regularisation: float,
    iterations: int,
    resident: bool
) -> list[float]? {
    let x = device_view(samples)
    if x is null { return null }
    let count = samples.rows
    let ones = Tensor.full([count, 1], 1.0)
    let augmented = jt.cat([x, ones], axis: -1)
    x.free()
    ones.free()
    let truth = Tensor([count, 1], values: wanted)
    let found = if resident and count * weights.len() <= RESIDENT_VALUES {
        descend_resident(augmented, truth, weights, learning_rate, regularisation, iterations)
    } else {
        descend_stepwise(augmented, truth, weights, learning_rate, regularisation, iterations)
    }
    augmented.free()
    truth.free()
    return found
}

#: Each step's gradient read back and the weights updated on the host, in
#: double precision: `LogisticRegression.step`'s update.
fn descend_stepwise(
    augmented: Tensor,
    truth: Tensor,
    start: list[float],
    learning_rate: float,
    regularisation: float,
    iterations: int
) -> list[float] {
    let count = augmented.shape[0]
    let width = start.len() - 1
    var weights = [value for value in start]
    for _step in 0..iterations {
        let current = Tensor([width + 1, 1], values: weights)
        let scores = jt.matmul(augmented, current)
        current.free()
        let predicted = jt.sigmoid(scores)
        scores.free()
        let errors = jt.subtract(predicted, truth)
        predicted.free()
        let summed = jt.matmul(augmented, errors, trans_a: true)
        errors.free()
        let gradient = summed.download()
        summed.free()
        let scale = learning_rate / float(count)
        for column in 0..width {
            #: The penalty pulls the weights back towards zero, which is
            #: what keeps a separable problem from running them off to
            #: infinity.
            weights[column] -= scale * (gradient[column] + regularisation * weights[column])
        }
        weights[width] -= scale * gradient[width]
    }
    return weights
}

#: The same descent with the weights kept on the device too, so the steps
#: queue one behind another with nothing read back until the last.
#:
#: Reading the gradient back each step put a round trip to the device in
#: every one of the 500. And the samples' transpose times the errors is a
#: matrix-vector product whose inner dimension is the whole training set,
#: which the GEMM splits too few ways: 3.5 ms of a 4.2 ms step at 100,000
#: samples. Here it is the errors spread along each feature's row of the
#: transposed samples, multiplied in and summed along the row -- three
#: passes over memory a reduction can split as finely as it likes.
#:
#: Each step is seven dispatches, and at 10,000 samples it is the
#: dispatches, not the arithmetic, that cost: the targets are spread along
#: every row once, negated, so adding the predictions to them is the
#: subtraction too, and the transposed samples carry the step size, so the
#: summed rows are already the step.
#:
#: The weights are float32 here, where the host keeps them in double.
fn descend_resident(
    augmented: Tensor,
    truth: Tensor,
    start: list[float],
    learning_rate: float,
    regularisation: float,
    iterations: int
) -> list[float] {
    let count = augmented.shape[0]
    let columns = start.len()
    let width = columns - 1
    let scale = learning_rate / float(count)
    let transposed = jt.transpose(augmented)
    let turned = jt.scale(transposed, scale)
    transposed.free()
    let negated = jt.scale(truth, -1.0)
    let negated_row = jt.reshape(negated, [count])
    let zeros = Tensor.zeros([columns, count])
    let spread = jt.add(zeros, negated_row)
    zeros.free()
    negated_row.free()
    negated.free()
    var keep: list[float] = []
    for _column in 0..width { keep.push(1.0 - scale * regularisation) }
    keep.push(1.0)
    let decay = Tensor([columns, 1], values: keep)
    var weights = Tensor([columns, 1], values: start)
    for _step in 0..iterations {
        let scores = jt.matmul(augmented, weights)
        let predicted = jt.sigmoid(scores)
        scores.free()
        let flat = jt.reshape(predicted, [count])
        let errors = jt.add(spread, flat)
        flat.free()
        predicted.free()
        let products = jt.multiply(turned, errors)
        errors.free()
        let summed = jt.sum(products, axis: -1)
        products.free()
        let moved = jt.reshape(summed, [columns, 1])
        let kept = jt.multiply(weights, decay)
        weights.free()
        weights = jt.subtract(kept, moved)
        kept.free()
        moved.free()
        summed.free()
    }
    let found = weights.download()
    weights.free()
    decay.free()
    spread.free()
    turned.free()
    return found
}
