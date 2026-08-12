#: GPU tensor operations that keep their inputs and results on the device.
import std.gpu as gpu
from std.gpu import Kernel
from .tensor import Tensor, element_count

let SOURCE = """#include <metal_stdlib>
using namespace metal;

kernel void binary_op(device const float* a [[buffer(0)]],
                      device const float* b [[buffer(1)]],
                      device float* output [[buffer(2)]],
                      constant uint& count [[buffer(3)]],
                      constant uint& operation [[buffer(4)]],
                      uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    if (operation == 0) output[id] = a[id] + b[id];
    else if (operation == 1) output[id] = a[id] - b[id];
    else if (operation == 2) output[id] = a[id] * b[id];
    else output[id] = a[id] / b[id];
}

kernel void unary_op(device const float* input [[buffer(0)]],
                     device float* output [[buffer(1)]],
                     constant uint& count [[buffer(2)]],
                     constant uint& operation [[buffer(3)]],
                     uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float value = input[id];
    if (operation == 0) output[id] = value;
    else if (operation == 1) output[id] = max(value, 0.0f);
    else if (operation == 2) output[id] = 1.0f / (1.0f + exp(-value));
    else if (operation == 3) output[id] = tanh(value);
    else if (operation == 4) output[id] = exp(value);
    else output[id] = log(value);
}

kernel void scalar_mul(device const float* input [[buffer(0)]],
                       device const float* scalar [[buffer(1)]],
                       device float* output [[buffer(2)]],
                       constant uint& count [[buffer(3)]],
                       uint id [[thread_position_in_grid]]) {
    if (id < count) output[id] = input[id] * scalar[0];
}

kernel void matrix_mul(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* output [[buffer(2)]],
                       constant uint& rows [[buffer(3)]],
                       constant uint& inner [[buffer(4)]],
                       constant uint& columns [[buffer(5)]],
                       uint id [[thread_position_in_grid]]) {
    if (id >= rows * columns) return;
    uint row = id / columns;
    uint column = id % columns;
    float total = 0.0f;
    for (uint k = 0; k < inner; ++k)
        total += a[row * inner + k] * b[k * columns + column];
    output[id] = total;
}

kernel void matrix_transpose(device const float* input [[buffer(0)]],
                             device float* output [[buffer(1)]],
                             constant uint& rows [[buffer(2)]],
                             constant uint& columns [[buffer(3)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= rows * columns) return;
    uint row = id / columns;
    uint column = id % columns;
    output[column * rows + row] = input[id];
}

kernel void row_softmax(device const float* input [[buffer(0)]],
                        device float* output [[buffer(1)]],
                        constant uint& rows [[buffer(2)]],
                        constant uint& columns [[buffer(3)]],
                        uint row [[thread_position_in_grid]]) {
    if (row >= rows) return;
    uint base = row * columns;
    float peak = input[base];
    for (uint column = 1; column < columns; ++column)
        peak = max(peak, input[base + column]);
    float total = 0.0f;
    for (uint column = 0; column < columns; ++column)
        total += exp(input[base + column] - peak);
    for (uint column = 0; column < columns; ++column)
        output[base + column] = exp(input[base + column] - peak) / total;
}

kernel void row_argmax(device const float* input [[buffer(0)]],
                       device float* output [[buffer(1)]],
                       constant uint& rows [[buffer(2)]],
                       constant uint& columns [[buffer(3)]],
                       uint row [[thread_position_in_grid]]) {
    if (row >= rows) return;
    uint base = row * columns;
    float peak = input[base];
    uint best = 0;
    for (uint column = 1; column < columns; ++column) {
        float value = input[base + column];
        if (value > peak) { peak = value; best = column; }
    }
    output[row] = float(best);
}

kernel void one_hot_rows(device const float* indices [[buffer(0)]],
                         device float* output [[buffer(1)]],
                         constant uint& rows [[buffer(2)]],
                         constant uint& depth [[buffer(3)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= rows * depth) return;
    uint row = id / depth;
    uint column = id % depth;
    output[id] = int(indices[row]) == int(column) ? 1.0f : 0.0f;
}
"""

fn binary(a: Tensor, b: Tensor, operation: int) -> Tensor {
    if not a.same_shape(b) { throw ValueError(f"tensor shapes differ: {a.shape} and {b.shape}") }
    let output = Tensor(a.shape)
    let kernel = Kernel.compile(SOURCE, "binary_op")
    defer { kernel.free() }
    kernel.dispatch([a.buffer(), b.buffer(), output.buffer()], a.size, [a.size, operation])
    return output
}

fn unary(input: Tensor, operation: int) -> Tensor {
    let output = Tensor(input.shape)
    let kernel = Kernel.compile(SOURCE, "unary_op")
    defer { kernel.free() }
    kernel.dispatch([input.buffer(), output.buffer()], input.size, [input.size, operation])
    return output
}

#: Elementwise sum of equal-shaped tensors.
pub fn add(a: Tensor, b: Tensor) -> Tensor { return binary(a, b, 0) }

#: Elementwise difference of equal-shaped tensors.
pub fn subtract(a: Tensor, b: Tensor) -> Tensor { return binary(a, b, 1) }

#: Elementwise product of equal-shaped tensors.
pub fn multiply(a: Tensor, b: Tensor) -> Tensor { return binary(a, b, 2) }

#: Elementwise quotient of equal-shaped tensors.
pub fn divide(a: Tensor, b: Tensor) -> Tensor { return binary(a, b, 3) }

#: Multiply every element by one host scalar.
pub fn scale(input: Tensor, value: float) -> Tensor {
    let output = Tensor(input.shape)
    let scalar = gpu.Buffer.of([value])
    let kernel = Kernel.compile(SOURCE, "scalar_mul")
    defer { scalar.free() }
    defer { kernel.free() }
    kernel.dispatch([input.buffer(), scalar, output.buffer()], input.size, [input.size])
    return output
}

#: Copy a tensor while giving the result a new shape with the same size.
pub fn reshape(input: Tensor, shape: list[int]) -> Tensor {
    if element_count(shape) != input.size {
        throw ValueError(f"cannot reshape {input.shape} into {shape}")
    }
    let output = Tensor(shape)
    let kernel = Kernel.compile(SOURCE, "unary_op")
    defer { kernel.free() }
    kernel.dispatch([input.buffer(), output.buffer()], input.size, [input.size, 0])
    return output
}

#: Matrix product of two rank-two tensors.
pub fn matmul(a: Tensor, b: Tensor) -> Tensor {
    if a.rank() != 2 or b.rank() != 2 {
        throw ValueError(f"matmul needs rank-two tensors, got {a.shape} and {b.shape}")
    }
    if a.shape[1] != b.shape[0] {
        throw ValueError(f"matmul inner dimensions differ: {a.shape} and {b.shape}")
    }
    let rows = a.shape[0]
    let inner = a.shape[1]
    let columns = b.shape[1]
    let output = Tensor([rows, columns])
    let kernel = Kernel.compile(SOURCE, "matrix_mul")
    defer { kernel.free() }
    kernel.dispatch(
        [a.buffer(), b.buffer(), output.buffer()],
        rows * columns,
        [rows, inner, columns]
    )
    return output
}

#: Transpose a rank-two tensor.
pub fn transpose(input: Tensor) -> Tensor {
    if input.rank() != 2 { throw ValueError(f"transpose needs rank two, got {input.shape}") }
    let rows = input.shape[0]
    let columns = input.shape[1]
    let output = Tensor([columns, rows])
    let kernel = Kernel.compile(SOURCE, "matrix_transpose")
    defer { kernel.free() }
    kernel.dispatch([input.buffer(), output.buffer()], input.size, [rows, columns])
    return output
}

#: Elementwise rectified linear activation.
pub fn relu(input: Tensor) -> Tensor { return unary(input, 1) }

#: Elementwise logistic sigmoid activation.
pub fn sigmoid(input: Tensor) -> Tensor { return unary(input, 2) }

#: Elementwise hyperbolic tangent activation.
pub fn tanh(input: Tensor) -> Tensor { return unary(input, 3) }

#: Elementwise exponential.
pub fn exp(input: Tensor) -> Tensor { return unary(input, 4) }

#: Elementwise natural logarithm.
pub fn log(input: Tensor) -> Tensor { return unary(input, 5) }

#: Normalize each row of a rank-two tensor into a probability distribution.
pub fn softmax(input: Tensor) -> Tensor {
    if input.rank() != 2 { throw ValueError(f"softmax needs rank two, got {input.shape}") }
    let rows = input.shape[0]
    let columns = input.shape[1]
    let output = Tensor(input.shape)
    let kernel = Kernel.compile(SOURCE, "row_softmax")
    defer { kernel.free() }
    kernel.dispatch([input.buffer(), output.buffer()], rows, [rows, columns])
    return output
}

#: Index of the largest value in each row of a rank-two tensor.
#:
#: Indices use float storage because all Metal buffers in `std.gpu` are
#: float32. Downloaded values are exact for practical class counts.
pub fn argmax(input: Tensor) -> Tensor {
    if input.rank() != 2 { throw ValueError(f"argmax needs rank two, got {input.shape}") }
    let rows = input.shape[0]
    let columns = input.shape[1]
    let output = Tensor([rows])
    let kernel = Kernel.compile(SOURCE, "row_argmax")
    defer { kernel.free() }
    kernel.dispatch([input.buffer(), output.buffer()], rows, [rows, columns])
    return output
}

#: Expand a rank-one tensor of integer-valued indices into one-hot rows.
pub fn one_hot(indices: Tensor, depth: int) -> Tensor {
    if indices.rank() != 1 { throw ValueError(f"one_hot needs rank one, got {indices.shape}") }
    if depth <= 0 { throw ValueError(f"depth must be positive, got {depth}") }
    let rows = indices.shape[0]
    let output = Tensor([rows, depth])
    let kernel = Kernel.compile(SOURCE, "one_hot_rows")
    defer { kernel.free() }
    kernel.dispatch([indices.buffer(), output.buffer()], rows * depth, [rows, depth])
    return output
}
