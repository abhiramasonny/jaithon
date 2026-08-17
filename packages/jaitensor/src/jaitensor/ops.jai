#: GPU tensor operations that keep their inputs and results on the device.
import std.gpu as gpu
from std.gpu import Kernel
from .tensor import GradNode, Tensor, element_count

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

kernel void broadcast_add(device const float* a [[buffer(0)]],
                          device const float* b [[buffer(1)]],
                          device float* output [[buffer(2)]],
                          constant uint& inner [[buffer(3)]],
                          constant uint& total [[buffer(4)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= total) return;
    output[id] = a[id] + b[id % inner];
}

kernel void binary_backward_op(device const float* a [[buffer(0)]],
                               device const float* b [[buffer(1)]],
                               device const float* upstream [[buffer(2)]],
                               device float* grad_a [[buffer(3)]],
                               device float* grad_b [[buffer(4)]],
                               constant uint& count [[buffer(5)]],
                               constant uint& operation [[buffer(6)]],
                               uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float u = upstream[id];
    if (operation == 0) { grad_a[id] = u; grad_b[id] = u; }
    else if (operation == 1) { grad_a[id] = u; grad_b[id] = -u; }
    else if (operation == 2) { grad_a[id] = u * b[id]; grad_b[id] = u * a[id]; }
    else {
        float bv = b[id];
        grad_a[id] = u / bv;
        grad_b[id] = -u * a[id] / (bv * bv);
    }
}

kernel void reduce_leading(device const float* upstream [[buffer(0)]],
                           device float* grad_small [[buffer(1)]],
                           constant uint& outer [[buffer(2)]],
                           constant uint& inner [[buffer(3)]],
                           uint id [[thread_position_in_grid]]) {
    if (id >= inner) return;
    float total = 0.0f;
    for (uint row = 0; row < outer; ++row) total += upstream[row * inner + id];
    grad_small[id] = total;
}

kernel void unary_backward_op(device const float* saved [[buffer(0)]],
                              device const float* upstream [[buffer(1)]],
                              device float* grad_input [[buffer(2)]],
                              constant uint& count [[buffer(3)]],
                              constant uint& operation [[buffer(4)]],
                              uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float s = saved[id];
    float u = upstream[id];
    if (operation == 0) grad_input[id] = u;
    else if (operation == 1) grad_input[id] = s > 0.0f ? u : 0.0f;
    else if (operation == 2) grad_input[id] = u * s * (1.0f - s);
    else if (operation == 3) grad_input[id] = u * (1.0f - s * s);
    else if (operation == 4) grad_input[id] = u * s;
    else grad_input[id] = u / s;
}

kernel void softmax_backward(device const float* output [[buffer(0)]],
                             device const float* upstream [[buffer(1)]],
                             device float* grad_input [[buffer(2)]],
                             constant uint& rows [[buffer(3)]],
                             constant uint& columns [[buffer(4)]],
                             uint row [[thread_position_in_grid]]) {
    if (row >= rows) return;
    uint base = row * columns;
    float dot = 0.0f;
    for (uint column = 0; column < columns; ++column) dot += upstream[base + column] * output[base + column];
    for (uint column = 0; column < columns; ++column)
        grad_input[base + column] = output[base + column] * (upstream[base + column] - dot);
}
"""

var _kernel_cache: dict[str, Kernel] = {}

fn cached_kernel(source: str, entry: str) -> Kernel {
    let key = entry + "\n" + source
    let existing = _kernel_cache.get(key, null)
    if existing is not null { return existing }
    let kernel = Kernel.compile(source, entry)
    _kernel_cache[key] = kernel
    return kernel
}

fn binary(a: Tensor, b: Tensor, operation: int) -> Tensor {
    if not a.same_shape(b) { throw ValueError(f"tensor shapes differ: {a.shape} and {b.shape}") }
    let output = Tensor(a.shape)
    let kernel = cached_kernel(SOURCE, "binary_op")
    kernel.dispatch_async([a.buffer(), b.buffer(), output.buffer()], a.size, [a.size, operation])
    return output
}

fn unary(input: Tensor, operation: int) -> Tensor {
    let output = Tensor(input.shape)
    let kernel = cached_kernel(SOURCE, "unary_op")
    kernel.dispatch_async([input.buffer(), output.buffer()], input.size, [input.size, operation])
    return output
}

fn is_trailing_suffix(small: list[int], large: list[int]) -> bool {
    if small.len() > large.len() { return false }
    let offset = large.len() - small.len()
    for index in 0..small.len() {
        if small[index] != large[offset + index] { return false }
    }
    return true
}

fn broadcast_add(a: Tensor, b: Tensor) -> Tensor {
    let output = Tensor(a.shape)
    let kernel = cached_kernel(SOURCE, "broadcast_add")
    kernel.dispatch_async([a.buffer(), b.buffer(), output.buffer()], a.size, [b.size, a.size])
    return output
}

fn binary_backward(a: Tensor, b: Tensor, upstream: Tensor, operation: int) -> list[Tensor] {
    let grad_a = Tensor(a.shape)
    let grad_b = Tensor(b.shape)
    let kernel = cached_kernel(SOURCE, "binary_backward_op")
    kernel.dispatch_async(
        [a.buffer(), b.buffer(), upstream.buffer(), grad_a.buffer(), grad_b.buffer()],
        a.size,
        [a.size, operation]
    )
    return [grad_a, grad_b]
}

fn reduce_leading(upstream: Tensor, small_shape: list[int]) -> Tensor {
    let inner = element_count(small_shape)
    let outer = upstream.size // inner
    let output = Tensor(small_shape)
    let kernel = cached_kernel(SOURCE, "reduce_leading")
    kernel.dispatch_async([upstream.buffer(), output.buffer()], inner, [outer, inner])
    return output
}

#: Elementwise sum of equal-shaped tensors, or of `a` with a `b` whose shape
#: is a trailing suffix of `a`'s (for example `[units]` against `[batch,
#: units]`).
pub fn add(a: Tensor, b: Tensor) -> Tensor {
    let broadcasting = not a.same_shape(b)
    if broadcasting and not is_trailing_suffix(b.shape, a.shape) {
        throw ValueError(f"cannot broadcast {b.shape} against {a.shape}")
    }
    let output = if broadcasting { broadcast_add(a, b) } else { binary(a, b, 0) }
    if a.requires_grad or b.requires_grad {
        output.attach_creator(GradNode([a, b], |upstream| if broadcasting {
            [unary(upstream, 0), reduce_leading(upstream, b.shape)]
        } else {
            binary_backward(a, b, upstream, 0)
        }))
    }
    return output
}

#: Elementwise difference of equal-shaped tensors.
pub fn subtract(a: Tensor, b: Tensor) -> Tensor {
    let output = binary(a, b, 1)
    if a.requires_grad or b.requires_grad {
        output.attach_creator(GradNode([a, b], |upstream| binary_backward(a, b, upstream, 1)))
    }
    return output
}

#: Elementwise product of equal-shaped tensors.
pub fn multiply(a: Tensor, b: Tensor) -> Tensor {
    let output = binary(a, b, 2)
    if a.requires_grad or b.requires_grad {
        output.attach_creator(GradNode([a, b], |upstream| binary_backward(a, b, upstream, 2)))
    }
    return output
}

#: Elementwise quotient of equal-shaped tensors.
pub fn divide(a: Tensor, b: Tensor) -> Tensor {
    let output = binary(a, b, 3)
    if a.requires_grad or b.requires_grad {
        output.attach_creator(GradNode([a, b], |upstream| binary_backward(a, b, upstream, 3)))
    }
    return output
}

#: Multiply every element by one host scalar.
pub fn scale(input: Tensor, value: float) -> Tensor {
    let output = Tensor(input.shape)
    let scalar = gpu.Buffer.of([value])
    defer { scalar.free() }
    let kernel = cached_kernel(SOURCE, "scalar_mul")
    kernel.dispatch_async([input.buffer(), scalar, output.buffer()], input.size, [input.size])
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [scale(upstream, value)]))
    }
    return output
}

#: Copy a tensor while giving the result a new shape with the same size.
pub fn reshape(input: Tensor, shape: list[int]) -> Tensor {
    if element_count(shape) != input.size {
        throw ValueError(f"cannot reshape {input.shape} into {shape}")
    }
    let output = Tensor(shape)
    let kernel = cached_kernel(SOURCE, "unary_op")
    kernel.dispatch_async([input.buffer(), output.buffer()], input.size, [input.size, 0])
    if input.requires_grad {
        let original_shape = input.shape
        output.attach_creator(GradNode([input], |upstream| [reshape(upstream, original_shape)]))
    }
    return output
}

fn raw_matmul(a: Tensor, b: Tensor) -> Tensor {
    let rows = a.shape[0]
    let inner = a.shape[1]
    let columns = b.shape[1]
    let output = Tensor([rows, columns])
    let kernel = cached_kernel(SOURCE, "matrix_mul")
    kernel.dispatch_async(
        [a.buffer(), b.buffer(), output.buffer()],
        rows * columns,
        [rows, inner, columns]
    )
    return output
}

fn raw_transpose(input: Tensor) -> Tensor {
    let rows = input.shape[0]
    let columns = input.shape[1]
    let output = Tensor([columns, rows])
    let kernel = cached_kernel(SOURCE, "matrix_transpose")
    kernel.dispatch_async([input.buffer(), output.buffer()], input.size, [rows, columns])
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
    let output = raw_matmul(a, b)
    if a.requires_grad or b.requires_grad {
        output.attach_creator(
            GradNode(
                [a, b],
                |upstream| [raw_matmul(upstream, raw_transpose(b)), raw_matmul(raw_transpose(a), upstream)]
            )
        )
    }
    return output
}

#: Transpose a rank-two tensor.
pub fn transpose(input: Tensor) -> Tensor {
    if input.rank() != 2 { throw ValueError(f"transpose needs rank two, got {input.shape}") }
    let output = raw_transpose(input)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [raw_transpose(upstream)]))
    }
    return output
}

fn unary_backward(saved: Tensor, upstream: Tensor, operation: int) -> Tensor {
    let grad_input = Tensor(saved.shape)
    let kernel = cached_kernel(SOURCE, "unary_backward_op")
    kernel.dispatch_async(
        [saved.buffer(), upstream.buffer(), grad_input.buffer()],
        saved.size,
        [saved.size, operation]
    )
    return grad_input
}

#: Elementwise rectified linear activation.
pub fn relu(input: Tensor) -> Tensor {
    let output = unary(input, 1)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [unary_backward(output, upstream, 1)]))
    }
    return output
}

#: Elementwise logistic sigmoid activation.
pub fn sigmoid(input: Tensor) -> Tensor {
    let output = unary(input, 2)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [unary_backward(output, upstream, 2)]))
    }
    return output
}

#: Elementwise hyperbolic tangent activation.
pub fn tanh(input: Tensor) -> Tensor {
    let output = unary(input, 3)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [unary_backward(output, upstream, 3)]))
    }
    return output
}

#: Elementwise exponential.
pub fn exp(input: Tensor) -> Tensor {
    let output = unary(input, 4)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [unary_backward(output, upstream, 4)]))
    }
    return output
}

#: Elementwise natural logarithm.
pub fn log(input: Tensor) -> Tensor {
    let output = unary(input, 5)
    if input.requires_grad {
        output.attach_creator(GradNode([input], |upstream| [unary_backward(input, upstream, 5)]))
    }
    return output
}

#: Normalize each row of a rank-two tensor into a probability distribution.
pub fn softmax(input: Tensor) -> Tensor {
    if input.rank() != 2 { throw ValueError(f"softmax needs rank two, got {input.shape}") }
    let rows = input.shape[0]
    let columns = input.shape[1]
    let output = Tensor(input.shape)
    let kernel = cached_kernel(SOURCE, "row_softmax")
    kernel.dispatch_async([input.buffer(), output.buffer()], rows, [rows, columns])
    if input.requires_grad {
        output.attach_creator(GradNode([input], fn(upstream: Tensor) -> list[Tensor] {
            let grad_input = Tensor(input.shape)
            let backward_kernel = cached_kernel(SOURCE, "softmax_backward")
            backward_kernel.dispatch_async([output.buffer(), upstream.buffer(), grad_input.buffer()], rows, [rows, columns])
            return [grad_input]
        }))
    }
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
    let kernel = cached_kernel(SOURCE, "row_argmax")
    kernel.dispatch_async([input.buffer(), output.buffer()], rows, [rows, columns])
    return output
}

#: Expand a rank-one tensor of integer-valued indices into one-hot rows.
pub fn one_hot(indices: Tensor, depth: int) -> Tensor {
    if indices.rank() != 1 { throw ValueError(f"one_hot needs rank one, got {indices.shape}") }
    if depth <= 0 { throw ValueError(f"depth must be positive, got {depth}") }
    let rows = indices.shape[0]
    let output = Tensor([rows, depth])
    let kernel = cached_kernel(SOURCE, "one_hot_rows")
    kernel.dispatch_async([indices.buffer(), output.buffer()], rows * depth, [rows, depth])
    return output
}
