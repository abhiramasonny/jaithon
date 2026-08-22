#: GPU-first tensor and neural-network training for Jaithon.
#:
#: Tensors live on the GPU. `Sequential` stacks `Layer` (dense, conv, norm,
#: attention). Hot kernels are `@trace`. `compile(..., mixed_precision: true)`
#: runs GEMM in float16. Shape annotations (`Tensor[float, N, M]`) are checked.
#:
#: ```
#: import jaitensor as jt
#: let data = jt.Dataset.from_rows([[0.0, 1.0], [1.0, 0.0]], [0, 1])
#: defer { data.free() }
#: let model = jt.Sequential([
#:     jt.Dense(128, activation: jt.Activation.Relu),
#:     jt.Dense(10, activation: jt.Activation.Softmax),
#: ])
#: defer { model.free() }
#: model.compile(jt.Adam(), mixed_precision: true)
#: model.fit(data, epochs: 5, batch_size: 2048, shuffle: true)
#: ```
import std.gpu as gpu
from .data import Dataset, DataLoader
from .layers import Activate, Activation, AvgPool2d, BatchNorm2d, Conv2d, Dense, Dropout, Embedding, Flatten, GroupNorm2d, Layer, LayerNorm, MaxPool2d, MultiHeadAttention
from .losses import Loss, BinaryCrossentropy, MeanSquaredError, SparseCategoricalCrossentropy
from .model import History, Metrics, Sequential
from .plan import ExecutionPlan
from .ops import add, add_correct_into, add_scaled_into, abs as abs_op, amax, argmax, avg_pool2d, batch_norm, cat, clamp, conv2d, conv2d_nchw, divide, dropout, exp, gather_rows, gelu, layer_norm, leaky_relu, log, matmul, max_with_index, maximum, mean, minimum, mse, multiply, neg, one_hot
from .ops import relu, reshape, rsqrt, scale, scaled_dot_product_attention, sigmoid, silu, slice_flat, softmax, sqrt, square, squeeze, stack, subtract, sum as reduce_sum, tanh, transpose, unsqueeze, where, max_pool2d, pad2d, permute, pow, repeat, nchw, nhwc, binary_cross_entropy, binary_cross_entropy_with_logits, eye, flip, upsample_nearest2d, split_heads, merge_heads, multi_head_attention, group_norm, tril
from .optimizers import Adam, Optimizer, SGD, zero_grad
from .plots import save_history_plot
from .tensor import Tensor, constant, fill, ones, zeros

export { Tensor, constant, zeros, ones, fill, Dataset, DataLoader }
export { Activate, Activation, AvgPool2d, BatchNorm2d, Conv2d, Dense, Dropout, Embedding, Flatten, GroupNorm2d, Layer, LayerNorm, MaxPool2d, MultiHeadAttention }
export { Optimizer, SGD, Adam, zero_grad, Loss, BinaryCrossentropy, MeanSquaredError, SparseCategoricalCrossentropy }
export { History, Metrics, Sequential, ExecutionPlan }
export { save_history_plot }
export { add, subtract, multiply, divide, scale, reshape, matmul, transpose, conv2d, conv2d_nchw, cat }
export { relu, sigmoid, tanh, gelu, silu, leaky_relu, exp, log, abs, neg, sqrt, rsqrt, square, softmax, amax, argmax, max_with_index, one_hot, slice_flat }
export { clamp, maximum, minimum, sum, mean, max_pool2d, avg_pool2d, dropout, mse, where, pow, squeeze, unsqueeze, stack, pad2d, permute, repeat, nchw, nhwc, binary_cross_entropy, binary_cross_entropy_with_logits }
export { batch_norm, layer_norm, group_norm, scaled_dot_product_attention, gather_rows }
export { eye, flip, upsample_nearest2d, split_heads, merge_heads, multi_head_attention, tril }
export { add_correct_into, add_scaled_into }

#: Whether the machine has a GPU backend that jaitensor can use.
pub fn is_available() -> bool { return gpu.is_available() }

#: Name of the active GPU, or `"none"` when none exists.
pub fn device_name() -> str { return gpu.device_name() }

#: Number of Metal devices.
pub fn device_count() -> int { return gpu.device_count() }

#: Select the Metal device used by later allocations.
pub fn set_device(index: int) -> void { gpu.set_device(index) }

#: Compute large GEMMs in float16 while keeping float32 storage.
pub fn set_mixed_precision(enabled: bool) -> void { gpu.set_mixed_precision(enabled) }

#: Elementwise absolute value of a tensor.
pub fn abs(input: Tensor) -> Tensor { return abs_op(input) }

#: Sum of every element, or of the last axis when `axis` is `-1`.
pub fn sum(input: Tensor, axis: int? = null) -> Tensor { return reduce_sum(input, axis: axis) }
