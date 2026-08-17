#: GPU-first tensor and neural-network training for Jaithon.
#:
#: `jaitensor` follows the composable part of TensorFlow/Keras. It provides
#: GPU-resident float32 tensors, common tensor operations, generic datasets,
#: dense `Sequential` models, validation metrics, prediction, weight files,
#: momentum SGD, and Adam. Jaithon code owns the model loop and queues Metal
#: kernels through `std.gpu`.
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
#: model.compile(jt.Adam())
#: model.fit(data, epochs: 5, batch_size: 2048)
#: ```
import std.gpu as gpu
from .data import Dataset
from .layers import Activation, Dense, Layer
from .losses import Loss, SparseCategoricalCrossentropy
from .model import History, Metrics, Sequential
from .ops import add, argmax, divide, exp, log, matmul, multiply, one_hot
from .ops import relu, reshape, scale, sigmoid, softmax, subtract, tanh, transpose
from .optimizers import Adam, Optimizer, SGD
from .plots import save_history_plot
from .tensor import Tensor, constant, fill, ones, zeros

export { Tensor, constant, zeros, ones, fill, Dataset }
export { Activation, Dense, Layer, Optimizer, SGD, Adam, Loss, SparseCategoricalCrossentropy }
export { History, Metrics, Sequential }
export { save_history_plot }
export { add, subtract, multiply, divide, scale, reshape, matmul, transpose }
export { relu, sigmoid, tanh, exp, log, softmax, argmax, one_hot }

#: Whether the machine has a GPU backend that jaitensor can use.
pub fn is_available() -> bool { return gpu.is_available() }

#: Name of the active GPU, or `"none"` when none exists.
pub fn device_name() -> str { return gpu.device_name() }
