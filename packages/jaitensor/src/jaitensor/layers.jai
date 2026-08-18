#: Neural-network layers.
import std.math
from std.gpu import Buffer
from std.random import Random
from .ops import conv2d, dense_affine, dense_static_backward, dense_static_forward
from .ops import dropout, gather_rows, gelu, layer_norm, leaky_relu, max_pool2d, avg_pool2d
from .ops import batch_norm, group_norm, multi_head_attention, silu, matmul
from .tensor import Tensor

#: Something `Sequential` can stack: turns an input tensor into an output
#: tensor, and reports the tensors it owns that training should update.
#:
#: `Sequential` only ever calls trait methods on a `Layer` — never
#: downcasts to a concrete type — so a future layer kind (Conv2d, etc.)
#: plugs in by implementing this trait, nothing in `model.jai` changes.
pub trait Layer {
    #: Turn one batch's input into this layer's output.
    fn forward(self, x: Tensor) -> Tensor

    #: This layer's trainable tensors, in a stable order.
    fn parameters(self) -> list[Tensor]

    #: Allocate this layer's parameters for a known input width.
    fn build(self, input_units: int, rng: Random) -> void

    #: Width of this layer's output.
    fn output_width(self) -> int

    #: Release this layer's parameters.
    fn free(self) -> void

    #: Whether `Sequential.predict` should apply softmax after this layer.
    fn wants_softmax_output(self) -> bool { return false }

    #: Hidden ReLU dense layer that the fused two-layer SGD path can target.
    fn is_relu_dense(self) -> bool { return false }

    #: Linear or softmax classification head for the fused two-layer SGD path.
    fn is_classification_head(self) -> bool { return false }

    #: Whether this layer can run the static compile-time forward/backward path.
    fn supports_static(self) -> bool { return false }

    #: Forward one batch without building an autograd graph.
    fn static_forward(self, x: Tensor) -> Tensor {
        throw RuntimeError("layer does not support static execution")
    }

    #: Backpropagate through one layer into its parameters; returns dInput.
    fn static_backward(self, x: Tensor, output: Tensor, upstream: Tensor) -> Tensor {
        throw RuntimeError("layer does not support static execution")
    }

    #: Allocate parameters from a possibly multi-dimensional input shape.
    fn build_from_shape(self, shape: list[int], rng: Random) -> void {
        var n = 1
        for dimension in shape { n *= dimension }
        self.build(n, rng)
    }

    #: Shape this layer emits, excluding the batch axis.
    fn output_shape(self) -> list[int] {
        return [self.output_width()]
    }

    #: Forward used by `Sequential.predict`. Dropout becomes identity.
    fn eval_forward(self, x: Tensor) -> Tensor { return self.forward(x) }
}

#: Activation applied by a dense layer.
pub enum Activation {
    #: Leave the layer output unchanged.
    Linear,
    #: Clamp negative values to zero.
    Relu,
    #: Map values through the logistic sigmoid function.
    Sigmoid,
    #: Map values through the hyperbolic tangent function.
    Tanh,
    #: Interpret the final logits as a class distribution.
    Softmax,
    #: Gaussian error linear unit (tanh approximation).
    Gelu,
    #: Sigmoid linear unit.
    Silu,
    #: Leaky ReLU with negative slope 0.01.
    LeakyRelu,
}

#: A fully connected layer, compatible with `Sequential`.
pub class Dense: Layer {
    #: Number of output units.
    pub let units: int
    #: Function applied after the affine transform.
    pub let activation: Activation
    #: Input width, zero until a model builds the layer.
    pub var input_units: int
    var weights_value: Tensor?
    var bias_value: Tensor?

    #: Configure a dense layer. `Sequential` allocates its parameters on build.
    pub fn init(self, units: int, activation: Activation = Activation.Linear) {
        if units <= 0 { throw ValueError(f"Dense units must be positive, got {units}") }
        self.units = units
        self.activation = activation
        self.input_units = 0
        self.weights_value = null
        self.bias_value = null
    }

    #: Allocate and initialize parameters after the input width is known.
    pub fn build(self, input_units: int, rng: Random) -> void {
        if self.weights_value is not null { return }
        if input_units <= 0 {
            throw ValueError(f"Dense input width must be positive, got {input_units}")
        }
        self.input_units = input_units
        let limit = math.sqrt(if self.activation == Activation.Relu {
            6.0 / float(input_units)
        } else {
            3.0 / float(input_units)
        })
        self.weights_value = Tensor([input_units, self.units], requires_grad: true)
        self.weights_value.buffer().fill_uniform(-limit, limit, rng.next_int())
        self.bias_value = Tensor.zeros([self.units])
        self.bias_value.requires_grad = true
    }

    #: Run one batch through this layer's affine transform and activation.
    pub fn forward(self, x: Tensor) -> Tensor {
        let weights = self.weights_value
        let bias = self.bias_value
        if weights is null or bias is null { throw RuntimeError("Dense layer has not been built") }
        let affine = dense_affine(x, weights, bias, self.activation_code())
        if self.activation == Activation.Gelu { return gelu(affine) }
        if self.activation == Activation.Silu { return silu(affine) }
        if self.activation == Activation.LeakyRelu { return leaky_relu(affine) }
        return affine
    }

    #: This layer's trainable tensors, in the order `get_weights` reports them.
    pub fn parameters(self) -> list[Tensor] {
        let weights = self.weights_value
        let bias = self.bias_value
        if weights is null or bias is null { throw RuntimeError("Dense layer has not been built") }
        return [weights, bias]
    }

    #: Width of this layer's output — what the next layer's `build` needs.
    pub fn output_width(self) -> int { return self.units }

    #: Whether `Sequential.predict` should apply softmax after this layer.
    pub fn wants_softmax_output(self) -> bool { return self.activation == Activation.Softmax }

    #: Hidden ReLU dense layer that the fused two-layer SGD path can target.
    pub fn is_relu_dense(self) -> bool { return self.activation == Activation.Relu }

    #: Linear or softmax classification head for the fused two-layer SGD path.
    pub fn is_classification_head(self) -> bool {
        return self.activation == Activation.Softmax or self.activation == Activation.Linear
    }

    pub fn supports_static(self) -> bool {
        return self.activation == Activation.Linear or
            self.activation == Activation.Relu or
            self.activation == Activation.Softmax
    }

    @trace
    pub fn static_forward(self, x: Tensor) -> Tensor {
        let weights = self.weights_value
        let bias = self.bias_value
        if weights is null or bias is null { throw RuntimeError("Dense layer has not been built") }
        return dense_static_forward(x, weights, bias, self.activation_code())
    }

    @trace
    pub fn static_backward(self, x: Tensor, output: Tensor, upstream: Tensor) -> Tensor {
        let weights = self.weights_value
        let bias = self.bias_value
        if weights is null or bias is null { throw RuntimeError("Dense layer has not been built") }
        return dense_static_backward(
            x,
            weights,
            bias,
            output,
            upstream,
            self.activation_code()
        )
    }

    #: Return the row-major `[input_units, units]` parameter buffer.
    pub fn weights(self) -> Buffer {
        let value = self.weights_value
        if value is null { throw RuntimeError("Dense layer has not been built") }
        return value.buffer()
    }

    #: Return the `units`-element bias buffer.
    pub fn bias(self) -> Buffer {
        let value = self.bias_value
        if value is null { throw RuntimeError("Dense layer has not been built") }
        return value.buffer()
    }

    #: Encode the activation for the generic Metal kernels.
    pub fn activation_code(self) -> int {
        return match self.activation {
            Activation.Linear => 0,
            Activation.Relu => 1,
            Activation.Sigmoid => 2,
            Activation.Tanh => 3,
            Activation.Softmax => 0,
            Activation.Gelu => 0,
            Activation.Silu => 0,
            Activation.LeakyRelu => 0,
        }
    }

    #: Copy the current weight matrix and bias vector to the host.
    pub fn get_weights(self) -> tuple[list[float], list[float]] {
        let weights = self.weights_value
        let bias = self.bias_value
        if weights is null or bias is null { throw RuntimeError("Dense layer has not been built") }
        return (weights.download(), bias.download())
    }

    #: Replace the weight matrix and bias vector from host values.
    pub fn set_weights(self, weights: list[float], bias: list[float]) -> void {
        let weight_tensor = self.weights_value
        let bias_tensor = self.bias_value
        if weight_tensor is null or bias_tensor is null {
            throw RuntimeError("Dense layer has not been built")
        }
        if weights.len() != self.input_units * self.units {
            throw ValueError(
                f"Dense weights have {weights.len()} values, expected " +
                    f"{self.input_units * self.units}"
            )
        }
        if bias.len() != self.units {
            throw ValueError(f"Dense bias has {bias.len()} values, expected {self.units}")
        }
        weight_tensor.upload(weights)
        bias_tensor.upload(bias)
    }

    #: Release parameter buffers. Freeing an unbuilt layer is harmless.
    pub fn free(self) -> void {
        let weights = self.weights_value
        let bias = self.bias_value
        if weights is not null { weights.free() }
        if bias is not null { bias.free() }
        self.weights_value = null
        self.bias_value = null
    }

    fn __str__(self) -> str {
        return f"Dense(input={self.input_units}, units={self.units}, activation={self.activation})"
    }
}

#: Collapse every non-batch axis so a dense layer can follow a convolution.
pub class Flatten: Layer {
    var width: int

    pub fn init(self) { self.width = 0 }

    pub fn build(self, input_units: int, rng: Random) -> void {
        let _ = rng
        if input_units <= 0 { throw ValueError("Flatten needs a positive input width") }
        self.width = input_units
    }

    pub fn forward(self, x: Tensor) -> Tensor {
        let batch = x.shape[0]
        return Tensor([batch, x.size // batch], storage: x.buffer().view(0, x.size))
    }

    pub fn parameters(self) -> list[Tensor] { return [] }

    pub fn output_width(self) -> int { return self.width }

    pub fn free(self) -> void { }

    fn __str__(self) -> str { return f"Flatten({self.width})" }
}

#: 2D convolution over NHWC tensors.
pub class Conv2d: Layer {
    pub let out_channels: int
    pub let kernel_size: int
    pub let stride: int
    pub let padding: int
    pub var in_channels: int
    pub var in_height: int
    pub var in_width: int
    var weights_value: Tensor?
    var bias_value: Tensor?

    pub fn init(
        self,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        in_channels: int = 0
    ) {
        if out_channels <= 0 or kernel_size <= 0 {
            throw ValueError("Conv2d out_channels and kernel_size must be positive")
        }
        if stride <= 0 { throw ValueError("Conv2d stride must be positive") }
        if padding < 0 { throw ValueError("Conv2d padding must be non-negative") }
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.in_height = 0
        self.in_width = 0
        self.weights_value = null
        self.bias_value = null
    }

    pub fn build(self, input_units: int, rng: Random) -> void {
        if self.in_channels <= 0 {
            throw ValueError("Conv2d needs in_channels or a 3-D input shape")
        }
        if self.in_height <= 0 or self.in_width <= 0 {
            if input_units % self.in_channels != 0 {
                throw ValueError("Conv2d cannot infer spatial size from a flat width")
            }
        }
        self._allocate(rng)
    }

    pub fn build_from_shape(self, shape: list[int], rng: Random) -> void {
        if shape.len() == 3 {
            self.in_height = shape[0]
            self.in_width = shape[1]
            self.in_channels = shape[2]
        } elif shape.len() == 1 {
            if self.in_channels <= 0 {
                throw ValueError("Conv2d needs in_channels when the input is flat")
            }
        } else {
            throw ValueError(f"Conv2d expected [H, W, C], got {shape}")
        }
        self._allocate(rng)
    }

    fn _allocate(self, rng: Random) -> void {
        if self.weights_value is not null { return }
        if self.in_channels <= 0 {
            throw ValueError("Conv2d in_channels must be set before build")
        }
        let fan = self.kernel_size * self.kernel_size * self.in_channels
        let limit = math.sqrt(6.0 / float(fan + self.out_channels))
        self.weights_value = Tensor(
            [self.kernel_size, self.kernel_size, self.in_channels, self.out_channels],
            requires_grad: true
        )
        self.weights_value.buffer().fill_uniform(-limit, limit, rng.next_int())
        self.bias_value = Tensor.zeros([self.out_channels])
        self.bias_value.requires_grad = true
    }

    pub fn forward(self, x: Tensor) -> Tensor {
        let weights = self.weights_value
        let bias = self.bias_value
        if weights is null or bias is null { throw RuntimeError("Conv2d layer has not been built") }
        var input = x
        if x.rank() == 2 {
            if self.in_height <= 0 or self.in_width <= 0 {
                throw ValueError("Conv2d needs a known spatial size for flat inputs")
            }
            input = Tensor(
                [x.shape[0], self.in_height, self.in_width, self.in_channels],
                storage: x.buffer()
            )
        }
        return conv2d(input, weights, bias: bias, stride: self.stride, padding: self.padding)
    }

    pub fn parameters(self) -> list[Tensor] {
        let weights = self.weights_value
        let bias = self.bias_value
        if weights is null or bias is null { throw RuntimeError("Conv2d layer has not been built") }
        return [weights, bias]
    }

    pub fn output_width(self) -> int {
        let shape = self.output_shape()
        var n = 1
        for dimension in shape { n *= dimension }
        return n
    }

    pub fn output_shape(self) -> list[int] {
        if self.in_height <= 0 or self.in_width <= 0 {
            return [self.out_channels]
        }
        let out_h = (self.in_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        let out_w = (self.in_width + 2 * self.padding - self.kernel_size) // self.stride + 1
        return [out_h, out_w, self.out_channels]
    }

    pub fn free(self) -> void {
        let weights = self.weights_value
        let bias = self.bias_value
        if weights is not null { weights.free() }
        if bias is not null { bias.free() }
        self.weights_value = null
        self.bias_value = null
    }

    fn __str__(self) -> str {
        return f"Conv2d({self.in_channels}->{self.out_channels}, k={self.kernel_size})"
    }
}

#: Per-channel batch normalization over NHWC activations.
pub class BatchNorm2d: Layer {
    pub let channels: int
    var last_shape: list[int]
    var gamma_value: Tensor?
    var beta_value: Tensor?

    pub fn init(self, channels: int = 0) {
        self.channels = channels
        self.last_shape = []
        self.gamma_value = null
        self.beta_value = null
    }

    pub fn build(self, input_units: int, rng: Random) -> void {
        let _ = rng
        let width = if self.channels > 0 { self.channels } else { input_units }
        if self.gamma_value is not null { return }
        self.gamma_value = Tensor.ones([width])
        self.gamma_value.requires_grad = true
        self.beta_value = Tensor.zeros([width])
        self.beta_value.requires_grad = true
    }

    pub fn build_from_shape(self, shape: list[int], rng: Random) -> void {
        let channels = if shape.len() == 3 { shape[2] } else { shape[shape.len() - 1] }
        if self.channels > 0 and self.channels != channels {
            throw ValueError(f"BatchNorm2d expected {self.channels} channels, got {channels}")
        }
        self.build(channels, rng)
        self.last_shape = shape
    }

    pub fn output_shape(self) -> list[int] {
        if self.last_shape.len() > 0 { return self.last_shape }
        return [self.output_width()]
    }

    pub fn forward(self, x: Tensor) -> Tensor {
        let gamma = self.gamma_value
        let beta = self.beta_value
        if gamma is null or beta is null { throw RuntimeError("BatchNorm2d has not been built") }
        return batch_norm(x, gamma, beta)
    }

    pub fn parameters(self) -> list[Tensor] {
        let gamma = self.gamma_value
        let beta = self.beta_value
        if gamma is null or beta is null { throw RuntimeError("BatchNorm2d has not been built") }
        return [gamma, beta]
    }

    pub fn output_width(self) -> int {
        let gamma = self.gamma_value
        if gamma is null { return self.channels }
        return gamma.size
    }

    pub fn free(self) -> void {
        let gamma = self.gamma_value
        let beta = self.beta_value
        if gamma is not null { gamma.free() }
        if beta is not null { beta.free() }
        self.gamma_value = null
        self.beta_value = null
    }

    fn __str__(self) -> str { return f"BatchNorm2d({self.output_width()})" }
}

#: NHWC group normalization. `groups` must divide the channel count.
pub class GroupNorm2d: Layer {
    pub let groups: int
    var channels: int
    var last_shape: list[int]
    var gamma_value: Tensor?
    var beta_value: Tensor?

    pub fn init(self, groups: int, channels: int = 0) {
        if groups <= 0 { throw ValueError("GroupNorm2d groups must be positive") }
        self.groups = groups
        self.channels = channels
        self.last_shape = []
        self.gamma_value = null
        self.beta_value = null
    }

    pub fn build(self, input_units: int, rng: Random) -> void {
        let _ = rng
        let width = if self.channels > 0 { self.channels } else { input_units }
        if width % self.groups != 0 {
            throw ValueError(f"GroupNorm2d channels {width} must divide groups {self.groups}")
        }
        if self.gamma_value is not null { return }
        self.channels = width
        self.gamma_value = Tensor.ones([width])
        self.gamma_value.requires_grad = true
        self.beta_value = Tensor.zeros([width])
        self.beta_value.requires_grad = true
    }

    pub fn build_from_shape(self, shape: list[int], rng: Random) -> void {
        let channels = if shape.len() == 3 { shape[2] } else { shape[shape.len() - 1] }
        if self.channels > 0 and self.channels != channels {
            throw ValueError(f"GroupNorm2d expected {self.channels} channels, got {channels}")
        }
        self.build(channels, rng)
        self.last_shape = shape
    }

    pub fn output_shape(self) -> list[int] {
        if self.last_shape.len() > 0 { return self.last_shape }
        return [self.output_width()]
    }

    pub fn forward(self, x: Tensor) -> Tensor {
        let gamma = self.gamma_value
        let beta = self.beta_value
        if gamma is null or beta is null { throw RuntimeError("GroupNorm2d has not been built") }
        return group_norm(x, gamma, beta, self.groups)
    }

    pub fn parameters(self) -> list[Tensor] {
        let gamma = self.gamma_value
        let beta = self.beta_value
        if gamma is null or beta is null { throw RuntimeError("GroupNorm2d has not been built") }
        return [gamma, beta]
    }

    pub fn output_width(self) -> int {
        let gamma = self.gamma_value
        if gamma is null { return self.channels }
        return gamma.size
    }

    pub fn free(self) -> void {
        let gamma = self.gamma_value
        let beta = self.beta_value
        if gamma is not null { gamma.free() }
        if beta is not null { beta.free() }
        self.gamma_value = null
        self.beta_value = null
    }

    fn __str__(self) -> str { return f"GroupNorm2d(groups={self.groups}, channels={self.output_width()})" }
}

#: Layer normalization over the last axis of a rank-two tensor.
pub class LayerNorm: Layer {
    var width: int
    var gamma_value: Tensor?
    var beta_value: Tensor?

    pub fn init(self, width: int = 0) {
        self.width = width
        self.gamma_value = null
        self.beta_value = null
    }

    pub fn build(self, input_units: int, rng: Random) -> void {
        let _ = rng
        let size = if self.width > 0 { self.width } else { input_units }
        self.width = size
        if self.gamma_value is not null { return }
        self.gamma_value = Tensor.ones([size])
        self.gamma_value.requires_grad = true
        self.beta_value = Tensor.zeros([size])
        self.beta_value.requires_grad = true
    }

    pub fn forward(self, x: Tensor) -> Tensor {
        let gamma = self.gamma_value
        let beta = self.beta_value
        if gamma is null or beta is null { throw RuntimeError("LayerNorm has not been built") }
        return layer_norm(x, gamma, beta)
    }

    pub fn parameters(self) -> list[Tensor] {
        let gamma = self.gamma_value
        let beta = self.beta_value
        if gamma is null or beta is null { throw RuntimeError("LayerNorm has not been built") }
        return [gamma, beta]
    }

    pub fn output_width(self) -> int { return self.width }

    pub fn free(self) -> void {
        let gamma = self.gamma_value
        let beta = self.beta_value
        if gamma is not null { gamma.free() }
        if beta is not null { beta.free() }
        self.gamma_value = null
        self.beta_value = null
    }

    fn __str__(self) -> str { return f"LayerNorm({self.width})" }
}

#: Multi-head self-attention over rank-two `[seq, dim]` inputs.
pub class MultiHeadAttention: Layer {
    pub let heads: int
    pub let dim: int
    var wq_value: Tensor?
    var wk_value: Tensor?
    var wv_value: Tensor?
    var wo_value: Tensor?

    pub fn init(self, dim: int, heads: int = 1) {
        if dim <= 0 or heads <= 0 { throw ValueError("MultiHeadAttention dim/heads must be positive") }
        if dim % heads != 0 { throw ValueError("MultiHeadAttention dim must divide heads") }
        self.dim = dim
        self.heads = heads
        self.wq_value = null
        self.wk_value = null
        self.wv_value = null
        self.wo_value = null
    }

    pub fn build(self, input_units: int, rng: Random) -> void {
        if input_units != self.dim {
            throw ValueError(f"MultiHeadAttention expected width {self.dim}, got {input_units}")
        }
        if self.wq_value is not null { return }
        let limit = math.sqrt(3.0 / float(self.dim))
        self.wq_value = Tensor([self.dim, self.dim], requires_grad: true)
        self.wk_value = Tensor([self.dim, self.dim], requires_grad: true)
        self.wv_value = Tensor([self.dim, self.dim], requires_grad: true)
        self.wo_value = Tensor([self.dim, self.dim], requires_grad: true)
        self.wq_value.buffer().fill_uniform(-limit, limit, rng.next_int())
        self.wk_value.buffer().fill_uniform(-limit, limit, rng.next_int())
        self.wv_value.buffer().fill_uniform(-limit, limit, rng.next_int())
        self.wo_value.buffer().fill_uniform(-limit, limit, rng.next_int())
    }

    pub fn forward(self, x: Tensor) -> Tensor {
        let wq = self.wq_value
        let wk = self.wk_value
        let wv = self.wv_value
        let wo = self.wo_value
        if wq is null or wk is null or wv is null or wo is null {
            throw RuntimeError("MultiHeadAttention has not been built")
        }
        if x.rank() != 2 or x.shape[1] != self.dim {
            throw ValueError(f"MultiHeadAttention needs [seq, {self.dim}], got {x.shape}")
        }
        let q = matmul(x, wq)
        let k = matmul(x, wk)
        let v = matmul(x, wv)
        let context = multi_head_attention(q, k, v, self.heads)
        q.free()
        k.free()
        v.free()
        let output = matmul(context, wo)
        context.free()
        return output
    }

    pub fn parameters(self) -> list[Tensor] {
        let wq = self.wq_value
        let wk = self.wk_value
        let wv = self.wv_value
        let wo = self.wo_value
        if wq is null or wk is null or wv is null or wo is null {
            throw RuntimeError("MultiHeadAttention has not been built")
        }
        return [wq, wk, wv, wo]
    }

    pub fn output_width(self) -> int { return self.dim }

    pub fn free(self) -> void {
        let wq = self.wq_value
        let wk = self.wk_value
        let wv = self.wv_value
        let wo = self.wo_value
        if wq is not null { wq.free() }
        if wk is not null { wk.free() }
        if wv is not null { wv.free() }
        if wo is not null { wo.free() }
        self.wq_value = null
        self.wk_value = null
        self.wv_value = null
        self.wo_value = null
    }

    fn __str__(self) -> str { return f"MultiHeadAttention(dim={self.dim}, heads={self.heads})" }
}

#: Inverted dropout. Inactive during `Sequential.predict` (`training` false).
pub class Dropout: Layer {
    pub let p: float
    var width: int

    pub fn init(self, p: float = 0.5) {
        if p < 0.0 or p >= 1.0 { throw ValueError(f"Dropout p must be in [0, 1), got {p}") }
        self.p = p
        self.width = 0
    }

    pub fn build(self, input_units: int, rng: Random) -> void {
        let _ = rng
        self.width = input_units
    }

    pub fn forward(self, x: Tensor) -> Tensor { return dropout(x, self.p, training: true) }

    pub fn eval_forward(self, x: Tensor) -> Tensor { return dropout(x, self.p, training: false) }

    pub fn parameters(self) -> list[Tensor] { return [] }

    pub fn output_width(self) -> int { return self.width }

    pub fn free(self) -> void { }

    fn __str__(self) -> str { return f"Dropout({self.p})" }
}

#: Lookup table: integer indices into rows of a `[vocab, dim]` weight matrix.
pub class Embedding: Layer {
    pub let vocab: int
    pub let dim: int
    var table: Tensor?

    pub fn init(self, vocab: int, dim: int) {
        if vocab <= 0 or dim <= 0 { throw ValueError("Embedding vocab and dim must be positive") }
        self.vocab = vocab
        self.dim = dim
        self.table = null
    }

    pub fn build(self, input_units: int, rng: Random) -> void {
        let _ = input_units
        if self.table is not null { return }
        let table = Tensor([self.vocab, self.dim], requires_grad: true)
        table.buffer().fill_uniform(-0.05, 0.05, rng.next_int())
        self.table = table
    }

    pub fn forward(self, x: Tensor) -> Tensor {
        let table = self.table
        if table is null { throw RuntimeError("Embedding has not been built") }
        return gather_rows(table, x)
    }

    pub fn parameters(self) -> list[Tensor] {
        let table = self.table
        if table is null { throw RuntimeError("Embedding has not been built") }
        return [table]
    }

    pub fn output_width(self) -> int { return self.dim }

    pub fn free(self) -> void {
        let table = self.table
        if table is not null { table.free() }
        self.table = null
    }

    fn __str__(self) -> str { return f"Embedding({self.vocab}, {self.dim})" }
}

#: NHWC max pooling window.
pub class MaxPool2d: Layer {
    pub let kernel_size: int
    pub let stride: int
    pub let padding: int
    var last_shape: list[int]

    pub fn init(self, kernel_size: int, stride: int = 0, padding: int = 0) {
        if kernel_size <= 0 { throw ValueError("MaxPool2d kernel_size must be positive") }
        self.kernel_size = kernel_size
        self.stride = if stride <= 0 { kernel_size } else { stride }
        self.padding = padding
        self.last_shape = []
    }

    pub fn build(self, input_units: int, rng: Random) -> void {
        let _ = input_units
        let _ = rng
    }

    pub fn build_from_shape(self, shape: list[int], rng: Random) -> void {
        let _ = rng
        self.last_shape = [d for d in shape]
    }

    pub fn forward(self, x: Tensor) -> Tensor {
        let y = max_pool2d(x, self.kernel_size, stride: self.stride, padding: self.padding)
        self.last_shape = [y.shape[1], y.shape[2], y.shape[3]]
        return y
    }

    pub fn parameters(self) -> list[Tensor] { return [] }

    pub fn output_width(self) -> int {
        var n = 1
        for d in self.last_shape { n *= d }
        return n
    }

    pub fn output_shape(self) -> list[int] { return [d for d in self.last_shape] }

    pub fn free(self) -> void { }

    fn __str__(self) -> str { return f"MaxPool2d(k={self.kernel_size})" }
}

#: NHWC average pooling window.
pub class AvgPool2d: Layer {
    pub let kernel_size: int
    pub let stride: int
    pub let padding: int
    var last_shape: list[int]

    pub fn init(self, kernel_size: int, stride: int = 0, padding: int = 0) {
        if kernel_size <= 0 { throw ValueError("AvgPool2d kernel_size must be positive") }
        self.kernel_size = kernel_size
        self.stride = if stride <= 0 { kernel_size } else { stride }
        self.padding = padding
        self.last_shape = []
    }

    pub fn build(self, input_units: int, rng: Random) -> void {
        let _ = input_units
        let _ = rng
    }

    pub fn build_from_shape(self, shape: list[int], rng: Random) -> void {
        let _ = rng
        self.last_shape = [d for d in shape]
    }

    pub fn forward(self, x: Tensor) -> Tensor {
        let y = avg_pool2d(x, self.kernel_size, stride: self.stride, padding: self.padding)
        self.last_shape = [y.shape[1], y.shape[2], y.shape[3]]
        return y
    }

    pub fn parameters(self) -> list[Tensor] { return [] }

    pub fn output_width(self) -> int {
        var n = 1
        for d in self.last_shape { n *= d }
        return n
    }

    pub fn output_shape(self) -> list[int] { return [d for d in self.last_shape] }

    pub fn free(self) -> void { }

    fn __str__(self) -> str { return f"AvgPool2d(k={self.kernel_size})" }
}
