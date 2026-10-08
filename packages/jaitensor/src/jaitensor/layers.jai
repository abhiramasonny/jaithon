#: Neural-network layers.
import std.math
from std.gpu import Buffer
from std.os import env
from std.random import Random
from .ops import conv2d, conv2d_out_dim, dense_affine, dense_static_backward, dense_static_forward
from .ops import dropout, gather_rows, gelu, layer_norm, leaky_relu, max_pool2d, avg_pool2d
from .ops import relu, reshape, sigmoid, softmax, tanh
from .ops import batch_norm, group_norm, multi_head_attention, silu, matmul
from .ops import add, copy_into, merge_heads, split_heads
from .ops.attention import hand_to_graph
from .ops.matmul import raw_matmul
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
    #: x
    Linear,
    #: max(0,x)
    Relu,
    #: Maps x through the logistic sigmoid function.
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
        return dense_static_backward(x, weights, bias, output, upstream, self.activation_code())
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
        return reshape(x, [batch, x.size // batch])
    }

    pub fn parameters(self) -> list[Tensor] { return [] }

    pub fn output_width(self) -> int { return self.width }

    pub fn free(self) -> void { }

    fn __str__(self) -> str { return f"Flatten({self.width})" }
}

#: Apply one of the `Activation` shapes to a tensor.
pub fn activate(x: Tensor, kind: Activation) -> Tensor {
    return match kind {
        Activation.Linear => x,
        Activation.Relu => relu(x),
        Activation.Sigmoid => sigmoid(x),
        Activation.Tanh => tanh(x),
        Activation.Softmax => softmax(x),
        Activation.Gelu => gelu(x),
        Activation.Silu => silu(x),
        Activation.LeakyRelu => leaky_relu(x),
    }
}

#: A standalone activation, for the places a layer has no `activation` of its
#: own — after a pool, a norm, or a residual add.
pub class Activate: Layer {
    #: Function this layer applies.
    pub let activation: Activation
    var last_shape: list[int]
    var width: int

    pub fn init(self, activation: Activation) {
        self.activation = activation
        self.last_shape = []
        self.width = 0
    }

    pub fn build(self, input_units: int, rng: Random) -> void {
        let _ = rng
        self.width = input_units
    }

    pub fn build_from_shape(self, shape: list[int], rng: Random) -> void {
        let _ = rng
        self.last_shape = [d for d in shape]
        var n = 1
        for dimension in shape { n *= dimension }
        self.width = n
    }

    pub fn forward(self, x: Tensor) -> Tensor { return activate(x, self.activation) }

    pub fn parameters(self) -> list[Tensor] { return [] }

    pub fn output_width(self) -> int { return self.width }

    pub fn output_shape(self) -> list[int] { return [d for d in self.last_shape] }

    pub fn wants_softmax_output(self) -> bool { return self.activation == Activation.Softmax }

    pub fn free(self) -> void { }

    fn __str__(self) -> str { return f"Activate({self.activation})" }
}

#: 2D convolution over NHWC tensors.
pub class Conv2d: Layer {
    pub let out_channels: int
    pub let kernel_size: int
    pub let stride: int
    pub let padding: int
    #: Function applied after the convolution.
    pub let activation: Activation
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
        in_channels: int = 0,
        activation: Activation = Activation.Linear
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
        self.activation = activation
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
        #: The same scheme `Dense` uses: He for a ReLU layer, unit-variance
        #: otherwise. The old form divided by `fan + out_channels`, which mixes
        #: a per-position fan-in with a channel count — the fan-out of a
        #: convolution is `kernel * kernel * out_channels`, so the limit came
        #: out too large and a stack of convolutions started well past the
        #: stable region.
        let limit = math.sqrt(if self.activation == Activation.Relu {
            6.0 / float(fan)
        } else {
            3.0 / float(fan)
        })
        self.weights_value = Tensor(
            [
                self.kernel_size,
                self.kernel_size,
                self.in_channels,
                self.out_channels,
            ],
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
            input = reshape(x, [x.shape[0], self.in_height, self.in_width, self.in_channels])
        }
        #: ReLU folds into the convolution's own dispatch (see `ops.conv2d`),
        #: so the common case never launches a second elementwise kernel over
        #: the feature map. Every other activation still runs as its own op.
        if self.activation == Activation.Relu {
            return conv2d(
                input,
                weights,
                bias: bias,
                stride: self.stride,
                padding: self.padding,
                activation: 1
            )
        }
        let output = conv2d(input, weights, bias: bias, stride: self.stride, padding: self.padding)
        return activate(output, self.activation)
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

    pub fn wants_softmax_output(self) -> bool { return self.activation == Activation.Softmax }

    fn __str__(self) -> str {
        return f"Conv2d({self.in_channels}->{self.out_channels}, k={self.kernel_size})"
    }
}

#: Per-channel batch normalization over NHWC activations.
#:
#: This normalises by the statistics of whatever batch it is handed, and it
#: does that when predicting as well as when training. It keeps no running
#: average, which is a real difference from what other libraries do and is
#: worth knowing before relying on it:
#:
#:   - a prediction depends on which other samples share the call, so the
#:     same input can come back different in a batch of one and a batch of
#:     thirty-two;
#:   - a batch of one whose spatial extent is also one has nothing to take a
#:     spread over, and every channel comes back as `beta`.
#:
#: Adding the running average is not a small change -- it is another two
#: tensors of state, a momentum to choose, and a different answer from every
#: model already trained against this one -- so it is written down here
#: rather than done quietly.
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

    fn __str__(self) -> str {
        return f"GroupNorm2d(groups={self.groups}, channels={self.output_width()})"
    }
}

#: Layer normalization over the last axis of a tensor of rank two or more;
#: leading axes fold into rows, so `[batch, seq, dim]` works as it is.
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

#: The product epilogue that adds `residual`, or none without one: 6 is
#: `linear_residual`'s.
fn epilogue_for(residual: Tensor?) -> int { return if residual is null { 0 } else { 6 } }

#: Multi-head self-attention over `[seq, dim]` or batched `[batch, seq, dim]`
#: inputs.
#:
#: `causal` lets each position attend only to itself and the positions before
#: it, which is what an autoregressive model trains with.
#:
#: A rank-three input is a batch of independent sequences. The projections
#: run as one product over all `batch * seq` rows and attention folds the
#: batch into the head axis, so a batch costs the same number of dispatches
#: as one sequence; nothing attends across sequences. Rank two is one
#: sequence, exactly as before, which is what `Sequential` hands the layer.
#:
#: `batch_first: false` reads rank three as `[seq, batch, dim]` instead, the
#: order torch's `nn.MultiheadAttention` defaults to. That order folds for
#: free: `[seq, batch * dim]` already is the packed layout of `batch * heads`
#: heads, so attention reads the projections where they lie. Batch-first has
#: to copy each of Q, K and V into it and the context back out, which at a
#: GPT's `[64, 128, 256]` is 0.57 ms of a 1.78 ms forward and backward. A model
#: whose every other op works row by row -- layer norm, the MLP, the residual
#: adds -- can keep its activations sequence-first and never pay it.
#:
#: For generation, `start_cache(max_seq)` keeps every key and value the layer
#: has projected in device buffers of `max_seq` rows. Each `forward` then
#: takes only the new rows -- a prompt, then one token at a time -- projects
#: them straight into the next rows of the cache, and attends them against
#: everything cached so far, so a token costs the same at position 500 as at
#: position 5 instead of recomputing the whole prefix. The cached forward
#: records no gradient. `reset_cache` starts a new sequence in the same
#: buffers; `free_cache` (or `free`) releases them. `start_cache(max_seq,
#: batch: n)` keeps `n` sequences that advance together, fed as
#: `[n, rows, dim]`.
pub class MultiHeadAttention: Layer {
    pub let heads: int
    pub let dim: int
    pub let causal: bool
    #: Whether a rank-three input is `[batch, seq, dim]` (true) or
    #: `[seq, batch, dim]` (false).
    pub let batch_first: bool
    var wq_value: Tensor?
    var wk_value: Tensor?
    var wv_value: Tensor?
    var wo_value: Tensor?
    var cache_k: Tensor?
    var cache_v: Tensor?
    var cached: int
    var cache_batch: int

    pub fn init(self, dim: int, heads: int = 1, causal: bool = false, batch_first: bool = true) {
        if dim <= 0 or heads <= 0 {
            throw ValueError("MultiHeadAttention dim/heads must be positive")
        }
        if dim % heads != 0 { throw ValueError("MultiHeadAttention dim must divide heads") }
        self.dim = dim
        self.heads = heads
        self.causal = causal
        self.batch_first = batch_first
        self.wq_value = null
        self.wk_value = null
        self.wv_value = null
        self.wo_value = null
        self.cache_k = null
        self.cache_v = null
        self.cached = 0
        self.cache_batch = 1
    }

    #: Keep keys and values across `forward` calls, for up to `max_seq` rows
    #: of each of `batch` sequences. Replaces any cache already held, and
    #: starts empty.
    #:
    #: The cache is position-major, `[max_seq, batch * dim]`: row `s` holds
    #: every sequence's key at position `s`, side by side. That is the layout
    #: batched attention folds a batch into anyway, so a decode step attends
    #: the cache where it lies, and one new row a sequence -- `[batch, dim]`
    #: -- is already exactly one row of it. For one sequence it is the plain
    #: `[max_seq, dim]`.
    pub fn start_cache(self, max_seq: int, batch: int = 1) -> void {
        if max_seq <= 0 { throw ValueError(f"start_cache needs a positive length, got {max_seq}") }
        if batch <= 0 { throw ValueError(f"start_cache needs a positive batch, got {batch}") }
        self.free_cache()
        self.cache_k = Tensor([max_seq, batch * self.dim])
        self.cache_v = Tensor([max_seq, batch * self.dim])
        self.cache_batch = batch
        self.cached = 0
    }

    #: How many sequences the cache holds side by side.
    pub fn cache_batch_size(self) -> int { return self.cache_batch }

    #: Forget the cached rows but keep the buffers, for the next sequence.
    #: Nothing is written: the rows are simply overwritten as they refill.
    pub fn reset_cache(self) -> void { self.cached = 0 }

    #: How many rows the cache holds.
    pub fn cache_length(self) -> int { return self.cached }

    #: Release the cache's buffers; `forward` goes back to whole sequences.
    pub fn free_cache(self) -> void {
        let k = self.cache_k
        let v = self.cache_v
        if k is not null { k.free() }
        if v is not null { v.free() }
        self.cache_k = null
        self.cache_v = null
        self.cached = 0
        self.cache_batch = 1
    }

    #: `forward` with the cache on: `x` holds only the rows not seen yet.
    #: A `residual` is added in the output product's epilogue.
    fn forward_cached(
        self,
        x: Tensor,
        wq: Tensor,
        wk: Tensor,
        wv: Tensor,
        wo: Tensor,
        keys: Tensor,
        values: Tensor,
        residual: Tensor? = null
    ) -> Tensor {
        let rows = x.shape[0]
        let capacity = keys.shape[0]
        if self.cached + rows > capacity {
            throw ValueError(
                f"MultiHeadAttention cache holds {capacity} rows; {self.cached} are used and {rows} more were given"
            )
        }
        let start = self.cached * self.dim
        #: The new keys and values are written by the projection itself into
        #: the next rows of the cache: device work only, no copy and no host
        #: write, so nothing here needs a fence against the buffer pool.
        let k_rows = keys.view(start, [rows, self.dim])
        let v_rows = values.view(start, [rows, self.dim])
        raw_matmul(x, wk, out: k_rows)
        raw_matmul(x, wv, out: v_rows)
        k_rows.free()
        v_rows.free()
        let q = raw_matmul(x, wq)
        let seen = self.cached + rows
        let k_seen = keys.view(0, [seen, self.dim])
        let v_seen = values.view(0, [seen, self.dim])
        let context = multi_head_attention(q, k_seen, v_seen, self.heads, causal: self.causal)
        k_seen.free()
        v_seen.free()
        q.free()
        let output = raw_matmul(context, wo, bias: residual, epilogue: epilogue_for(residual))
        context.free()
        self.cached = seen
        return output
    }

    #: The cached forward for `[batch, rows, dim]`, one batch of sequences
    #: advancing together. Records no gradient, like the rank-two one.
    fn forward_cached_batched(
        self,
        x: Tensor,
        wq: Tensor,
        wk: Tensor,
        wv: Tensor,
        wo: Tensor,
        keys: Tensor,
        values: Tensor
    ) -> Tensor {
        let batch = x.shape[0]
        let rows = x.shape[1]
        if batch != self.cache_batch {
            throw ValueError(
                f"MultiHeadAttention cache holds {self.cache_batch} sequences, got a batch of {batch}"
            )
        }
        if batch == 1 {
            #: One sequence is the rank-two cache exactly; the folds below
            #: would only copy it.
            let single = x.view(0, [rows, self.dim])
            let out = self.forward_cached(single, wq, wk, wv, wo, keys, values)
            single.free()
            return Tensor([1, rows, self.dim], storage: out.buffer())
        }
        let capacity = keys.shape[0]
        if self.cached + rows > capacity {
            throw ValueError(
                f"MultiHeadAttention cache holds {capacity} rows; {self.cached} are used and {rows} more were given"
            )
        }
        let width = batch * self.dim
        let start = self.cached * width
        let seen = self.cached + rows
        let flat = x.view(0, [batch * rows, self.dim])
        var query = flat
        if rows == 1 {
            #: `[batch, dim]` is already the next cache row, `[1, batch * dim]`,
            #: so the projections write it in place, as the rank-two path does.
            let k_rows = keys.view(start, [batch, self.dim])
            let v_rows = values.view(start, [batch, self.dim])
            raw_matmul(flat, wk, out: k_rows)
            raw_matmul(flat, wv, out: v_rows)
            k_rows.free()
            v_rows.free()
            let q = raw_matmul(flat, wq)
            query = Tensor([1, width], storage: q.buffer())
        } else {
            #: A prompt: `[batch * rows, dim]` is sequence-major, the cache is
            #: position-major, and `merge_heads` with one "head" a sequence is
            #: precisely that transpose.
            for (weight, cache) in [(wk, keys), (wv, values)] {
                let projected = raw_matmul(flat, weight)
                let folded = merge_heads(projected, batch, rows)
                projected.free()
                let target = cache.view(start, [rows, width])
                copy_into(folded, target)
                target.free()
                folded.free()
            }
            let q = raw_matmul(flat, wq)
            query = merge_heads(q, batch, rows)
            q.free()
        }
        flat.free()
        let k_seen = keys.view(0, [seen, width])
        let v_seen = values.view(0, [seen, width])
        let context = multi_head_attention(
            query,
            k_seen,
            v_seen,
            self.heads * batch,
            causal: self.causal
        )
        k_seen.free()
        v_seen.free()
        query.free()
        let per_row = if rows == 1 {
            Tensor([batch, self.dim], storage: context.buffer())
        } else {
            split_heads(context, batch)
        }
        if rows != 1 { context.free() }
        let output = raw_matmul(per_row, wo)
        per_row.free()
        self.cached = seen
        return Tensor([batch, rows, self.dim], storage: output.buffer())
    }

    #: The cached forward for sequence-first `[rows, batch, dim]`. The new rows
    #: of every sequence are already the next `rows` rows of the
    #: position-major cache, whatever `rows` is, so the projections write them
    #: in place and nothing is folded.
    fn forward_cached_seq_first(
        self,
        x: Tensor,
        wq: Tensor,
        wk: Tensor,
        wv: Tensor,
        wo: Tensor,
        keys: Tensor,
        values: Tensor,
        residual: Tensor? = null
    ) -> Tensor {
        let rows = x.shape[0]
        let batch = x.shape[1]
        if batch != self.cache_batch {
            throw ValueError(
                f"MultiHeadAttention cache holds {self.cache_batch} sequences, got a batch of {batch}"
            )
        }
        let capacity = keys.shape[0]
        if self.cached + rows > capacity {
            throw ValueError(
                f"MultiHeadAttention cache holds {capacity} rows; {self.cached} are used and {rows} more were given"
            )
        }
        let width = batch * self.dim
        let start = self.cached * width
        let seen = self.cached + rows
        let flat = x.view(0, [rows * batch, self.dim])
        let k_rows = keys.view(start, [rows * batch, self.dim])
        let v_rows = values.view(start, [rows * batch, self.dim])
        raw_matmul(flat, wk, out: k_rows)
        raw_matmul(flat, wv, out: v_rows)
        k_rows.free()
        v_rows.free()
        let q = raw_matmul(flat, wq)
        flat.free()
        let query = Tensor([rows, width], storage: q.buffer())
        let k_seen = keys.view(0, [seen, width])
        let v_seen = values.view(0, [seen, width])
        let context = multi_head_attention(
            query,
            k_seen,
            v_seen,
            self.heads * batch,
            causal: self.causal
        )
        k_seen.free()
        v_seen.free()
        query.free()
        let per_row = Tensor([rows * batch, self.dim], storage: context.buffer())
        let output = raw_matmul(per_row, wo, bias: residual, epilogue: epilogue_for(residual))
        per_row.free()
        self.cached = seen
        return Tensor([rows, batch, self.dim], storage: output.buffer())
    }

    #: The whole-sequence forward for sequence-first `[seq, batch, dim]`:
    #: the rank-two path with `batch * heads` heads, and no copies.
    fn forward_seq_first(
        self,
        x: Tensor,
        wq: Tensor,
        wk: Tensor,
        wv: Tensor,
        wo: Tensor
    ) -> Tensor {
        let seq = x.shape[0]
        let batch = x.shape[1]
        let width = batch * self.dim
        let flat = reshape(x, [seq * batch, self.dim])
        let q = matmul(flat, wq)
        let k = matmul(flat, wk)
        let v = matmul(flat, wv)
        let q2 = reshape(q, [seq, width])
        let k2 = reshape(k, [seq, width])
        let v2 = reshape(v, [seq, width])
        let context = multi_head_attention(q2, k2, v2, self.heads * batch, causal: self.causal)
        hand_to_graph(context, [q2, k2, v2, q, k, v])
        let per_row = reshape(context, [seq * batch, self.dim])
        let product = matmul(per_row, wo)
        if product.creator is null {
            for held in [flat, per_row, context] { held.free() }
            return Tensor([seq, batch, self.dim], storage: product.buffer())
        }
        let output = reshape(product, [seq, batch, self.dim])
        hand_to_graph(output, [flat, per_row, context])
        return output
    }

    #: The whole-sequence forward for `[batch, seq, dim]`.
    #:
    #: Ownership follows the rank-two path: whatever a recorded graph reads
    #: rides on it, through `hand_to_graph`, and what nothing recorded is
    #: freed here. `flat` is read by the projections' backward and `per_row`
    #: by the output product's, so they go to the graph with `context`; the
    #: unrecorded result adopts the product's buffer rather than viewing it.
    fn forward_batched(self, x: Tensor, wq: Tensor, wk: Tensor, wv: Tensor, wo: Tensor) -> Tensor {
        let batch = x.shape[0]
        let seq = x.shape[1]
        let flat = reshape(x, [batch * seq, self.dim])
        let q = matmul(flat, wq)
        let k = matmul(flat, wk)
        let v = matmul(flat, wv)
        let q3 = reshape(q, [batch, seq, self.dim])
        let k3 = reshape(k, [batch, seq, self.dim])
        let v3 = reshape(v, [batch, seq, self.dim])
        let context = multi_head_attention(q3, k3, v3, self.heads, causal: self.causal)
        hand_to_graph(context, [q3, k3, v3, q, k, v])
        let per_row = reshape(context, [batch * seq, self.dim])
        let product = matmul(per_row, wo)
        if product.creator is null {
            for held in [flat, per_row, context] { held.free() }
            return Tensor([batch, seq, self.dim], storage: product.buffer())
        }
        let output = reshape(product, [batch, seq, self.dim])
        hand_to_graph(output, [flat, per_row, context])
        return output
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
        let keys = self.cache_k
        let values = self.cache_v
        if x.rank() == 3 and x.shape[2] == self.dim {
            if not self.batch_first {
                if keys is not null and values is not null {
                    return self.forward_cached_seq_first(x, wq, wk, wv, wo, keys, values)
                }
                return self.forward_seq_first(x, wq, wk, wv, wo)
            }
            if keys is not null and values is not null {
                return self.forward_cached_batched(x, wq, wk, wv, wo, keys, values)
            }
            return self.forward_batched(x, wq, wk, wv, wo)
        }
        if x.rank() != 2 or x.shape[1] != self.dim {
            throw ValueError(
                f"MultiHeadAttention needs [seq, {self.dim}] or a rank-three batch of them, got {x.shape}"
            )
        }
        if keys is not null and values is not null {
            if self.cache_batch != 1 {
                throw ValueError(
                    f"MultiHeadAttention cache holds {self.cache_batch} sequences; feed it a batch of them"
                )
            }
            return self.forward_cached(x, wq, wk, wv, wo, keys, values)
        }
        let q = matmul(x, wq)
        let k = matmul(x, wk)
        let v = matmul(x, wv)
        let context = multi_head_attention(q, k, v, self.heads, causal: self.causal)
        #: Attention's backward reads q, k and v, and the output product's
        #: backward reads `context`, so freeing them here made the first
        #: `backward()` of any model holding this layer throw. Whether one is
        #: still read depends on its consumer, not on itself: a frozen
        #: projection records nothing, yet attention's graph reads it as soon
        #: as any other input trains. `hand_to_graph` gives each to the graph
        #: that reads it, for `backward()` or `discard_graph()` to return, and
        #: frees only what no graph reads.
        hand_to_graph(context, [q, k, v])
        let output = matmul(context, wo)
        hand_to_graph(output, [context])
        return output
    }

    #: `residual + forward(x)`, the first half of a transformer block.
    #:
    #: With the cache on, the add rides on the output product's epilogue, the
    #: way `linear_residual` does it, so a decoded token pays one dispatch a
    #: layer less. Anywhere else it is `forward` and then `add`, recorded as
    #: usual. `residual` is the output's shape.
    pub fn forward_add(self, x: Tensor, residual: Tensor) -> Tensor {
        if residual.size != x.size {
            throw ValueError(
                f"forward_add residual {residual.shape} does not match input {x.shape}"
            )
        }
        let wq = self.wq_value
        let wk = self.wk_value
        let wv = self.wv_value
        let wo = self.wo_value
        let keys = self.cache_k
        let values = self.cache_v
        if wq is not null and wk is not null and wv is not null and wo is not null {
            if keys is not null and values is not null {
                if x.rank() == 2 and x.shape[1] == self.dim and self.cache_batch == 1 {
                    return self.forward_cached(x, wq, wk, wv, wo, keys, values, residual: residual)
                }
                if x.rank() == 3 and x.shape[2] == self.dim and not self.batch_first {
                    return self.forward_cached_seq_first(
                        x,
                        wq,
                        wk,
                        wv,
                        wo,
                        keys,
                        values,
                        residual: residual
                    )
                }
            }
        }
        let y = self.forward(x)
        let output = add(residual, y)
        hand_to_graph(output, [y])
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
        self.free_cache()
    }

    fn __str__(self) -> str {
        let mask = if self.causal { ", causal" } else { "" }
        let order = if self.batch_first { "" } else { ", seq first" }
        return f"MultiHeadAttention(dim={self.dim}, heads={self.heads}{mask}{order})"
    }
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
    #: How many indices a sample carries, which the layer learns when it is
    #: built. One means a single index a sample, which is the rank-one case.
    var span: int
    var table: Tensor?

    pub fn init(self, vocab: int, dim: int) {
        if vocab <= 0 or dim <= 0 { throw ValueError("Embedding vocab and dim must be positive") }
        self.vocab = vocab
        self.dim = dim
        self.span = 1
        self.table = null
    }

    pub fn build(self, input_units: int, rng: Random) -> void {
        self.span = max(1, input_units)
        if self.table is not null { return }
        let table = Tensor([self.vocab, self.dim], requires_grad: true)
        table.buffer().fill_uniform(-0.05, 0.05, rng.next_int())
        self.table = table
    }

    #: A batch of index rows, not only a single row of them.
    #:
    #: `gather_rows` wants the indices flat, so a `[batch, span]` block is
    #: read as `batch * span` lookups and the result put back as
    #: `[batch, span * dim]` -- which is the width the layer after this one
    #: was sized for. Without this the layer could not be used in a model
    #: that trains in batches at all, which is every model here.
    pub fn forward(self, x: Tensor) -> Tensor {
        let table = self.table
        if table is null { throw RuntimeError("Embedding has not been built") }
        if x.rank() < 2 { return gather_rows(table, x) }
        if x.rank() > 2 {
            throw ValueError(f"Embedding takes indices of rank one or two, got {x.shape}")
        }
        return gather_rows(table, x, [x.shape[0], x.shape[1] * self.dim])
    }

    #: One row of the table per index, keeping the indices' shape:
    #: `[batch, seq]` gives `[batch, seq, dim]`, `[n]` gives `[n, dim]`.
    #:
    #: `forward` is what a `Sequential` wants, a `[batch, seq * dim]` block
    #: the next layer was sized for; a transformer wants the rows kept apart,
    #: which is this. Both are the same gather, read under a different shape,
    #: and the gradient reaches the table either way.
    pub fn lookup(self, indices: Tensor) -> Tensor {
        let table = self.table
        if table is null { throw RuntimeError("Embedding has not been built") }
        var shape = [dimension for dimension in indices.shape]
        shape.push(self.dim)
        return gather_rows(table, indices, shape)
    }

    pub fn parameters(self) -> list[Tensor] {
        let table = self.table
        if table is null { throw RuntimeError("Embedding has not been built") }
        return [table]
    }

    pub fn output_width(self) -> int { return self.span * self.dim }

    pub fn free(self) -> void {
        let table = self.table
        if table is not null { table.free() }
        self.table = null
    }

    fn __str__(self) -> str { return f"Embedding({self.vocab}, {self.dim})" }
}

var _position_view: int = -1

fn position_view_on() -> bool {
    if _position_view < 0 {
        let text = env("JAITENSOR_POSITION_VIEW")
        _position_view = if text is not null and text == "0" { 0 } else { 1 }
    }
    return _position_view == 1
}

#: Learned positional embeddings: row `p` of a `[max_seq, dim]` table is added
#: to every sequence's position `p`.
#:
#: `forward` takes `[seq, dim]` or `[batch, seq, dim]` activations and adds
#: positions `0..seq`; `forward_at(x, start)` adds `start..start + seq`, which
#: is what a cached decoder feeding one token at a time needs. Only the `seq`
#: rows in use are gathered and they broadcast over the batch, so the
#: gradient reaches the table as one sum over the batch rather than `batch`
#: colliding atomic adds per row. `batch_first: false` takes `[seq, batch,
#: dim]`, to sit beside a sequence-first `MultiHeadAttention`; there the rows
#: are gathered once per sequence, since the batch is not a leading axis.
pub class PositionalEmbedding: Layer {
    pub let max_seq: int
    pub let dim: int
    #: Half-width of the uniform initialisation `build` draws from.
    pub let init_limit: float
    #: Whether a rank-three input is `[batch, seq, dim]` (true) or
    #: `[seq, batch, dim]` (false).
    pub let batch_first: bool
    var table: Tensor?
    var index: Tensor?
    var index_start: int
    var index_repeat: int

    pub fn init(self, max_seq: int, dim: int, init_limit: float = 0.05, batch_first: bool = true) {
        if max_seq <= 0 or dim <= 0 {
            throw ValueError("PositionalEmbedding max_seq and dim must be positive")
        }
        self.max_seq = max_seq
        self.dim = dim
        self.init_limit = init_limit
        self.batch_first = batch_first
        self.table = null
        self.index = null
        self.index_start = -1
        self.index_repeat = 0
    }

    pub fn build(self, input_units: int, rng: Random) -> void {
        let _ = input_units
        if self.table is not null { return }
        let table = Tensor([self.max_seq, self.dim], requires_grad: true)
        table.buffer().fill_uniform(-self.init_limit, self.init_limit, rng.next_int())
        self.table = table
    }

    #: The table's rows `start..start + seq` as `[seq, dim]`, recorded for the
    #: gradient when the table trains. When it does not -- generation -- the
    #: rows are contiguous, so this is a view of them: no index upload and no
    #: gather for every decoded token. Free it either way; a view's free
    #: leaves the table alone. `JAITENSOR_POSITION_VIEW=0` gathers anyway, for
    #: an A/B in one binary; read once.
    pub fn positions(self, seq: int, start: int = 0) -> Tensor {
        let table = self.table
        if table is null { throw RuntimeError("PositionalEmbedding has not been built") }
        if seq <= 0 or start < 0 or start + seq > self.max_seq {
            throw ValueError(
                f"positions {start}..{start + seq} run past the {self.max_seq} this table holds"
            )
        }
        if not table.requires_grad and position_view_on() {
            return table.view(start * self.dim, [seq, self.dim])
        }
        return self.gather_positions(table, seq, start, 1, null)
    }

    #: `x` plus positions `0..seq`.
    pub fn forward(self, x: Tensor) -> Tensor { return self.forward_at(x, 0) }

    #: `x` plus positions `start..start + seq`, for `[seq, dim]` or a
    #: rank-three batch of them.
    pub fn forward_at(self, x: Tensor, start: int) -> Tensor {
        if x.rank() < 2 or x.rank() > 3 or x.shape[x.rank() - 1] != self.dim {
            throw ValueError(
                f"PositionalEmbedding needs [seq, {self.dim}] or a rank-three batch of them, got {x.shape}"
            )
        }
        if x.rank() == 3 and not self.batch_first {
            let table = self.table
            if table is null { throw RuntimeError("PositionalEmbedding has not been built") }
            let seq = x.shape[0]
            if start < 0 or start + seq > self.max_seq {
                throw ValueError(
                    f"positions {start}..{start + seq} run past the {self.max_seq} this table holds"
                )
            }
            let rows = self.gather_positions(table, seq, start, x.shape[1], x.shape)
            let output = add(x, rows)
            hand_to_graph(output, [rows])
            return output
        }
        let rows = self.positions(x.shape[x.rank() - 2], start)
        let output = add(x, rows)
        hand_to_graph(output, [rows])
        return output
    }

    #: The table's rows at positions `start..start + seq`, each `repeat`
    #: times over, shaped `shape`.
    #:
    #: A gather that records reads its index again in the backward, which may
    #: come after several more forwards. Those can ask for another length or
    #: offset and replace the cached index, so a recorded gather reads a
    #: device copy of it instead, which the graph owns and frees with its other
    #: intermediates. Freeing the cached index for another then leaves every
    #: graph its own: two forwards of different lengths before one backward
    #: used to throw that the index had already been freed. A gather that does
    #: not record -- generation -- reads the cached index directly.
    fn gather_positions(
        self,
        table: Tensor,
        seq: int,
        start: int,
        repeat: int,
        shape: list[int]?
    ) -> Tensor {
        let cached = self.indices(seq, start, repeat)
        if not table.requires_grad { return gather_rows(table, cached, shape) }
        let owned = Tensor([cached.size])
        copy_into(cached, owned)
        let rows = gather_rows(table, owned, shape)
        let node = rows.creator
        if node is not null { node.extras.push(owned) } else { owned.free() }
        return rows
    }

    #: The index tensor for positions `start..start + seq`, each `repeat`
    #: times over, kept between calls: training asks for the same one every
    #: step. Only work already queued reads it -- a gather that did not record,
    #: or the copy a recorded one took -- so a different one is a new tensor,
    #: not an upload over the old, and the old one can go.
    fn indices(self, seq: int, start: int, repeat: int) -> Tensor {
        let held = self.index
        if held is not null and held.size == seq * repeat and self.index_start == start {
            if self.index_repeat == repeat { return held }
        }
        if held is not null { held.free() }
        var values: list[float] = []
        for offset in 0..seq {
            for _copy in 0..repeat { values.push(float(start + offset)) }
        }
        let made = Tensor([seq * repeat], values)
        self.index = made
        self.index_start = start
        self.index_repeat = repeat
        return made
    }

    pub fn parameters(self) -> list[Tensor] {
        let table = self.table
        if table is null { throw RuntimeError("PositionalEmbedding has not been built") }
        return [table]
    }

    pub fn output_width(self) -> int { return self.dim }

    pub fn free(self) -> void {
        let table = self.table
        if table is not null { table.free() }
        let held = self.index
        if held is not null { held.free() }
        self.table = null
        self.index = null
        self.index_start = -1
        self.index_repeat = 0
    }

    fn __str__(self) -> str { return f"PositionalEmbedding({self.max_seq}, {self.dim})" }
}

#: NHWC max pooling window.
#: What a pooling window leaves of an NHWC `[height, width, channels]` shape.
#:
#: `Sequential` builds each layer from the one before it reports, so a pool that
#: reported its input shape sized every later layer as if it had not pooled at
#: all — a `Conv2d`/`MaxPool2d` stack failed on its first forward with a matmul
#: shape error rather than at build time.
fn pooled_shape(shape: list[int], kernel_size: int, stride: int, padding: int) -> list[int] {
    if shape.len() != 3 { return [d for d in shape] }
    return [
        conv2d_out_dim(shape[0], kernel_size, stride, padding),
        conv2d_out_dim(shape[1], kernel_size, stride, padding),
        shape[2],
    ]
}

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
        self.last_shape = pooled_shape(shape, self.kernel_size, self.stride, self.padding)
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
        self.last_shape = pooled_shape(shape, self.kernel_size, self.stride, self.padding)
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
