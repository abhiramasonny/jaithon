#: Neural-network layers.
import std.math
from std.gpu import Buffer
from std.random import Random
from .ops import dense_affine
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
        let values = [rng.float_range(-limit, limit) for _index in 0..input_units * self.units]
        self.weights_value = Tensor([input_units, self.units], values, requires_grad: true)
        self.bias_value = Tensor.zeros([self.units])
        self.bias_value.requires_grad = true
    }

    #: Run one batch through this layer's affine transform and activation.
    pub fn forward(self, x: Tensor) -> Tensor {
        let weights = self.weights_value
        let bias = self.bias_value
        if weights is null or bias is null { throw RuntimeError("Dense layer has not been built") }
        let affine = dense_affine(x, weights, bias, self.activation_code())
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
