#: Neural-network layers.
import std.math
from std.gpu import Buffer
from std.random import Random
from .tensor import Tensor

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
pub class Dense {
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
        self.weights_value = Tensor([input_units, self.units], values)
        self.bias_value = Tensor.zeros([self.units])
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
