#: Optimizers used by `Sequential.compile`.
import std.gpu as gpu
from std.gpu import Buffer, Kernel
from .tensor import Tensor

let SOURCE = """#include <metal_stdlib>
using namespace metal;

kernel void parameter_update(device float* value [[buffer(0)]],
                             device const float* gradient [[buffer(1)]],
                             device float* first_moment [[buffer(2)]],
                             device float* second_moment [[buffer(3)]],
                             device const float* settings [[buffer(4)]],
                             constant uint& count [[buffer(5)]],
                             constant uint& kind [[buffer(6)]],
                             constant uint& step [[buffer(7)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float g = gradient[id];
    if (kind == 0) {
        float velocity = settings[1] * first_moment[id] + g;
        first_moment[id] = velocity;
        float update = settings[2] > 0.5f ? settings[1] * velocity + g : velocity;
        value[id] -= settings[0] * update;
    } else {
        float first = settings[1] * first_moment[id] + (1.0f - settings[1]) * g;
        float second = settings[2] * second_moment[id] + (1.0f - settings[2]) * g * g;
        first_moment[id] = first;
        second_moment[id] = second;
        float corrected_first = first / (1.0f - pow(settings[1], float(step)));
        float corrected_second = second / (1.0f - pow(settings[2], float(step)));
        value[id] -= settings[0] * corrected_first / (sqrt(corrected_second) + settings[3]);
    }
}

kernel void parameter_update_sgd(device float* value [[buffer(0)]],
                                 device const float* gradient [[buffer(1)]],
                                 device const float* settings [[buffer(2)]],
                                 constant uint& count [[buffer(3)]],
                                 uint id [[thread_position_in_grid]]) {
    if (id < count) value[id] -= settings[0] * gradient[id];
}

kernel void parameter_update_sgd4(device float* value [[buffer(0)]],
                                  device const float* gradient [[buffer(1)]],
                                  device const float* settings [[buffer(2)]],
                                  constant uint& count [[buffer(3)]],
                                  uint id [[thread_position_in_grid]]) {
    uint i = id * 4;
    if (i >= count) return;
    *((device float4*)(value + i)) -= settings[0] * *((device const float4*)(gradient + i));
}
"""

class ParameterState {
    pub let first_moment: Buffer
    pub let second_moment: Buffer

    pub var settings: Buffer?
    #: What was last written into `settings`, so that a step asking for the
    #: same numbers again need not write them again.
    pub var written: list[float]

    pub fn init(self, size: int) {
        self.first_moment = Buffer.of([0.0 for _index in 0..size])
        self.second_moment = Buffer.of([0.0 for _index in 0..size])
        self.settings = null
        self.written = []
    }

    pub fn free(self) -> void {
        self.first_moment.free()
        self.second_moment.free()
        let settings = self.settings
        if settings is not null { settings.free() }
    }
}

var _states: dict[int, ParameterState] = {}
var _steps: dict[int, int] = {}
var _update_kernel: Kernel? = null
var _sgd_kernel: Kernel? = null
var _sgd4_kernel: Kernel? = null

#: Clear every given parameter's optimizer state (momentum, second moment,
#: step count).
#:
#: `Sequential.compile` calls this so recompiling an already-built model
#: with a different optimizer starts clean, instead of silently
#: reinterpreting the previous optimizer's state as its own — `_states`/
#: `_steps` are keyed by `Tensor.id`, which doesn't change across a
#: recompile, so nothing else would ever reset them.
pub fn reset_state(parameters: list[Tensor]) -> void {
    for parameter in parameters {
        let state = _states.get(parameter.id, null)
        if state is not null { state.free() }
        _states.remove(parameter.id)
        _steps.remove(parameter.id)
    }
}

fn kernel_for_update() -> Kernel {
    let existing = _update_kernel
    if existing is not null { return existing }
    let kernel = Kernel.compile(SOURCE, "parameter_update")
    _update_kernel = kernel
    return kernel
}

fn kernel_for_sgd() -> Kernel {
    let existing = _sgd_kernel
    if existing is not null { return existing }
    let kernel = Kernel.compile(SOURCE, "parameter_update_sgd")
    _sgd_kernel = kernel
    return kernel
}

fn kernel_for_sgd4() -> Kernel {
    let existing = _sgd4_kernel
    if existing is not null { return existing }
    let kernel = Kernel.compile(SOURCE, "parameter_update_sgd4")
    _sgd4_kernel = kernel
    return kernel
}

#: Compile optimizer kernels before the first timed training step.
pub fn warmup_kernels() -> void {
    let _kernel = kernel_for_update()
    let _sgd = kernel_for_sgd()
    let _sgd4 = kernel_for_sgd4()
}

fn state_for(parameter: Tensor) -> ParameterState {
    let key = parameter.id
    let existing = _states.get(key, null)
    if existing is not null { return existing }
    let state = ParameterState(parameter.size)
    _states[key] = state
    return state
}

fn next_step(parameter: Tensor) -> int {
    let key = parameter.id
    let previous = _steps.get(key, 0)
    let updated = previous + 1
    _steps[key] = updated
    return updated
}

#: Allocate optimizer buffers before the first timed step.
pub fn prepare_state(parameters: list[Tensor], settings: list[float]) -> void {
    for parameter in parameters {
        let state = state_for(parameter)
        if state.settings is null { state.settings = Buffer.of(settings) }
    }
}

#: Clear every parameter's accumulated gradient, in place.
#:
#: `Optimizer.step` reads `.grad` and leaves it alone, so a training loop has
#: to reset it between batches — otherwise the batches sum and the effective
#: step grows without bound. Zeroing rather than dropping keeps the buffer, so
#: the backward pass that follows writes into memory it already has.
pub fn zero_grad(parameters: list[Tensor]) -> void {
    for parameter in parameters {
        let gradient = parameter.grad
        if gradient is not null { gradient.zero_() }
    }
}

#: Whether two settings lists hold the same numbers.
fn same_numbers(a: list[float], b: list[float]) -> bool {
    if a.len() != b.len() { return false }
    for (index, value) in a.enumerate() {
        if value != b[index] { return false }
    }
    return true
}

fn apply_update(parameter: Tensor, settings: list[float], kind: int) -> void {
    let gradient = parameter.grad
    if gradient is null {
        throw RuntimeError("optimizer.step() needs every parameter to have a gradient")
    }
    let state = state_for(parameter)
    let existing_settings = state.settings
    #: The settings go across on every step, not only the first.
    #:
    #: Keeping the buffer and not writing to it again meant the device read
    #: the first step's numbers for ever: a learning-rate schedule moved the
    #: parameter by its opening rate at every step and decayed nothing, which
    #: is silent because the loss still falls. A handful of floats a step is
    #: not worth the bug.
    #:
    #: The length changes when the kind does -- plain descent carries three
    #: numbers and Adam six -- so a buffer of the wrong size is replaced
    #: rather than written into.
    let settings_buffer = if existing_settings is null or existing_settings.count != settings.len() {
        if existing_settings is not null {
            existing_settings.free()
        }
        let created = Buffer.of(settings)
        state.settings = created
        state.written = settings.copy()
        created
    } else {
        #: Only when they have actually changed. The update kernel reads this
        #: buffer every step, so writing it from the host means waiting for
        #: the step before to finish reading it -- a fence per parameter per
        #: step, which cost eight per cent of the `norm` benchmark when every
        #: step was writing the same three numbers. A schedule changes them a
        #: few times in a run; a fixed rate never does.
        if not same_numbers(state.written, settings) {
            existing_settings.upload(settings)
            state.written = settings.copy()
        }
        existing_settings
    }
    if kind == 0 and settings[1] == 0.0 {
        if parameter.size % 4 == 0 {
            let kernel = kernel_for_sgd4()
            kernel.dispatch_async(
                [parameter.buffer(), gradient.buffer(), settings_buffer],
                parameter.size // 4,
                [parameter.size]
            )
        } else {
            let kernel = kernel_for_sgd()
            kernel.dispatch_async(
                [parameter.buffer(), gradient.buffer(), settings_buffer],
                parameter.size,
                [parameter.size]
            )
        }
        return
    }
    let step = next_step(parameter)
    let kernel = kernel_for_update()
    kernel.dispatch_async(
        [
            parameter.buffer(),
            gradient.buffer(),
            state.first_moment,
            state.second_moment,
            settings_buffer,
        ],
        parameter.size,
        [parameter.size, kind, step]
    )
}

#: Configuration shared by GPU parameter-update kernels.
pub trait Optimizer {
    #: Backend update algorithm identifier.
    fn kind(self) -> int

    #: Numeric settings copied to one small device buffer.
    fn settings(self) -> list[float]

    #: Short name used in summaries.
    fn name(self) -> str

    #: Apply one update to every given parameter, using its accumulated `.grad`.
    fn step(self, parameters: list[Tensor]) -> void
}

#: Minibatch stochastic gradient descent.
pub class SGD: Optimizer {
    #: Step size applied to each minibatch gradient.
    pub let learning_rate: float
    #: Fraction of the previous update retained as velocity.
    pub let momentum: float
    #: Whether to evaluate the update at the momentum look-ahead point.
    pub let nesterov: bool

    #: Configure SGD, optionally with momentum and Nesterov acceleration.
    pub fn init(self, learning_rate: float = 0.01, momentum: float = 0.0, nesterov: bool = false) {
        if learning_rate <= 0.0 {
            throw ValueError(f"learning_rate must be positive, got {learning_rate}")
        }
        if momentum < 0.0 or momentum >= 1.0 {
            throw ValueError(f"momentum must be in [0, 1), got {momentum}")
        }
        if nesterov and momentum == 0.0 {
            throw ValueError("Nesterov SGD needs non-zero momentum")
        }
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
    }

    #: Select the SGD update in the shared Metal kernel.
    pub fn kind(self) -> int { return 0 }

    #: Encode learning rate, momentum, and Nesterov mode for the GPU.
    pub fn settings(self) -> list[float] {
        return [self.learning_rate, self.momentum, if self.nesterov { 1.0 } else { 0.0 }]
    }

    #: Optimizer name used in model output.
    pub fn name(self) -> str { return "SGD" }

    #: Apply one SGD update to every given parameter, using its `.grad`.
    pub fn step(self, parameters: list[Tensor]) -> void {
        for parameter in parameters { apply_update(parameter, self.settings(), self.kind()) }
    }
}

#: Adam with first- and second-moment bias correction.
pub class Adam: Optimizer {
    #: Base step size.
    pub let learning_rate: float
    #: Exponential decay for the first moment.
    pub let beta1: float
    #: Exponential decay for the second moment.
    pub let beta2: float
    #: Denominator stabilizer.
    pub let epsilon: float

    #: Configure Adam with common defaults.
    pub fn init(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-7
    ) {
        if learning_rate <= 0.0 {
            throw ValueError(f"learning_rate must be positive, got {learning_rate}")
        }
        if beta1 < 0.0 or beta1 >= 1.0 { throw ValueError(f"beta1 must be in [0, 1), got {beta1}") }
        if beta2 < 0.0 or beta2 >= 1.0 { throw ValueError(f"beta2 must be in [0, 1), got {beta2}") }
        if epsilon <= 0.0 { throw ValueError(f"epsilon must be positive, got {epsilon}") }
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
    }

    #: Select the Adam update in the shared Metal kernel.
    pub fn kind(self) -> int { return 1 }

    #: Encode learning rate, decay rates, and epsilon for the GPU.
    pub fn settings(self) -> list[float] {
        return [self.learning_rate, self.beta1, self.beta2, self.epsilon]
    }

    #: Optimizer name used in model output.
    pub fn name(self) -> str { return "Adam" }

    #: Apply one Adam update to every given parameter, using its `.grad`.
    pub fn step(self, parameters: list[Tensor]) -> void {
        for parameter in parameters { apply_update(parameter, self.settings(), self.kind()) }
    }
}
