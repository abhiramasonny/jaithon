#: Optimizers used by `Sequential.compile`, and the helpers a hand-written
#: training loop needs around them: gradient clipping and a one-cycle
#: learning-rate schedule.
import std.gpu as gpu
import std.math
from std.gpu import Buffer, Kernel
from .tensor import Tensor

#: The update kernels take their settings as dispatch arguments, not from a
#: buffer.
#:
#: They used to read a small settings buffer kept per parameter, and that
#: buffer had to be written from the host whenever a setting changed. A host
#: write to a buffer that queued work reads has to wait for that work, so a
#: learning-rate schedule that moves every step drained the whole queue once
#: a step: the host could not start encoding the next step until the device
#: had finished the last one. Arguments are copied at encode time and never
#: race anything.
#:
#: Kinds: 0 SGD, 1 Adam, 2 SGD with L2 weight decay, 3 Adam with L2 weight
#: decay, 4 AdamW (decoupled weight decay). Kinds 0 and 1 compute exactly
#: what they always have.
let SOURCE = """#include <metal_stdlib>
using namespace metal;

kernel void parameter_update(device float* value [[buffer(0)]],
                             device const float* gradient [[buffer(1)]],
                             device float* first_moment [[buffer(2)]],
                             device float* second_moment [[buffer(3)]],
                             constant uint& count [[buffer(4)]],
                             constant uint& kind [[buffer(5)]],
                             constant uint& step [[buffer(6)]],
                             constant float& rate [[buffer(7)]],
                             constant float& s1 [[buffer(8)]],
                             constant float& s2 [[buffer(9)]],
                             constant float& s3 [[buffer(10)]],
                             constant float& decay [[buffer(11)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    float g = gradient[id];
    if (kind == 0u || kind == 2u) {
        if (kind == 2u) g += decay * value[id];
        if (s1 == 0.0f) {
            value[id] -= rate * g;
            return;
        }
        float velocity = s1 * first_moment[id] + g;
        first_moment[id] = velocity;
        float update = s2 > 0.5f ? s1 * velocity + g : velocity;
        value[id] -= rate * update;
    } else {
        if (kind == 3u) g += decay * value[id];
        float first = s1 * first_moment[id] + (1.0f - s1) * g;
        float second = s2 * second_moment[id] + (1.0f - s2) * g * g;
        first_moment[id] = first;
        second_moment[id] = second;
        float corrected_first = first / (1.0f - pow(s1, float(step)));
        float corrected_second = second / (1.0f - pow(s2, float(step)));
        float w = value[id];
        if (kind == 4u) w *= 1.0f - rate * decay;
        value[id] = w - rate * corrected_first / (sqrt(corrected_second) + s3);
    }
}

kernel void parameter_update_sgd(device float* value [[buffer(0)]],
                                 device const float* gradient [[buffer(1)]],
                                 constant uint& count [[buffer(2)]],
                                 constant float& rate [[buffer(3)]],
                                 uint id [[thread_position_in_grid]]) {
    if (id < count) value[id] -= rate * gradient[id];
}

kernel void parameter_update_sgd4(device float* value [[buffer(0)]],
                                  device const float* gradient [[buffer(1)]],
                                  constant uint& count [[buffer(2)]],
                                  constant float& rate [[buffer(3)]],
                                  uint id [[thread_position_in_grid]]) {
    uint i = id * 4;
    if (i >= count) return;
    *((device float4*)(value + i)) -= rate * *((device const float4*)(gradient + i));
}

/* One partial sum of squares per threadgroup, written to its own slot, so the
 * total is summed in a fixed order and comes out the same every run. */
kernel void grad_square_partials(device const float* gradient [[buffer(0)]],
                                 device float* partials [[buffer(1)]],
                                 constant uint& count [[buffer(2)]],
                                 constant uint& slot [[buffer(3)]],
                                 uint group [[threadgroup_position_in_grid]],
                                 uint groups [[threadgroups_per_grid]],
                                 uint lid [[thread_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup float lanes[8];
    float total = 0.0f;
    for (uint i = group * 256u + lid; i < count; i += groups * 256u) {
        float v = gradient[i];
        total += v * v;
    }
    total = simd_sum(total);
    if (lane == 0u) lanes[warp] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid != 0u) return;
    float joined = 0.0f;
    for (uint w = 0u; w < 8u; ++w) joined += lanes[w];
    partials[slot + group] = joined;
}

kernel void grad_norm_join(device const float* partials [[buffer(0)]],
                           device float* norm [[buffer(1)]],
                           device float* coefficient [[buffer(2)]],
                           constant uint& slots [[buffer(3)]],
                           constant float& max_norm [[buffer(4)]],
                           uint id [[thread_position_in_grid]]) {
    if (id != 0u) return;
    float total = 0.0f;
    for (uint i = 0u; i < slots; ++i) total += partials[i];
    float length = sqrt(total);
    norm[0] = length;
    // min() would make a NaN norm a coefficient of 1 and leave the finite
    // gradients alone; PyTorch makes them all NaN. Tested on the bits, which
    // fast math cannot assume away.
    bool unknown = (as_type<uint>(length) & 0x7FFFFFFFu) > 0x7F800000u;
    coefficient[0] = unknown ? length : min(1.0f, max_norm / (length + 1e-6f));
}

kernel void grad_scale(device float* gradient [[buffer(0)]],
                       device const float* coefficient [[buffer(1)]],
                       constant uint& count [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    if (id < count) gradient[id] *= coefficient[0];
}
"""

class ParameterState {
    pub let first_moment: Buffer
    pub let second_moment: Buffer
    #: Which optimizer these belong to. The state is kept against the
    #: parameter, and a parameter does not know what stepped it last.
    pub let kind: int

    pub fn init(self, size: int, kind: int, moments: bool) {
        #: Plain descent without momentum reads neither of these -- its
        #: kernel does not take them as arguments -- so it gets a placeholder
        #: rather than two buffers the size of the parameter. That was half a
        #: megabyte per sixty-five thousand weights, allocated and untouched.
        let held = if moments { size } else { 1 }
        self.first_moment = Buffer.of([0.0 for _index in 0..held])
        self.second_moment = Buffer.of([0.0 for _index in 0..held])
        self.kind = kind
    }

    pub fn free(self) -> void {
        self.first_moment.free()
        self.second_moment.free()
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

#: Whether an optimizer of this kind reads the moment buffers.
#:
#: `settings[1]` is the momentum for both kinds of descent; zero means a
#: kernel path that takes no moments at all.
fn wants_moments(kind: int, settings: list[float]) -> bool {
    if kind != 0 and kind != 2 { return true }
    return settings.len() > 1 and settings[1] != 0.0
}

fn state_for(parameter: Tensor, kind: int, moments: bool) -> ParameterState {
    let key = parameter.id
    let existing = _states.get(key, null)
    if existing is not null {
        #: A parameter handed to a different kind of optimizer starts again.
        #: The state is keyed by the parameter alone and a parameter does not
        #: know what stepped it last, so without this Adam read whatever
        #: plain descent had left in `first_moment` as its own first moment:
        #: after five momentum steps at 0.9, Adam's opening step moved the
        #: parameter by 1.93 where a clean one moves 0.01.
        let big_enough = not moments or existing.first_moment.count >= parameter.size
        if existing.kind == kind and big_enough { return existing }
        existing.free()
        _steps.remove(key)
    }
    let state = ParameterState(parameter.size, kind, moments)
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

#: The buffers this optimizer keeps for `parameter` -- momentum's velocity, or
#: Adam's first and second moments -- made if it has none yet.
#:
#: For a path that steps the parameter some other way and must carry the same
#: state: a compiled step reads and writes these in place of buffers of its
#: own, so that a model moving between paths, or on to a step built for
#: another batch size, goes on from where it was rather than from rest.
pub fn moment_buffers(parameter: Tensor, kind: int) -> list[Buffer] {
    let state = state_for(parameter, kind, true)
    if kind == 0 { return [state.first_moment] }
    return [state.first_moment, state.second_moment]
}

#: How many updates Adam has made to `parameter`; its bias correction reads it.
pub fn steps_taken(parameter: Tensor) -> int { return _steps.get(parameter.id, 0) }

#: Record that each of `parameters` has had `count` updates.
pub fn set_steps_taken(parameters: list[Tensor], count: int) -> void {
    for parameter in parameters { _steps[parameter.id] = count }
}

#: Allocate optimizer buffers before the first timed step.
#:
#: `kind` is the optimizer's own, and a state that is already there for it is
#: kept. This used to prepare a plain descent's state whatever the optimizer
#: was, on the reasoning that a state of the wrong kind is replaced on the
#: first step anyway -- which is true, and is the bug: `Sequential` prepares on
#: every `fit`, so an Adam model fitted an epoch at a time had its moments and
#: step count thrown away at the start of every one, and trained a different
#: model from one fitted in a single call.
pub fn prepare_state(parameters: list[Tensor], settings: list[float], kind: int = 0) -> void {
    let moments = wants_moments(kind, settings)
    for parameter in parameters {
        let _state = state_for(parameter, kind, moments)
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

#: The IEEE single-precision bit pattern of `value`, rounded to nearest even.
#:
#: The same number the device would hold had `value` been written into a
#: float buffer, so an argument passed as bits and one read from a buffer are
#: the same float. Rounding carries into the exponent: a value just under a
#: power of two rounds up to that power, not to half of it.
pub fn float32_bits(value: float) -> int {
    if value != value { return 0x7FC00000 }
    var magnitude = value
    var sign = 0
    if magnitude < 0.0 {
        sign = 1
        magnitude = -magnitude
    }
    if magnitude == 0.0 { return sign << 31 }
    if magnitude == math.INF { return sign << 31 | 0x7F800000 }
    let split = __prim__.f64_frexp(magnitude)
    var biased = split[1] + 126
    if biased <= 0 {
        #: Subnormal: a whole number of the smallest step. A count that
        #: rounds up to 2^23 is the smallest normal, and that is also what
        #: its bits spell.
        return sign << 31 | round_half_even(__prim__.f64_ldexp(magnitude, 149))
    }
    var fraction = round_half_even((split[0] * 2.0 - 1.0) * 8388608.0)
    if fraction >= 8388608 {
        fraction = 0
        biased += 1
    }
    if biased >= 255 { return sign << 31 | 0x7F800000 }
    return sign << 31 | biased << 23 | fraction
}

fn round_half_even(value: float) -> int {
    let below = math.floor(value)
    let rest = value - below
    var whole = int(below)
    if rest > 0.5 or rest == 0.5 and whole % 2 == 1 { whole += 1 }
    return whole
}

#: The five float arguments every update kernel is handed, as bits: rate,
#: then the kind's own three settings, then weight decay. Made once a step.
fn setting_bits(settings: list[float]) -> list[int] {
    var bits: list[int] = []
    for index in 0..5 {
        bits.push(if index < settings.len() { float32_bits(settings[index]) } else { 0 })
    }
    return bits
}

fn apply_update(parameter: Tensor, settings: list[float], bits: list[int], kind: int) -> void {
    let gradient = parameter.grad
    if gradient is null {
        throw RuntimeError("optimizer.step() needs every parameter to have a gradient")
    }
    let state = state_for(parameter, kind, wants_moments(kind, settings))
    if kind == 0 and settings[1] == 0.0 {
        if parameter.size % 4 == 0 {
            let kernel = kernel_for_sgd4()
            kernel.dispatch_async(
                [parameter.buffer(), gradient.buffer()],
                parameter.size // 4,
                [parameter.size, bits[0]]
            )
        } else {
            let kernel = kernel_for_sgd()
            kernel.dispatch_async(
                [parameter.buffer(), gradient.buffer()],
                parameter.size,
                [parameter.size, bits[0]]
            )
        }
        return
    }
    let step = next_step(parameter)
    let kernel = kernel_for_update()
    kernel.dispatch_async(
        [parameter.buffer(), gradient.buffer(), state.first_moment, state.second_moment],
        parameter.size,
        [parameter.size, kind, step, bits[0], bits[1], bits[2], bits[3], bits[4]]
    )
}

#: Step every parameter with one optimizer's numbers.
fn step_all(parameters: list[Tensor], settings: list[float], kind: int) -> void {
    #: Converted once a step, not once a parameter: they are the same numbers
    #: for every one of them.
    let bits = if kind == 0 or kind == 2 {
        setting_bits([settings[0], settings[1], settings[2], 0.0, if settings.len() > 3 {
            settings[3]
        } else {
            0.0
        }])
    } else {
        setting_bits(settings)
    }
    for parameter in parameters { apply_update(parameter, settings, bits, kind) }
}

#: Configuration shared by GPU parameter-update kernels.
pub trait Optimizer {
    #: Backend update algorithm identifier.
    fn kind(self) -> int

    #: Numeric settings, learning rate first.
    fn settings(self) -> list[float]

    #: Short name used in summaries.
    fn name(self) -> str

    #: Apply one update to every given parameter, using its accumulated `.grad`.
    fn step(self, parameters: list[Tensor]) -> void
}

#: NaN first: an ordered comparison with NaN throws a TypeError.
fn check_decay(weight_decay: float) -> void {
    if weight_decay != weight_decay or weight_decay < 0.0 {
        throw ValueError(f"weight_decay must be non-negative, got {weight_decay}")
    }
}

#: Minibatch stochastic gradient descent.
#:
#: `weight_decay` adds `weight_decay * w` to each gradient before the
#: momentum, as PyTorch's `SGD(weight_decay=...)` does. A decaying optimizer
#: reports its own kind, so the fused and compiled training paths -- which
#: have no decay term -- leave it to the paths that call `step`, rather than
#: training without the decay that was asked for.
#:
#: `learning_rate` may be assigned between steps; a schedule does not need a
#: new optimizer, and the state kept against each parameter carries on.
pub class SGD: Optimizer {
    #: Step size applied to each minibatch gradient. Assignable.
    pub var learning_rate: float
    #: Fraction of the previous update retained as velocity.
    pub let momentum: float
    #: Whether to evaluate the update at the momentum look-ahead point.
    pub let nesterov: bool
    #: L2 penalty coefficient added to the gradient.
    pub let weight_decay: float

    #: Configure SGD, optionally with momentum, Nesterov acceleration and
    #: weight decay.
    pub fn init(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        nesterov: bool = false,
        weight_decay: float = 0.0
    ) {
        if learning_rate <= 0.0 {
            throw ValueError(f"learning_rate must be positive, got {learning_rate}")
        }
        if momentum < 0.0 or momentum >= 1.0 {
            throw ValueError(f"momentum must be in [0, 1), got {momentum}")
        }
        if nesterov and momentum == 0.0 {
            throw ValueError("Nesterov SGD needs non-zero momentum")
        }
        check_decay(weight_decay)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
    }

    #: 0, or 2 when the update carries weight decay.
    pub fn kind(self) -> int { return if self.weight_decay == 0.0 { 0 } else { 2 } }

    #: Learning rate, momentum, Nesterov mode, and the weight decay when there
    #: is one.
    pub fn settings(self) -> list[float] {
        let nesterov = if self.nesterov { 1.0 } else { 0.0 }
        if self.weight_decay == 0.0 { return [self.learning_rate, self.momentum, nesterov] }
        return [self.learning_rate, self.momentum, nesterov, self.weight_decay]
    }

    #: Optimizer name used in model output.
    pub fn name(self) -> str { return "SGD" }

    #: Apply one SGD update to every given parameter, using its `.grad`.
    pub fn step(self, parameters: list[Tensor]) -> void {
        step_all(parameters, self.settings(), self.kind())
    }
}

#: Adam with first- and second-moment bias correction.
#:
#: `weight_decay` here is the L2 form, added to the gradient before the
#: moments, as in PyTorch's `Adam(weight_decay=...)`. For decay applied to the
#: weights directly, use `AdamW`.
pub class Adam: Optimizer {
    #: Base step size. Assignable.
    pub var learning_rate: float
    #: Exponential decay for the first moment.
    pub let beta1: float
    #: Exponential decay for the second moment.
    pub let beta2: float
    #: Denominator stabilizer.
    pub let epsilon: float
    #: L2 penalty coefficient added to the gradient.
    pub let weight_decay: float

    #: Configure Adam with common defaults.
    pub fn init(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-7,
        weight_decay: float = 0.0
    ) {
        check_adam(learning_rate, beta1, beta2, epsilon)
        check_decay(weight_decay)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
    }

    #: 1, or 3 when the gradient carries an L2 term.
    pub fn kind(self) -> int { return if self.weight_decay == 0.0 { 1 } else { 3 } }

    #: Learning rate, decay rates and epsilon, then the weight decay when
    #: there is one.
    pub fn settings(self) -> list[float] {
        if self.weight_decay == 0.0 {
            return [self.learning_rate, self.beta1, self.beta2, self.epsilon]
        }
        return [self.learning_rate, self.beta1, self.beta2, self.epsilon, self.weight_decay]
    }

    #: Optimizer name used in model output.
    pub fn name(self) -> str { return "Adam" }

    #: Apply one Adam update to every given parameter, using its `.grad`.
    pub fn step(self, parameters: list[Tensor]) -> void {
        step_all(parameters, self.settings(), self.kind())
    }
}

fn check_adam(learning_rate: float, beta1: float, beta2: float, epsilon: float) -> void {
    if learning_rate <= 0.0 {
        throw ValueError(f"learning_rate must be positive, got {learning_rate}")
    }
    if beta1 < 0.0 or beta1 >= 1.0 { throw ValueError(f"beta1 must be in [0, 1), got {beta1}") }
    if beta2 < 0.0 or beta2 >= 1.0 { throw ValueError(f"beta2 must be in [0, 1), got {beta2}") }
    if epsilon <= 0.0 { throw ValueError(f"epsilon must be positive, got {epsilon}") }
}

#: Adam with decoupled weight decay (Loshchilov and Hutter).
#:
#: Each step first shrinks every weight by `learning_rate * weight_decay` of
#: itself, then takes the Adam step on the undecayed gradient -- PyTorch's
#: `AdamW` exactly. With `weight_decay` zero it is plain Adam and trains on
#: every path Adam does.
pub class AdamW: Optimizer {
    #: Base step size. Assignable.
    pub var learning_rate: float
    #: Exponential decay for the first moment.
    pub let beta1: float
    #: Exponential decay for the second moment.
    pub let beta2: float
    #: Denominator stabilizer.
    pub let epsilon: float
    #: Fraction of each weight removed per unit of learning rate.
    pub let weight_decay: float

    #: Configure AdamW. The decay default is PyTorch's.
    pub fn init(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-7,
        weight_decay: float = 0.01
    ) {
        check_adam(learning_rate, beta1, beta2, epsilon)
        check_decay(weight_decay)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
    }

    #: 4, or Adam's 1 when there is no decay to apply.
    pub fn kind(self) -> int { return if self.weight_decay == 0.0 { 1 } else { 4 } }

    #: Learning rate, decay rates and epsilon, then the weight decay when
    #: there is one.
    pub fn settings(self) -> list[float] {
        if self.weight_decay == 0.0 {
            return [self.learning_rate, self.beta1, self.beta2, self.epsilon]
        }
        return [self.learning_rate, self.beta1, self.beta2, self.epsilon, self.weight_decay]
    }

    #: Optimizer name used in model output.
    pub fn name(self) -> str { return "AdamW" }

    #: Apply one AdamW update to every given parameter, using its `.grad`.
    pub fn step(self, parameters: list[Tensor]) -> void {
        step_all(parameters, self.settings(), self.kind())
    }
}

var _norm_kernels: list[Kernel] = []

fn norm_kernels() -> list[Kernel] {
    if _norm_kernels.len() == 0 {
        _norm_kernels = [
            Kernel.compile(SOURCE, "grad_square_partials"),
            Kernel.compile(SOURCE, "grad_norm_join"),
            Kernel.compile(SOURCE, "grad_scale"),
        ]
    }
    return _norm_kernels
}

#: Scale every gradient down so that together they have an L2 norm of at most
#: `max_norm`, as PyTorch's `clip_grad_norm_` does.
#:
#: The norm is over all the gradients as one vector. When it exceeds
#: `max_norm`, every gradient is multiplied by `max_norm / (norm + 1e-6)`;
#: otherwise they are left as they are. Parameters without a gradient are
#: skipped. `max_norm` may be `math.INF`, PyTorch's way of measuring the norm
#: without clipping. A NaN norm makes every gradient NaN, as PyTorch's does.
#:
#: Returns a one-element tensor holding the norm before clipping, which the
#: caller frees. It is computed and applied on the device, so nothing waits for
#: it: download it only on the steps where it is wanted.
pub fn clip_grad_norm(parameters: list[Tensor], max_norm: float) -> Tensor {
    if max_norm != max_norm or max_norm <= 0.0 {
        throw ValueError(f"max_norm must be positive, got {max_norm}")
    }
    #: Converted before anything is allocated, so nothing is left behind if
    #: it throws.
    let bound = float32_bits(max_norm)
    var gradients: list[Tensor] = []
    var groups: list[int] = []
    var slots = 0
    for parameter in parameters {
        let gradient = parameter.grad
        if gradient is null or gradient.size == 0 { continue }
        let count = min(64, max(1, (gradient.size + 4095) // 4096))
        gradients.push(gradient)
        groups.push(count)
        slots += count
    }
    if gradients.len() == 0 { return Tensor.zeros([1]) }
    let kernels = norm_kernels()
    let partials = Tensor([slots])
    let norm = Tensor([1])
    let coefficient = Tensor([1])
    var slot = 0
    for (index, gradient) in gradients.enumerate() {
        kernels[0].dispatch_async(
            [gradient.buffer(), partials.buffer()],
            groups[index] * 256,
            [gradient.size, slot],
            group_size: 256
        )
        slot += groups[index]
    }
    kernels[1].dispatch_async(
        [partials.buffer(), norm.buffer(), coefficient.buffer()],
        1,
        [slots, bound]
    )
    for gradient in gradients {
        kernels[2].dispatch_async(
            [gradient.buffer(), coefficient.buffer()],
            gradient.size,
            [gradient.size]
        )
    }
    partials.free()
    coefficient.free()
    return norm
}

#: The learning rate of a one-cycle schedule at `step` (counted from zero),
#: identical to PyTorch's `OneCycleLR` with its defaults (cosine annealing,
#: two phases).
#:
#: The rate rises from `max_lr / div_factor` to `max_lr` over the first
#: `pct_start` of `total_steps`, then falls to
#: `max_lr / div_factor / final_div_factor` at the last step. Assign it to an
#: optimizer's `learning_rate` before each step.
pub fn one_cycle_lr(
    step: int,
    total_steps: int,
    max_lr: float,
    pct_start: float = 0.3,
    div_factor: float = 25.0,
    final_div_factor: float = 10000.0
) -> float {
    if total_steps <= 0 { throw ValueError(f"total_steps must be positive, got {total_steps}") }
    if step < 0 or step > total_steps {
        throw ValueError(f"step {step} is outside a schedule of {total_steps} steps")
    }
    if pct_start <= 0.0 or pct_start >= 1.0 {
        throw ValueError(f"pct_start must be in (0, 1), got {pct_start}")
    }
    let initial = max_lr / div_factor
    let lowest = initial / final_div_factor
    let peak_step = pct_start * float(total_steps) - 1.0
    let last_step = float(total_steps - 1)
    let at = float(step)
    if at <= peak_step {
        return cosine_between(initial, max_lr, at / peak_step)
    }
    return cosine_between(max_lr, lowest, (at - peak_step) / (last_step - peak_step))
}

fn cosine_between(start: float, end: float, fraction: float) -> float {
    return end + (start - end) / 2.0 * (math.cos(math.PI * fraction) + 1.0)
}
