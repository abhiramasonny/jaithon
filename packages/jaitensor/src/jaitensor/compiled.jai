#: A whole training step as one compiled graph.
#:
#: Training a network here normally runs one operation at a time: a few hundred
#: dispatches for a step, each costing about five microseconds of device time
#: whatever it does, with the processor walking the layer list between them.
#: For a small convolutional model that overhead is most of the step.
#:
#: The platform can differentiate a graph it already holds, so the forward
#: pass, the loss, every gradient and the update that follows them can all be
#: built as one graph and compiled once. What was hundreds of dispatches
#: becomes one, and the compiler gets to see the whole step at once -- which is
#: how the hand-fused dense path came to be eight times PyTorch, done here for
#: any model whose layers can be expressed.
#:
#: A model that cannot be expressed compiles to nothing and the caller keeps
#: the ordinary path; `refusal` says which layer stopped it.
#:
#: One step of this and one step of the ordinary path agree to about a part in
#: a billion -- the same arithmetic in a different order. Over many steps they
#: drift apart, as any two orderings of float32 do once a ReLU is in the way:
#: a value either side of zero takes a different branch and the two runs are
#: on slightly different trajectories from then on. Both are training the same
#: model correctly; neither is the reference for the other.
#:
#: ```
#: let step = CompiledStep.of(model, batch_size, learning_rate)
#: if step is not null { step.run(features, labels) }
#: ```
import std.gpu as gpu
from std.os import env
from std.gpu import Buffer
from .layers import Activation, AvgPool2d, Conv2d, Dense, Flatten, Layer, LayerNorm, MaxPool2d
from .tensor import Tensor

#: Reduction kinds the graph builder takes.
const REDUCE_SUM = 0
const REDUCE_MEAN = 1

#: Binary op codes the graph builder takes.
const OP_ADD = 0
const OP_SUB = 1
const OP_MUL = 2
const OP_DIV = 3
const OP_POW = 6

#: Unary op codes the graph builder takes.
const UNARY_RELU = 0
const UNARY_SIGMOID = 1
const UNARY_TANH = 2
const UNARY_EXP = 3
const UNARY_LOG = 4
const UNARY_NEGATE = 6
const UNARY_SQRT = 5
const UNARY_SILU = 9

#: What the forward pass of one layer became, and the parameters it used.
#:
#: The parameters are collected as they are consumed rather than asked of the
#: layer, because the platform's differentiation takes the process down when
#: handed a tensor the loss does not actually depend on.
class Built {
    pub var value: int
    pub var shape: list[int]

    pub fn init(self, value: int, shape: list[int]) {
        self.value = value
        self.shape = shape
    }
}

#: The state a step is built up in.
class Build {
    pub let builder: int
    pub var params: list[Tensor]
    pub var param_ids: list[int]
    pub var refused: str?

    pub fn init(self, builder: int) {
        self.builder = builder
        self.params = []
        self.param_ids = []
        self.refused = null
    }

    #: Feed a parameter in as a graph input and remember it, so the gradient
    #: pass can be asked for exactly the ones the forward touched.
    pub fn parameter(self, tensor: Tensor) -> int {
        for (index, held) in self.params.enumerate() {
            if held is tensor { return self.param_ids[index] }
        }
        let id = __prim__.graph_input(self.builder, tensor.shape)
        self.params.push(tensor)
        self.param_ids.push(id)
        return id
    }

    pub fn refuse(self, why: str) -> void {
        if self.refused is null { self.refused = why }
    }
}

fn product(shape: list[int]) -> int {
    var total = 1
    for dim in shape { total = total * dim }
    return total
}

#: A layer's activation, applied to what its affine part produced.
#:
#: Switched on the activation itself and not on `activation_code`, which
#: reports Gelu, Silu and LeakyRelu as linear -- the layers apply those as a
#: second operation after the affine one, so a step built from the code alone
#: would drop them and train a different model without saying so.
#:
#: Softmax is linear here on purpose, and only ever on the last layer, which
#: `Sequential` enforces: the loss takes logits and does the softmax itself.
fn emit_activation(state: Build, value: int, activation: Activation) -> int {
    let b = state.builder
    if activation == Activation.Linear or activation == Activation.Softmax { return value }
    if activation == Activation.Relu { return __prim__.graph_unary(b, value, UNARY_RELU) }
    if activation == Activation.Sigmoid { return __prim__.graph_unary(b, value, UNARY_SIGMOID) }
    if activation == Activation.Tanh { return __prim__.graph_unary(b, value, UNARY_TANH) }
    if activation == Activation.Silu { return __prim__.graph_unary(b, value, UNARY_SILU) }
    if activation == Activation.LeakyRelu { return __prim__.graph_leaky_relu(b, value, 0.01) }
    if activation == Activation.Gelu {
        #: The same tanh approximation `ops.gelu` computes, so the two agree:
        #: 0.5 x (1 + tanh(0.7978845608 (x + 0.044715 x^3))).
        let cubed = __prim__.graph_binary(
            b,
            __prim__.graph_binary(b, value, value, OP_MUL),
            value,
            OP_MUL
        )
        let weighted = __prim__.graph_binary(b, cubed, scalar(state, 0.044715), OP_MUL)
        let inner = __prim__.graph_binary(b, value, weighted, OP_ADD)
        let scaled = __prim__.graph_binary(b, inner, scalar(state, 0.7978845608), OP_MUL)
        let curved = __prim__.graph_unary(b, scaled, UNARY_TANH)
        let shifted = __prim__.graph_binary(b, curved, scalar(state, 1.0), OP_ADD)
        let halved = __prim__.graph_binary(b, value, scalar(state, 0.5), OP_MUL)
        return __prim__.graph_binary(b, halved, shifted, OP_MUL)
    }
    state.refuse(f"an activation this step cannot express ({activation})")
    return -1
}

fn scalar(state: Build, value: float) -> int {
    return __prim__.graph_constant(state.builder, [value], [1])
}

#: One layer's forward, as graph operations. Null when it cannot be expressed.
#:
#: Each kind gets its own function taking `any` and guarding at the top,
#: because that is the shape the checker narrows a type in.
fn emit_dense(state: Build, layer: any, input: Built) -> Built? {
    if not layer is Dense { return null }
    let b = state.builder
    let params = layer.parameters()
    if params.len() != 2 {
        state.refuse("a dense layer without both a weight and a bias")
        return null
    }
    let weights = state.parameter(params[0])
    let bias = state.parameter(params[1])
    let width = params[0].shape[1]
    let producted = __prim__.graph_matmul(b, input.value, weights)
    let biased = __prim__.graph_binary(b, producted, bias, OP_ADD)
    let shape = [input.shape[0], width]
    let activated = emit_activation(state, biased, layer.activation)
    if activated < 0 { return null }
    return Built(activated, shape)
}

fn emit_flatten(state: Build, layer: any, input: Built) -> Built? {
    if not layer is Flatten { return null }
    let rows = input.shape[0]
    let width = product(input.shape) // rows
    return Built(__prim__.graph_reshape(state.builder, input.value, [rows, width]), [rows, width])
}

fn emit_conv(state: Build, layer: any, input: Built) -> Built? {
    if not layer is Conv2d { return null }
    let b = state.builder
    let params = layer.parameters()
    if params.len() < 1 {
        state.refuse("a convolution with no kernel")
        return null
    }
    if input.shape.len() != 4 {
        state.refuse("a convolution over something that is not a batch of images")
        return null
    }
    let kernel = state.parameter(params[0])
    let bias = if params.len() > 1 { state.parameter(params[1]) } else { -1 }
    let pad = layer.padding
    let stride = layer.stride
    let size = layer.kernel_size
    #: stride y, stride x, pad top, pad bottom, pad left, pad right, dilation
    #: y, dilation x, groups, then the two layout flags. jaitensor keeps images
    #: channels-last and kernels as height, width, in, out, which is not what
    #: the builder assumes by default.
    let settings = [stride, stride, pad, pad, pad, pad, 1, 1, 1, 1, 1]
    let made = __prim__.graph_conv(b, input.value, kernel, bias, settings)
    if made < 0 {
        state.refuse("a convolution the platform would not build")
        return null
    }
    let height = (input.shape[1] + 2 * pad - size) // stride + 1
    let width = (input.shape[2] + 2 * pad - size) // stride + 1
    let shape = [input.shape[0], height, width, params[0].shape[3]]
    let activated = emit_activation(state, made, layer.activation)
    if activated < 0 { return null }
    return Built(activated, shape)
}

#: Both poolings, which differ only in what they do with a window.
#:
#: `average` divides by however many real values the window held, which is what
#: jaitensor's own kernel counts, so padding never counts toward the mean.
fn emit_pooling(
    state: Build,
    input: Built,
    size: int,
    stride: int,
    pad: int,
    average: bool
) -> Built? {
    if input.shape.len() != 4 {
        state.refuse("a pool over something that is not an image batch")
        return null
    }
    #: kernel y, kernel x, stride y, stride x, pad top, bottom, left, right,
    #: ceil mode, pad counts toward an average, channels last.
    let settings = [size, size, stride, stride, pad, pad, pad, pad, 0, 0, 1]
    let kind = if average { 1 } else { 0 }
    let made = __prim__.graph_pool(state.builder, input.value, settings, kind)
    if made < 0 {
        state.refuse("a pool the platform would not build")
        return null
    }
    let height = (input.shape[1] + 2 * pad - size) // stride + 1
    let width = (input.shape[2] + 2 * pad - size) // stride + 1
    return Built(made, [input.shape[0], height, width, input.shape[3]])
}

fn emit_pool(state: Build, layer: any, input: Built) -> Built? {
    if not layer is MaxPool2d { return null }
    return emit_pooling(state, input, layer.kernel_size, layer.stride, layer.padding, false)
}

fn emit_avg_pool(state: Build, layer: any, input: Built) -> Built? {
    if not layer is AvgPool2d { return null }
    return emit_pooling(state, input, layer.kernel_size, layer.stride, layer.padding, true)
}

#: Normalisation over the last axis, with the same epsilon the kernel uses.
fn emit_layer_norm(state: Build, layer: any, input: Built) -> Built? {
    if not layer is LayerNorm { return null }
    if input.shape.len() != 2 {
        state.refuse("a normalisation over something that is not a batch of rows")
        return null
    }
    let params = layer.parameters()
    if params.len() != 2 {
        state.refuse("a normalisation without both a scale and a shift")
        return null
    }
    let gamma = state.parameter(params[0])
    let beta = state.parameter(params[1])
    let made = __prim__.graph_layer_norm(state.builder, input.value, gamma, beta, [1], 1.0e-5)
    if made < 0 {
        state.refuse("a normalisation the platform would not build")
        return null
    }
    return Built(made, input.shape)
}

#: Batch normalisation is deliberately absent, and it is expressible: it is the
#: row-wise normalisation over three axes instead of one, and a step built with
#: it matched the eager one to 6e-8 over eight steps. It is left out because it
#: is slower. On the `norm` benchmark the whole-graph step took 2.05 s against
#: the eager path's 1.49 s, and 3.28 s against 2.58 s -- expressing it means the
#: model compiles, and compiling it replaces kernels that beat the graph's own
#: normalisation. Worth revisiting only if that stops being true.
fn emit_layer(state: Build, layer: Layer, input: Built) -> Built? {
    if layer is Dense { return emit_dense(state, layer, input) }
    if layer is Flatten { return emit_flatten(state, layer, input) }
    if layer is Conv2d { return emit_conv(state, layer, input) }
    if layer is MaxPool2d { return emit_pool(state, layer, input) }
    if layer is AvgPool2d { return emit_avg_pool(state, layer, input) }
    if layer is LayerNorm { return emit_layer_norm(state, layer, input) }
    state.refuse("a layer this step does not express")
    return null
}

#: Mean sparse softmax cross-entropy, as graph operations.
#:
#: The platform's own loss rather than a log-softmax and a pick assembled here.
#: A hand-built one works forwards and then fails to differentiate: a
#: comparison, which is how one picks a label's own score without a one-hot
#: operation, has no derivative, and MPSGraph aborts the process rather than
#: saying so.
fn emit_loss(state: Build, logits: Built, labels: int, classes: int) -> int {
    let b = state.builder
    let onehot = __prim__.graph_onehot(b, labels, classes)
    if onehot < 0 { return -1 }
    return __prim__.graph_softmax_cross_entropy(b, logits.value, onehot, 1, 2)
}

#: A training step compiled to one graph, with the parameters it updates.
#:
#: The parameters are held twice over. The graph reads one copy and writes the
#: other, and the two swap after every step: a compiled graph that read and
#: wrote the same memory would be racing itself, and this is the same
#: ping-pong the hand-fused dense path uses.
pub class CompiledStep {
    let handle: int
    pub let params: list[Tensor]
    var shadow: list[Buffer]
    #: Whatever the optimizer carries from one step to the next -- momentum's
    #: velocity, or Adam's two moments and its step count -- fed in and
    #: produced in the same order, ping-ponged the way the parameters are.
    #: Empty for an optimizer that carries nothing.
    var carried: list[Buffer]
    var carried_shadow: list[Buffer]
    var live_is_shadow: bool
    pub let batch_size: int
    pub let classes: int
    var freed: bool

    pub fn init(
        self,
        handle: int,
        params: list[Tensor],
        shadow: list[Buffer],
        carried: list[Buffer],
        carried_shadow: list[Buffer],
        batch_size: int,
        classes: int
    ) {
        self.handle = handle
        self.params = params
        self.shadow = shadow
        self.carried = carried
        self.carried_shadow = carried_shadow
        self.live_is_shadow = false
        self.batch_size = batch_size
        self.classes = classes
        self.freed = false
    }

    #: One step, leaving the loss on the device.
    #:
    #: `scores` receives the batch's logits, which is what an accuracy count
    #: needs and what the graph is producing anyway. `into` receives the loss.
    #: Neither is read here: a training loop that read the loss every step
    #: would wait for the device every step, which costs more than the step.
    pub fn step_into(self, features: Tensor, labels: Tensor, scores: Tensor, into: Tensor) -> void {
        if self.freed { throw RuntimeError("this CompiledStep has been released") }
        var feeds: list[int] = [features.buffer().handle_of(), labels.buffer().handle_of()]
        var results: list[int] = [into.buffer().handle_of(), scores.buffer().handle_of()]
        for (index, param) in self.params.enumerate() {
            let own = param.buffer()
            let other = self.shadow[index]
            let reads = if self.live_is_shadow { other } else { own }
            let writes = if self.live_is_shadow { own } else { other }
            feeds.push(reads.handle_of())
            results.push(writes.handle_of())
        }
        for index in 0..self.carried.len() {
            let own = self.carried[index]
            let other = self.carried_shadow[index]
            let reads = if self.live_is_shadow { other } else { own }
            let writes = if self.live_is_shadow { own } else { other }
            feeds.push(reads.handle_of())
            results.push(writes.handle_of())
        }
        __prim__.graph_run(self.handle, feeds, results)
        self.live_is_shadow = not self.live_is_shadow
    }

    #: One step, waiting for the loss. For a caller that wants it now, which a
    #: training loop should not.
    pub fn run(self, features: Tensor, labels: Tensor, scores: Tensor? = null) -> float {
        let logits = scores ?? Tensor([self.batch_size, self.classes])
        let loss = Tensor([1])
        self.step_into(features, labels, logits, loss)
        let value = loss.download()[0]
        loss.free()
        if scores is null { logits.free() }
        return value
    }

    #: Put the current values back where the model's own tensors expect them,
    #: which matters only when the last step wrote to the shadow side.
    pub fn settle(self) -> void {
        if not self.live_is_shadow { return }
        for (index, param) in self.params.enumerate() {
            param.buffer().upload(self.shadow[index].download())
        }
        for index in 0..self.carried.len() {
            self.carried[index].upload(self.carried_shadow[index].download())
        }
        self.live_is_shadow = false
    }

    pub fn free(self) -> void {
        if self.freed { return }
        self.freed = true
        self.settle()
        for buffer in self.shadow { buffer.free() }
        for buffer in self.carried { buffer.free() }
        for buffer in self.carried_shadow { buffer.free() }
        __prim__.graph_plan_free(self.handle)
    }
}

#: Why the last attempt to compile a step gave up, or null when it did not.
#:
#: Read through a call rather than exported as a variable: an imported binding
#: takes the value it had when the import ran, so a caller watching a variable
#: would always see the null it started as.
var _refusal: str? = null

pub fn refusal() -> str? { return _refusal }

var _allowed = -1

#: `JAITENSOR_COMPILED_STEP=0` keeps every model on the ordinary path, for
#: telling a difference in speed from a difference in arithmetic.
fn allowed() -> bool {
    if _allowed < 0 {
        _allowed = if (env("JAITENSOR_COMPILED_STEP") ?? "1") == "0" { 0 } else { 1 }
    }
    return _allowed == 1
}

#: Build a compiled step for `layers`, or null when they cannot be expressed.
#:
#: `input_shape` is one batch's worth including the batch dimension. SGD with
#: or without momentum is expressed; momentum's velocity is carried through the
#: graph as another pair of tensors, fed in and produced, the same way the
#: parameters are.
pub fn compile_step(
    layers: list[Layer],
    input_shape: list[int],
    classes: int,
    learning_rate: float,
    momentum: float = 0.0,
    nesterov: bool = false,
    adam: bool = false,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1.0e-7
) -> CompiledStep? {
    if not allowed() {
        _refusal = "compiled steps are switched off"
        return null
    }
    if not gpu.is_available() {
        _refusal = "there is no device to compile for"
        return null
    }
    let builder = __prim__.graph_new()
    let state = Build(builder)
    let features = __prim__.graph_input(builder, input_shape)
    let labels = __prim__.graph_input(builder, [input_shape[0]])

    var carried = Built(features, input_shape)
    for layer in layers {
        let next = emit_layer(state, layer, carried)
        if next is null {
            _refusal = state.refused ?? "a layer this step does not express"
            __prim__.graph_free(builder)
            return null
        }
        carried = next
    }
    if carried.shape.len() != 2 or carried.shape[1] != classes {
        _refusal = f"the model ends at {carried.shape}, not a row of {classes} scores"
        __prim__.graph_free(builder)
        return null
    }
    if state.params.len() == 0 {
        _refusal = "the model has nothing to train"
        __prim__.graph_free(builder)
        return null
    }

    let loss = emit_loss(state, carried, labels, classes)
    if loss < 0 {
        _refusal = "the loss would not build"
        __prim__.graph_free(builder)
        return null
    }
    let grads = __prim__.graph_gradients(builder, loss, state.param_ids)

    #: The update, in the graph, so it is part of the same compiled thing as
    #: the pass that produced the gradient.
    #:
    #: Without momentum that is `param - rate * gradient`. With it the velocity
    #: comes in as a tensor of its own and goes out updated:
    #: `v = momentum * v + g`, and the step is `v` again, or
    #: `momentum * v + g` when Nesterov is asked for.
    let b = builder
    var updated: list[int] = []
    var carried_ids: list[int] = []
    var carried_out: list[int] = []
    var carried_sizes: list[list[int]] = []
    let rate = __prim__.graph_constant(b, [learning_rate], [1])

    #: Adam counts its steps, and the count is the same for every parameter, so
    #: one counter serves them all. Kept in the graph rather than fed each step:
    #: writing a scalar from the host every iteration would make the host wait
    #: for the step that read it last.
    var beat_next = -1
    if adam {
        let beat = __prim__.graph_input(b, [1])
        carried_ids.push(beat)
        carried_sizes.push([1])
        beat_next = __prim__.graph_binary(b, beat, scalar(state, 1.0), OP_ADD)
        carried_out.push(beat_next)
    }

    for (index, id) in state.param_ids.enumerate() {
        if grads[index] < 0 {
            _refusal = f"parameter {index} got no gradient"
            __prim__.graph_free(b)
            return null
        }
        let gradient = grads[index]
        let shape = state.params[index].shape
        var move = gradient

        if adam {
            #: first = b1 f + (1 - b1) g and second = b2 s + (1 - b2) g^2, each
            #: divided by what it would have been had it not started at rest,
            #: and the step is first / (sqrt(second) + epsilon).
            let held_first = __prim__.graph_input(b, shape)
            let held_second = __prim__.graph_input(b, shape)
            carried_ids.push(held_first)
            carried_ids.push(held_second)
            carried_sizes.push(shape)
            carried_sizes.push(shape)

            let fresh_first = __prim__.graph_binary(
                b,
                __prim__.graph_binary(b, held_first, scalar(state, beta1), OP_MUL),
                __prim__.graph_binary(
                    b,
                    gradient,
                    scalar(state, 1.0 - beta1),
                    OP_MUL
                ),
                OP_ADD
            )
            let squared = __prim__.graph_binary(b, gradient, gradient, OP_MUL)
            let fresh_second = __prim__.graph_binary(
                b,
                __prim__.graph_binary(
                    b,
                    held_second,
                    scalar(state, beta2),
                    OP_MUL
                ),
                __prim__.graph_binary(
                    b,
                    squared,
                    scalar(state, 1.0 - beta2),
                    OP_MUL
                ),
                OP_ADD
            )
            carried_out.push(fresh_first)
            carried_out.push(fresh_second)

            let left1 = __prim__.graph_binary(
                b,
                scalar(state, 1.0),
                __prim__.graph_binary(b, scalar(state, beta1), beat_next, OP_POW),
                OP_SUB
            )
            let left2 = __prim__.graph_binary(
                b,
                scalar(state, 1.0),
                __prim__.graph_binary(b, scalar(state, beta2), beat_next, OP_POW),
                OP_SUB
            )
            let corrected_first = __prim__.graph_binary(b, fresh_first, left1, OP_DIV)
            let corrected_second = __prim__.graph_binary(b, fresh_second, left2, OP_DIV)
            let root = __prim__.graph_unary(b, corrected_second, UNARY_SQRT)
            let denominator = __prim__.graph_binary(b, root, scalar(state, epsilon), OP_ADD)
            move = __prim__.graph_binary(b, corrected_first, denominator, OP_DIV)
        } elif momentum != 0.0 {
            let previous = __prim__.graph_input(b, shape)
            carried_ids.push(previous)
            carried_sizes.push(shape)
            let decay = scalar(state, momentum)
            let kept = __prim__.graph_binary(b, previous, decay, OP_MUL)
            let fresh = __prim__.graph_binary(b, kept, gradient, OP_ADD)
            carried_out.push(fresh)
            move = if nesterov {
                __prim__.graph_binary(b, __prim__.graph_binary(b, fresh, decay, OP_MUL), gradient, OP_ADD)
            } else {
                fresh
            }
        }

        let scaled = __prim__.graph_binary(b, move, rate, OP_MUL)
        updated.push(__prim__.graph_binary(b, id, scaled, OP_SUB))
    }

    var inputs: list[int] = [features, labels]
    for id in state.param_ids { inputs.push(id) }
    for id in carried_ids { inputs.push(id) }
    var outputs: list[int] = [loss, carried.value]
    for id in updated { outputs.push(id) }
    for id in carried_out { outputs.push(id) }

    let handle = __prim__.graph_compile(builder, inputs, outputs)
    if handle < 0 {
        _refusal = "the step would not compile"
        return null
    }
    var shadow: list[Buffer] = []
    for param in state.params { shadow.push(Buffer(param.size)) }
    #: Everything carried starts at rest, including the step count.
    var held: list[Buffer] = []
    var held_shadow: list[Buffer] = []
    for shape in carried_sizes {
        var size = 1
        for dimension in shape { size = size * dimension }
        let first = Buffer(size)
        let second = Buffer(size)
        first.fill_zero()
        second.fill_zero()
        held.push(first)
        held_shadow.push(second)
    }
    _refusal = null
    return CompiledStep(handle, state.params, shadow, held, held_shadow, input_shape[0], classes)
}
