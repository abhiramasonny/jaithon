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
from .layers import Activation, AvgPool2d, BatchNorm2d, Conv2d, Dense, Flatten, Layer, LayerNorm
from .layers import MaxPool2d
from .optimizers import moment_buffers, set_steps_taken, steps_taken
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

const BN_GRAPH_DEFAULT = 0

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
#: The products a step is allowed to work out in half precision.
#:
#: Only the convolution and the matrix product: those are where the arithmetic
#: is, and they are what a framework's autocast narrows too. Everything else --
#: the normalisations, the loss, the optimiser and the weights themselves --
#: stays wide, so the values carried between steps never lose a bit.
#:
#: Off unless the caller asked for mixed precision, and the equivalence tests
#: run with it off, which is why they can hold the step to 1e-5 of the eager
#: path.
fn narrow(state: Build, value: int) -> int {
    if not gpu.mixed_precision() { return value }
    return __prim__.graph_cast(state.builder, value, 1)
}

fn widen(state: Build, value: int) -> int {
    if not gpu.mixed_precision() { return value }
    return __prim__.graph_cast(state.builder, value, 0)
}

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
    let producted = widen(
        state,
        __prim__.graph_matmul(b, narrow(state, input.value), narrow(state, weights))
    )
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
    #: The convolution takes its bias in the same call, so narrowing means
    #: narrowing all three and widening what comes back.
    let made = if bias < 0 {
        widen(state, __prim__.graph_conv(b, narrow(state, input.value), narrow(state, kernel), -1, settings))
    } else {
        widen(state, __prim__.graph_conv(b, narrow(state, input.value), narrow(state, kernel), narrow(state, bias), settings))
    }
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

#: How to express a batch normalisation, or `0` not to.
#:
#: Kept a knob because which of the two shapes the framework likes is not
#: something to guess at: `1` reduces over the three leading axes of the
#: NHWC block as it stands, `2` folds those three into one first so the
#: reduction runs down the rows of a plain matrix.
fn batch_norm_route() -> int {
    let text = env("JAITENSOR_BN_GRAPH")
    if text is null { return BN_GRAPH_DEFAULT }
    return int(text)
}

#: Per-channel normalisation over a stack of pictures.
#:
#: The layer holds no running average -- it normalises by the batch's own
#: statistics whether it is training or not -- so there is no state for the
#: graph to carry forward and the forward pass is the whole of it.
fn emit_batch_norm(state: Build, layer: any, input: Built) -> Built? {
    if not layer is BatchNorm2d { return null }
    let route = batch_norm_route()
    if route == 0 {
        state.refuse("a batch normalisation, which is faster left to its kernels")
        return null
    }
    if input.shape.len() != 4 {
        state.refuse("a batch normalisation over something that is not a stack of pictures")
        return null
    }
    let params = layer.parameters()
    if params.len() != 2 {
        state.refuse("a batch normalisation without both a scale and a shift")
        return null
    }
    let gamma = state.parameter(params[0])
    let beta = state.parameter(params[1])
    if route == 1 {
        let made = __prim__.graph_layer_norm(
            state.builder,
            input.value,
            gamma,
            beta,
            [0, 1, 2],
            1.0e-5
        )
        if made < 0 {
            state.refuse("a batch normalisation the platform would not build")
            return null
        }
        return Built(made, input.shape)
    }
    #: The three leading axes folded into one. A reduction down the rows of a
    #: matrix is the shape the framework is tuned for; the same reduction
    #: written as three axes of a four-dimensional block strides over the
    #: channel it is summing and never reads two neighbours together.
    let rows = input.shape[0] * input.shape[1] * input.shape[2]
    let channels = input.shape[3]
    let flat = __prim__.graph_reshape(state.builder, input.value, [rows, channels])
    if flat < 0 {
        state.refuse("a batch normalisation the platform would not reshape")
        return null
    }
    let made = __prim__.graph_layer_norm(state.builder, flat, gamma, beta, [0], 1.0e-5)
    if made < 0 {
        state.refuse("a batch normalisation the platform would not build")
        return null
    }
    let back = __prim__.graph_reshape(state.builder, made, input.shape)
    if back < 0 {
        state.refuse("a batch normalisation the platform would not reshape back")
        return null
    }
    return Built(back, input.shape)
}

#: Batch normalisation is expressible and is left out anyway. Four ways have
#: been tried now, all correct, all slower on the `norm` benchmark than not
#: compiling at all. At the hard level (small and wide, seconds):
#:
#:   the framework's own normalisation, over three axes    2.05 s and 3.28 s
#:   mean, variance and the affine built out of primitives 2.02 s and 3.65 s
#:   not expressing it, so the model stays on its kernels  1.49 s and 2.58 s
#:
#: and at the medium level, where `emit_batch_norm` above can still be asked
#: for either shape with `JAITENSOR_BN_GRAPH`:
#:
#:   1, the three leading axes reduced where they lie     0.710 s and 1.195 s
#:   2, those three folded into one first                 0.691 s and 1.205 s
#:   0, not expressing it                                 0.527 s and 0.917 s
#:
#: Route 2 was the interesting one and it did not pay. A reduction over three
#: axes of an NHWC block strides over the channel it is summing, so the guess
#: was that flattening it into a reduction down the rows of a plain matrix --
#: the shape the framework is tuned for, and the shape the eager kernels had
#: to be rewritten into for the same reason -- would fix it. It made no
#: difference, which says the cost is not in the reduction.
#:
#: Where it is: the same model with the layer removed entirely compiles to
#: 0.250 s, so expressing the layer costs 0.44 s of graph against 0.074 s of
#: kernels. Six times, for a mean and a variance over data the convolution
#: has just touched. That points at the gradient the framework builds for it
#: rather than at the forward, and until that is what gets measured there is
#: no reason to expect a fifth shape of the same expression to be different.
#:
#: Expressing a layer here is not free to get wrong: it decides whether the
#: model compiles at all, and compiling this one replaces hand-written kernels
#: that beat the graph. Worth revisiting only if that stops being true.
#:
#: What it costs to leave out, taken apart on the same benchmark at the
#: medium level (small and wide, seconds):
#:
#:   with the normalisation, compiled step allowed     0.500 and 0.860
#:   with it, compiled step turned off                 0.497 and 0.858
#:   without it, compiled step allowed                 0.233 and 0.538
#:   without it, compiled step turned off              0.423 and 0.721
#:
#: The first two lines being the same is the whole point: one layer this
#: cannot express takes the model off the graph entirely. The layer's own
#: kernels are cheap -- 0.074 and 0.137 of those seconds -- and the route
#: they cost is dear, 0.190 and 0.183. So an expression of it has to come in
#: under about three and a half times what the kernels cost to be worth
#: having, and both of the ones tried came in far above that.
fn emit_layer(state: Build, layer: Layer, input: Built) -> Built? {
    if layer is Dense { return emit_dense(state, layer, input) }
    if layer is Flatten { return emit_flatten(state, layer, input) }
    if layer is Conv2d { return emit_conv(state, layer, input) }
    if layer is MaxPool2d { return emit_pool(state, layer, input) }
    if layer is AvgPool2d { return emit_avg_pool(state, layer, input) }
    if layer is LayerNorm { return emit_layer_norm(state, layer, input) }
    if layer is BatchNorm2d { return emit_batch_norm(state, layer, input) }
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
    #: Which of `carried` belong to the optimizer's own table rather than to
    #: this step; see `compile_step`'s `shared_state`. Those are not freed here.
    var borrowed: list[bool]
    #: Whether `carried[0]` is Adam's step count and is to be kept in step with
    #: the optimizer's table, by `resume` and `settle`.
    var shares_count: bool
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
        borrowed: list[bool],
        shares_count: bool,
        batch_size: int,
        classes: int
    ) {
        self.handle = handle
        self.params = params
        self.shadow = shadow
        self.carried = carried
        self.carried_shadow = carried_shadow
        self.borrowed = borrowed
        self.shares_count = shares_count
        self.live_is_shadow = false
        self.batch_size = batch_size
        self.classes = classes
        self.freed = false
    }

    #: Take up Adam's step count where the optimizer's table has it, before
    #: the first step of a run. Another path may have stepped the parameters
    #: since this step last did, and the bias correction reads the count.
    pub fn resume(self) -> void {
        if not self.shares_count { return }
        self.settle_values()
        self.carried[0].upload([float(steps_taken(self.params[0]))])
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
    #: Adam's step count goes back to the optimizer's table too, when this
    #: step shares it.
    pub fn settle(self) -> void {
        self.settle_values()
        if self.shares_count {
            set_steps_taken(self.params, int(self.carried[0].download()[0]))
        }
    }

    fn settle_values(self) -> void {
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
        for (index, buffer) in self.carried.enumerate() {
            if not self.borrowed[index] { buffer.free() }
        }
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

#: Turn the compiled step on or off from inside the process.
#:
#: The environment variable is read once and kept, which is right for a
#: benchmark and useless for a test: comparing the two paths means running
#: both of them in one run, and the only way to do that was two processes
#: comparing numbers through a file.
pub fn set_compiled_step(on: bool) -> void { _allowed = if on { 1 } else { 0 } }

#: Whether the compiled step is currently allowed, so that a caller which
#: turns it off can put it back the way it found it.
pub fn compiled_step_allowed() -> bool { return allowed() }

var _route_policy: str? = null

#: How a step's plan picks between encoding into the shared command buffer
#: and letting the platform run it.
#:
#: A step is called back to back, a batch at a time, so what matters is how
#: many go through a second, not how long one takes: encoding returns at once
#: and lets the next batch's staging overlap this one's arithmetic, where a run
#: blocks on every call. The plan's own default times one call at a time, and
#: on a [784-512-512-10] step the two read within noise that way, so the route
#: was a coin toss -- 273 ms in one process and 176-186 in three, 302 ms run
#: and 202 encode when each was forced. "throughput" asks the plan to time
#: calls back to back instead.
#:
#: `JAITENSOR_STEP_ROUTE_POLICY=latency` asks for the one-call rule, for an A/B
#: in one binary. Read once.
fn route_policy() -> str {
    if _route_policy is null {
        _route_policy = if (env("JAITENSOR_STEP_ROUTE_POLICY") ?? "") == "latency" {
            "latency"
        } else {
            "throughput"
        }
    }
    return _route_policy ?? "throughput"
}

#: Compile a step's plan under `route_policy`.
#:
#: A runtime whose `graph_compile` takes no policy rejects the fourth argument
#: with a TypeError before it builds anything. Such a runtime only has the
#: one-call rule, so the step is compiled under that. Any other TypeError is a
#: fault of its own and goes on up: retrying it would build the graph twice and
#: quietly take the other rule.
fn compile_plan(builder: int, inputs: list[int], outputs: list[int]) -> int {
    try {
        return __prim__.graph_compile(builder, inputs, outputs, route_policy())
    } catch error: TypeError {
        if not error.message.contains("takes 3 arguments") { throw error }
        return __prim__.graph_compile(builder, inputs, outputs)
    }
}

var _flat = -1

fn flat_policy() -> int {
    if _flat < 0 {
        let text = env("JAITENSOR_COMPILED_FLAT") ?? "1"
        _flat = if text == "0" { 0 } elif text == "2" { 2 } else { 1 }
    }
    return _flat
}

#: Whether a model fitted on a flat (rank-2) dataset trains through a compiled
#: step.
#:
#: Only images ever did, so a dense model on flat rows missed it whenever it
#: missed the hand-fused shapes -- two hidden layers, a width under 256, an
#: optimiser carrying state -- and fell to the static path: thirty-odd kernels
#: a step where this is one graph. An image model is the same graph behind a
#: Flatten that is only a reshape. Measured 2026-10-05 on fashion, three
#: epochs, under the GPU lock (n=5, alternated): [784-512-512-10] SGD b512
#: 267 ms static to 171 compiled, [784-128-64-10] momentum b64 1242 to 989,
#: [784-32-10] SGD b32 518 to 330. Plain descent and momentum add two or three
#: graph operations a parameter, and no shape tried with them was slower
#: compiled, so for them there is no size floor.
#:
#: Adam is gated on the work in a step, `work` -- the batch times the
#: parameter count, roughly the multiply-adds of one forward pass. Its update
#: is a dozen graph operations a parameter where the static path runs one
#: kernel, and when the arithmetic is small the step is all host work, so
#: encoding those operations decides it, and which side wins moves with
#: whatever else the machine is doing. [784-128-64-10] Adam b64 (7M) read
#: 1267 ms static against 1468 compiled in one session of five alternated
#: rounds, every round the same way, and 947 against 739 in another, again
#: every round. Under a load average of 10-40 (n=4, paired ratios
#: static/compiled): 7M 0.82x, [784-512-512-10] b64 (43M) 0.58x, b512
#: (343M) 0.73x, and only b2048 (1.4G) won, 1.18x (1.02-1.43x). A step that
#: only sometimes wins is not taken: below `ADAM_FLAT_MIN_WORK` an Adam model
#: stays on the static path it had before flat data compiled.
#:
#: Also gated is the batch that does not fill: a compiled step is built for
#: one batch size and leaves a short last batch untrained, which the static
#: path does not. Flat data compiles only when that tail is at most a
#: sixty-fourth of the epoch -- 96 of fashion's 60000 rows at b512 -- so a
#: small table with a large batch keeps every row it has.
#:
#: `JAITENSOR_COMPILED_FLAT`: 0 never compiles for a flat dataset (the old
#: rule), 2 compiles whatever the tail and the work, and anything else is the
#: default above. Read once; `set_flat_policy` overrides it in-process.
pub fn flat_step_wanted(samples: int, batch_size: int, adam: bool = false, work: int = 0) -> bool {
    let policy = flat_policy()
    if policy == 0 { return false }
    if policy == 2 { return true }
    if batch_size <= 0 or samples < batch_size { return false }
    if samples % batch_size * 64 > samples { return false }
    return not adam or work >= ADAM_FLAT_MIN_WORK
}

#: The least work a step (batch times parameters) must carry before an Adam
#: model on flat rows is compiled; see `flat_step_wanted`.
pub let ADAM_FLAT_MIN_WORK = 1024 * 1024 * 1024

var _fused_mlp = -1

#: Whether a model the fused forward-backward kernels take, trained with
#: momentum or Adam, keeps that path rather than a compiled step.
#:
#: That path fuses the forward and backward passes and then runs the update
#: as one kernel per parameter; the compiled step fuses the update into the
#: graph and carries the optimiser's state through it. The graph loses:
#: [784-256-10] on fashion at b512, three epochs, measured 2026-10-05 under
#: the GPU lock (n=5, alternated), Adam 88.0 ms fused against 119.7 compiled
#: and momentum 85.9 against 92.1 -- Adam's update is a dozen graph operations
#: a parameter, and encoding them costs more than the kernels they replace.
#: So the fused kernels are kept. `JAITENSOR_FUSED_MLP_STEP=0` offers them to
#: the compiled step instead, which takes them only where `flat_step_wanted`
#: does: momentum always, but Adam only past `ADAM_FLAT_MIN_WORK`, so a small
#: Adam model stays fused unless `JAITENSOR_COMPILED_FLAT=2` lifts that gate
#: too. Read once; `set_prefer_fused_mlp_step` overrides it in-process.
pub fn prefer_fused_mlp_step() -> bool {
    if _fused_mlp < 0 {
        _fused_mlp = if (env("JAITENSOR_FUSED_MLP_STEP") ?? "1") == "0" { 0 } else { 1 }
    }
    return _fused_mlp == 1
}

#: Override `JAITENSOR_FUSED_MLP_STEP` from inside the process. Returns the
#: preference it replaced.
pub fn set_prefer_fused_mlp_step(prefer: bool) -> bool {
    let was = prefer_fused_mlp_step()
    _fused_mlp = if prefer { 1 } else { 0 }
    return was
}

#: Override `JAITENSOR_COMPILED_FLAT` from inside the process (0, 1 or 2), so
#: a test can compare the paths in one run. Returns the policy it replaced.
pub fn set_flat_policy(policy: int) -> int {
    let was = flat_policy()
    _flat = policy
    return was
}

#: Build a compiled step for `layers`, or null when they cannot be expressed.
#:
#: `input_shape` is one batch's worth including the batch dimension. SGD with
#: or without momentum is expressed; momentum's velocity is carried through the
#: graph as another pair of tensors, fed in and produced, the same way the
#: parameters are.
#:
#: With `shared_state` the step carries the optimizer table's own velocity or
#: moments, and keeps Adam's step count in step with it (`resume`, `settle`),
#: where without it every step starts its optimiser from rest. A model needs
#: it: it moves between this path and the others as the batch size changes,
#: and the state is the same state whichever path is stepping.
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
    epsilon: float = 1.0e-7,
    shared_state: bool = false
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

    let handle = compile_plan(builder, inputs, outputs)
    if handle < 0 {
        _refusal = "the step would not compile"
        return null
    }
    var shadow: list[Buffer] = []
    for param in state.params { shadow.push(Buffer(param.size)) }
    #: In the order the graph carries them: Adam's step count first, then
    #: each parameter's velocity or pair of moments.
    var table: list[Buffer] = []
    if shared_state and (adam or momentum != 0.0) {
        let kind = if adam { 1 } else { 0 }
        for param in state.params { table.extend(moment_buffers(param, kind)) }
    }
    let skip = if adam { 1 } else { 0 }
    #: Everything the step owns starts at rest, including the step count.
    var held: list[Buffer] = []
    var held_shadow: list[Buffer] = []
    var borrowed: list[bool] = []
    for (index, shape) in carried_sizes.enumerate() {
        var size = 1
        for dimension in shape { size = size * dimension }
        let slot = index - skip
        if slot >= 0 and slot < table.len() {
            held.push(table[slot])
            borrowed.push(true)
        } else {
            let first = Buffer(size)
            first.fill_zero()
            held.push(first)
            borrowed.push(false)
        }
        let second = Buffer(size)
        second.fill_zero()
        held_shadow.push(second)
    }
    _refusal = null
    return CompiledStep(
        handle,
        state.params,
        shadow,
        held,
        held_shadow,
        borrowed,
        shared_state and adam,
        input_shape[0],
        classes
    )
}
