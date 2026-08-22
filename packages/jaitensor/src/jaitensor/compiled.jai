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
from .layers import Activation, Conv2d, Dense, Flatten, Layer, MaxPool2d
from .tensor import Tensor

#: Reduction kinds the graph builder takes.
const REDUCE_SUM = 0
const REDUCE_MEAN = 1

#: Binary op codes the graph builder takes.
const OP_ADD = 0
const OP_SUB = 1
const OP_MUL = 2

#: Unary op codes the graph builder takes.
const UNARY_RELU = 0
const UNARY_EXP = 3
const UNARY_LOG = 4
const UNARY_NEGATE = 6

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
    let code = layer.activation_code()
    if code == 0 { return Built(biased, shape) }
    if code == 1 { return Built(__prim__.graph_unary(b, biased, UNARY_RELU), shape) }
    state.refuse(f"a dense activation this step cannot express ({code})")
    return null
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
    if layer.activation == Activation.Linear { return Built(made, shape) }
    if layer.activation == Activation.Relu {
        return Built(__prim__.graph_unary(b, made, UNARY_RELU), shape)
    }
    state.refuse(f"a convolution activation this step cannot express ({layer.activation})")
    return null
}

fn emit_pool(state: Build, layer: any, input: Built) -> Built? {
    if not layer is MaxPool2d { return null }
    if input.shape.len() != 4 {
        state.refuse("a pool over something that is not an image batch")
        return null
    }
    let size = layer.kernel_size
    let stride = layer.stride
    #: kernel y, kernel x, stride y, stride x, pad top, bottom, left, right,
    #: ceil mode, pad counts toward an average, channels last.
    let settings = [size, size, stride, stride, 0, 0, 0, 0, 0, 0, 1]
    let made = __prim__.graph_pool(state.builder, input.value, settings, 0)
    if made < 0 {
        state.refuse("a pool the platform would not build")
        return null
    }
    let height = (input.shape[1] - size) // stride + 1
    let width = (input.shape[2] - size) // stride + 1
    return Built(made, [input.shape[0], height, width, input.shape[3]])
}

fn emit_layer(state: Build, layer: Layer, input: Built) -> Built? {
    if layer is Dense { return emit_dense(state, layer, input) }
    if layer is Flatten { return emit_flatten(state, layer, input) }
    if layer is Conv2d { return emit_conv(state, layer, input) }
    if layer is MaxPool2d { return emit_pool(state, layer, input) }
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
    var live_is_shadow: bool
    pub let batch_size: int
    pub let classes: int
    var freed: bool

    pub fn init(
        self,
        handle: int,
        params: list[Tensor],
        shadow: list[Buffer],
        batch_size: int,
        classes: int
    ) {
        self.handle = handle
        self.params = params
        self.shadow = shadow
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
        self.live_is_shadow = false
    }

    pub fn free(self) -> void {
        if self.freed { return }
        self.freed = true
        self.settle()
        for buffer in self.shadow { buffer.free() }
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
#: `input_shape` is one batch's worth including the batch dimension. Only plain
#: SGD is expressed: momentum carries state between steps, which would have to
#: live in the graph too.
pub fn compile_step(
    layers: list[Layer],
    input_shape: list[int],
    classes: int,
    learning_rate: float
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

    #: `param - rate * gradient`, in the graph, so the update is part of the
    #: same compiled thing as the pass that produced the gradient.
    var updated: list[int] = []
    for (index, id) in state.param_ids.enumerate() {
        if grads[index] < 0 {
            _refusal = f"parameter {index} got no gradient"
            __prim__.graph_free(builder)
            return null
        }
        let rate = __prim__.graph_constant(builder, [learning_rate], [1])
        let scaled = __prim__.graph_binary(builder, grads[index], rate, OP_MUL)
        updated.push(__prim__.graph_binary(builder, id, scaled, OP_SUB))
    }

    var inputs: list[int] = [features, labels]
    for id in state.param_ids { inputs.push(id) }
    var outputs: list[int] = [loss, carried.value]
    for id in updated { outputs.push(id) }

    let handle = __prim__.graph_compile(builder, inputs, outputs)
    if handle < 0 {
        _refusal = "the step would not compile"
        return null
    }
    var shadow: list[Buffer] = []
    for param in state.params { shadow.push(Buffer(param.size)) }
    _refusal = null
    return CompiledStep(handle, state.params, shadow, input_shape[0], classes)
}
