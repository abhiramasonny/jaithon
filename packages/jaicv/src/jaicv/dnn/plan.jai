#: Compiling a whole imported model into one graph, instead of running its
#: operators one at a time.
#:
#: `run_graph` walks a model node by node, allocating a tensor between each
#: pair. That is the only thing to do when a graph's shapes are not known until
#: it runs, and it stays the fallback. But when the shapes ARE known -- which
#: for a detector fed fixed-size images they always are -- handing the whole
#: thing to the platform's graph compiler is a different proposition. It fuses
#: across operators, keeps intermediates out of device memory, and decides for
#: itself which parts of the work belong on the GPU, on the neural accelerator
#: and on the CPU. None of that is reachable an operator at a time.
#:
#: Two things make it possible. The first is a survey run: one pass through the
#: interpreter records what shape every value is, which is far more reliable
#: than reimplementing shape inference for sixty operators. The second is
#: constant folding: an exported model computes its reshapes and its slice
#: bounds from `Shape` nodes, and those have to become numbers before a static
#: graph can be built. A value is folded only when nothing it descends from is
#: an input -- so a model that computed a shape from its data, rather than from
#: the shape of its data, is left to the interpreter rather than silently
#: baked at the wrong size.
#:
#: Anything unsupported means no plan at all, and the caller runs the operators
#: as before. Partial compilation would mean handing the compiler fragments,
#: which is most of the cost and little of the benefit.
import jaitensor as jt
from jaitensor import Tensor
from .graph import Graph, GraphNode, attr_float, attr_int, attr_ints, attr_str
from .graph import normalize_axis, run_graph
from std.os import env

#: How many elements a value may hold and still be folded at compile time.
#: Shape arithmetic deals in a handful of integers; this is loose enough for
#: any of it and tight enough that no real tensor is ever evaluated twice.
let FOLD_LIMIT = 4096

#: The unary operators the builder knows, by the code it takes.
let UNARY_CODES = {
    "Relu": 0,
    "Sigmoid": 1,
    "Tanh": 2,
    "Exp": 3,
    "Log": 4,
    "Sqrt": 5,
    "Neg": 6,
    "Abs": 7,
    "Erf": 8,
    "Silu": 9,
    "Floor": 10,
    "Ceil": 11,
    "Reciprocal": 12,
    "Sqr": 13,
}

let BINARY_CODES = {"Add": 0, "Sub": 1, "Mul": 2, "Div": 3, "Max": 4, "Min": 5, "Pow": 6}

let REDUCE_CODES = {"ReduceSum": 0, "ReduceMean": 1, "ReduceMax": 2, "ReduceMin": 3}

#: The ONNX element types, as the importer numbers them.
let INTEGER_TYPES = [2, 3, 4, 5, 6, 7, 12, 13]
let FLOAT_TYPES = [1, 10, 11, 16]

#: How a nearest resize picks its source pixel, in the builder's numbering.
let ROUNDING_CODES = {
    "round_prefer_ceil": 0,
    "round_prefer_floor": 1,
    "ceil": 2,
    "floor": 3,
}

#: Operators whose result depends only on the shapes of things, and which are
#: therefore worth evaluating once at compile time rather than every run.
let FOLDABLE = {
    "Shape": true,
    "Constant": true,
    "ConstantOfShape": true,
    "Gather": true,
    "Unsqueeze": true,
    "Squeeze": true,
    "Concat": true,
    "Slice": true,
    "Reshape": true,
    "Cast": true,
    "Add": true,
    "Sub": true,
    "Mul": true,
    "Div": true,
    "Mod": true,
    "Equal": true,
    "Where": true,
    "Range": true,
}

#: A model compiled to one graph, ready to be fed.
pub class Plan {
    let handle: int
    pub let inputs: list[str]
    pub let input_shapes: list[list[int]]
    pub let outputs: list[str]
    pub let output_shapes: list[list[int]]
    var freed: bool

    pub fn init(
        self,
        handle: int,
        inputs: list[str],
        input_shapes: list[list[int]],
        outputs: list[str],
        output_shapes: list[list[int]]
    ) {
        self.handle = handle
        self.inputs = [name for name in inputs]
        self.input_shapes = [[d for d in shape] for shape in input_shapes]
        self.outputs = [name for name in outputs]
        self.output_shapes = [[d for d in shape] for shape in output_shapes]
        self.freed = false
    }

    #: Whether this plan was built for the shapes it is being handed. A plan is
    #: compiled for one set of sizes and is wrong for any other, so a caller
    #: that changes them gets the interpreter back rather than a wrong answer.
    pub fn fits(self, feeds: dict[str, Tensor]) -> bool {
        for (index, name) in self.inputs.enumerate() {
            let fed = feeds.get(name, null)
            if fed is null { return false }
            if fed.shape != self.input_shapes[index] { return false }
        }
        return true
    }

    #: Feed the plan and get its outputs, by the names the model gave them.
    pub fn run(self, feeds: dict[str, Tensor]) -> dict[str, Tensor] {
        if self.freed { throw RuntimeError("this Plan has already been freed") }
        var handles: list[int] = []
        for name in self.inputs {
            let fed = feeds.get(name, null)
            if fed is null { throw ValueError(f"the plan's input {name} was not fed") }
            handles.push(fed.buffer().handle_of())
        }
        var produced: dict[str, Tensor] = {}
        var out_handles: list[int] = []
        for (index, name) in self.outputs.enumerate() {
            let tensor = Tensor(self.output_shapes[index])
            produced[name] = tensor
            out_handles.push(tensor.buffer().handle_of())
        }
        __prim__.graph_run(self.handle, handles, out_handles)
        return produced
    }

    pub fn free(self) -> void {
        if self.freed { return }
        self.freed = true
        __prim__.graph_plan_free(self.handle)
    }
}

#: Which values a compiler may treat as constant.
#:
#: An initializer is one, and so is anything every one of whose inputs is one.
#: A graph input is not, and neither is anything downstream of one -- that is
#: the whole point, since a reshape whose target came from the data would be
#: baked at whatever size the survey happened to see.
fn static_values(graph: Graph, order: list[GraphNode]) -> dict[str, bool] {
    var known: dict[str, bool] = {}
    for (name, _weight) in graph.initializers.items() { known[name] = true }
    for node in order {
        var all_known = true
        for name in node.inputs {
            if name.len() == 0 { continue }
            if not known.get(name, false) { all_known = false }
        }
        #: `Shape` reads only the shape of its input, so it is constant even
        #: when that input is not -- which is what makes the rest foldable.
        let constant = node.op == "Shape" or node.op == "Constant" or
            (all_known and FOLDABLE.get(node.op, false))
        if not constant { continue }
        for name in node.outputs {
            if name.len() > 0 { known[name] = true }
        }
    }
    return known
}

#: Every dimension of `shape` as one flat list, for a builder that takes them
#: that way.
fn dims_of(shape: list[int]) -> list[int] { return [value for value in shape] }

fn as_ints(values: list[float]) -> list[int] {
    var out: list[int] = []
    for value in values { out.push(int(value)) }
    return out
}

#: The state a single compilation carries. Kept in one place so the operator
#: cases below read as translations rather than as bookkeeping.
class Build {
    pub var builder: int
    pub var ids: dict[str, int]
    pub var shapes: dict[str, list[int]]
    pub var constants: dict[str, list[float]]
    pub var known: dict[str, bool]

    pub fn init(
        self,
        builder: int,
        shapes: dict[str, list[int]],
        constants: dict[str, list[float]],
        known: dict[str, bool]
    ) {
        self.builder = builder
        self.ids = {}
        self.shapes = shapes
        self.constants = constants
        self.known = known
    }

    #: The builder id for a value, making a constant for it the first time a
    #: folded one is actually needed as a tensor.
    pub fn id_of(self, name: str) -> int {
        let existing = self.ids.get(name, null)
        if existing is not null { return existing }
        let values = self.constants.get(name, null)
        let shape = self.shapes.get(name, null)
        if values is null or shape is null { return -1 }
        let id = __prim__.graph_constant(self.builder, values, dims_of(shape))
        self.ids[name] = id
        return id
    }

    pub fn shape_of(self, name: str) -> list[int]? { return self.shapes.get(name, null) }
}

#: One node, translated into the graph under construction. Returns false when
#: the operator is one this compiler cannot express, which abandons the whole
#: compilation.
fn emit(state: Build, node: GraphNode) -> bool {
    let b = state.builder
    let out = if node.outputs.len() > 0 { node.outputs[0] } else { "" }

    let unary = UNARY_CODES.get(node.op, -1)
    if unary >= 0 and node.inputs.len() == 1 {
        state.ids[out] = __prim__.graph_unary(b, state.id_of(node.inputs[0]), unary)
        return true
    }
    let binary = BINARY_CODES.get(node.op, -1)
    if binary >= 0 and node.inputs.len() == 2 {
        state.ids[out] = __prim__.graph_binary(
            b,
            state.id_of(node.inputs[0]),
            state.id_of(node.inputs[1]),
            binary
        )
        return true
    }

    if node.op == "Conv" {
        let x = state.shape_of(node.inputs[0])
        let w = state.shape_of(node.inputs[1])
        if x is null or w is null or x.len() != 4 or w.len() != 4 { return false }
        let window = [w[2], w[3]]
        let strides = attr_ints(node, "strides", [1, 1])
        let dilations = attr_ints(node, "dilations", [1, 1])
        let pads = resolved_pads(node, [x[2], x[3]], window, strides, dilations)
        if pads is null { return false }
        let bias = if node.inputs.len() > 2 { state.id_of(node.inputs[2]) } else { -1 }
        state.ids[out] = __prim__.graph_conv(
            b,
            state.id_of(node.inputs[0]),
            state.id_of(node.inputs[1]),
            bias,
            [
                strides[0],
                strides[1],
                pads[0],
                pads[2],
                pads[1],
                pads[3],
                dilations[0],
                dilations[1],
                attr_int(node, "group", 1),
            ]
        )
        return true
    }

    if node.op == "MaxPool" or node.op == "AveragePool" {
        let x = state.shape_of(node.inputs[0])
        if x is null or x.len() != 4 { return false }
        let window = attr_ints(node, "kernel_shape", [1, 1])
        let strides = attr_ints(node, "strides", [1, 1])
        let pads = resolved_pads(node, [x[2], x[3]], window, strides, [1, 1])
        if pads is null { return false }
        state.ids[out] = __prim__.graph_pool(
            b,
            state.id_of(node.inputs[0]),
            [window[0], window[1], strides[0], strides[1], pads[0], pads[2], pads[1], pads[3]],
            if node.op == "MaxPool" { 0 } else { 1 }
        )
        return true
    }

    if node.op == "Concat" {
        var parts: list[int] = []
        for name in node.inputs { parts.push(state.id_of(name)) }
        let rank = state.shape_of(node.inputs[0])
        if rank is null { return false }
        state.ids[out] = __prim__.graph_concat(
            b,
            parts,
            normalize_axis(attr_int(node, "axis", 0), rank.len(), node.label())
        )
        return true
    }

    if node.op == "Reshape" or node.op == "Squeeze" or node.op == "Unsqueeze" or
        node.op == "Flatten" {
        let shape = state.shape_of(out)
        if shape is null { return false }
        state.ids[out] = __prim__.graph_reshape(b, state.id_of(node.inputs[0]), dims_of(shape))
        return true
    }

    if node.op == "Transpose" {
        let x = state.shape_of(node.inputs[0])
        if x is null { return false }
        var perm = attr_ints(node, "perm", [])
        if perm.len() == 0 {
            for index in 0..x.len() { perm.push(x.len() - 1 - index) }
        }
        state.ids[out] = __prim__.graph_transpose(b, state.id_of(node.inputs[0]), perm)
        return true
    }

    if node.op == "Softmax" {
        let x = state.shape_of(node.inputs[0])
        if x is null { return false }
        state.ids[out] = __prim__.graph_softmax(
            b,
            state.id_of(node.inputs[0]),
            normalize_axis(attr_int(node, "axis", -1), x.len(), node.label())
        )
        return true
    }

    if node.op == "Cast" {
        #: Every tensor here is float, so a cast to a float type is nothing at
        #: all; a cast to an integer one truncates, and a cast to boolean is
        #: one where the value was not zero. The same three cases the
        #: interpreter has.
        let target = attr_int(node, "to", 1)
        if target in FLOAT_TYPES {
            state.ids[out] = state.id_of(node.inputs[0])
            return true
        }
        let code = if target in INTEGER_TYPES { 15 } elif target == 9 { 16 } else { -1 }
        if code < 0 { return false }
        state.ids[out] = __prim__.graph_unary(b, state.id_of(node.inputs[0]), code)
        return true
    }

    if node.op == "Identity" or node.op == "Dropout" {
        state.ids[out] = state.id_of(node.inputs[0])
        return true
    }

    if node.op == "Resize" or node.op == "Upsample" {
        let want = state.shape_of(out)
        if want is null or want.len() != 4 { return false }
        if attr_str(node, "mode", "nearest") != "nearest" { return false }
        #: Only the coordinate rules that map exactly are compiled. Half-pixel
        #: sampling is left to the interpreter rather than approximated, since
        #: being one pixel out is the kind of difference that survives a test
        #: and shows up in a detection.
        let placing = attr_str(node, "coordinate_transformation_mode", "half_pixel")
        var center = 0
        var corners = 0
        if placing == "asymmetric" {
            center = 0
            corners = 0
        } elif placing == "align_corners" {
            corners = 1
        } else {
            return false
        }
        let rounding = ROUNDING_CODES.get(attr_str(node, "nearest_mode", "round_prefer_floor"), -1)
        if rounding < 0 { return false }
        state.ids[out] = __prim__.graph_resize_nearest(
            b,
            state.id_of(node.inputs[0]),
            want[2],
            want[3],
            rounding,
            center,
            corners
        )
        return true
    }

    if node.op == "LayerNormalization" {
        let x = state.shape_of(node.inputs[0])
        if x is null { return false }
        let raw = attr_int(node, "axis", -1)
        let axis = if raw < 0 { x.len() + raw } else { raw }
        if axis < 0 or axis >= x.len() { return false }
        var axes: list[int] = []
        for index in axis..x.len() { axes.push(index) }
        let gamma = if node.inputs.len() > 1 { state.id_of(node.inputs[1]) } else { -1 }
        let beta = if node.inputs.len() > 2 and node.inputs[2].len() > 0 {
            state.id_of(node.inputs[2])
        } else {
            -1
        }
        state.ids[out] = __prim__.graph_layer_norm(
            b,
            state.id_of(node.inputs[0]),
            gamma,
            beta,
            axes,
            attr_float(node, "epsilon", 1.0e-5)
        )
        return true
    }

    if node.op == "Gemm" {
        let c = if node.inputs.len() > 2 and node.inputs[2].len() > 0 {
            state.id_of(node.inputs[2])
        } else {
            -1
        }
        state.ids[out] = __prim__.graph_gemm(
            b,
            state.id_of(node.inputs[0]),
            state.id_of(node.inputs[1]),
            c,
            attr_int(node, "transA", 0),
            attr_int(node, "transB", 0),
            attr_float(node, "alpha", 1.0),
            attr_float(node, "beta", 1.0)
        )
        return true
    }

    if node.op == "ReduceMean" or node.op == "ReduceSum" or node.op == "ReduceMax" or
        node.op == "ReduceMin" {
        let x = state.shape_of(node.inputs[0])
        let want = state.shape_of(out)
        if x is null or want is null { return false }
        var axes = attr_ints(node, "axes", [])
        if axes.len() == 0 and node.inputs.len() > 1 {
            let given = state.constants.get(node.inputs[1], null)
            if given is null { return false }
            axes = as_ints(given)
        }
        if axes.len() == 0 {
            for index in 0..x.len() { axes.push(index) }
        }
        var over: list[int] = []
        for axis in axes { over.push(normalize_axis(axis, x.len(), node.label())) }
        let kind = REDUCE_CODES.get(node.op, -1)
        if kind < 0 { return false }
        let reduced = __prim__.graph_reduce(b, state.id_of(node.inputs[0]), over, kind)
        #: MPSGraph keeps the reduced axes as ones; ONNX drops them unless
        #: `keepdims` says otherwise, and the survey already knows which.
        state.ids[out] = if want.len() == x.len() {
            reduced
        } else {
            __prim__.graph_reshape(b, reduced, dims_of(want))
        }
        return true
    }

    if node.op == "Gather" {
        let x = state.shape_of(node.inputs[0])
        let want = state.shape_of(out)
        if x is null or want is null { return false }
        let picked = __prim__.graph_gather(
            b,
            state.id_of(node.inputs[0]),
            state.id_of(node.inputs[1]),
            normalize_axis(attr_int(node, "axis", 0), x.len(), node.label())
        )
        #: ONNX drops the axis for a scalar index and keeps it for a vector
        #: one; the survey already knows which happened, so the result is
        #: reshaped to whatever it should have been.
        state.ids[out] = __prim__.graph_reshape(b, picked, dims_of(want))
        return true
    }

    if node.op == "MatMul" {
        state.ids[out] = __prim__.graph_matmul(
            b,
            state.id_of(node.inputs[0]),
            state.id_of(node.inputs[1])
        )
        return true
    }

    #: `Split` and `Slice` both come out as slices, which is what they are; the
    #: shape survey has already worked out how wide each piece is.
    if node.op == "Split" {
        let x = state.shape_of(node.inputs[0])
        if x is null { return false }
        let axis = normalize_axis(attr_int(node, "axis", 0), x.len(), node.label())
        var at = 0
        for name in node.outputs {
            let piece = state.shape_of(name)
            if piece is null { return false }
            var starts: list[int] = []
            var ends: list[int] = []
            var steps: list[int] = []
            for index in 0..x.len() {
                starts.push(if index == axis { at } else { 0 })
                ends.push(if index == axis { at + piece[axis] } else { x[index] })
                steps.push(1)
            }
            state.ids[name] = __prim__.graph_slice(
                b,
                state.id_of(node.inputs[0]),
                starts,
                ends,
                steps
            )
            at += piece[axis]
        }
        return true
    }

    if node.op == "Slice" {
        let x = state.shape_of(node.inputs[0])
        let want = state.shape_of(out)
        if x is null or want is null { return false }
        let bounds = slice_bounds(state, node, x, want)
        if bounds is null { return false }
        state.ids[out] = __prim__.graph_slice(
            b,
            state.id_of(node.inputs[0]),
            bounds[0],
            bounds[1],
            bounds[2]
        )
        return true
    }

    return false
}

#: A `Slice` node's per-axis bounds, resolved against the input's real shape.
#:
#: ONNX writes an open end as the largest integer there is and allows negative
#: indices; the builder wants neither, so both are settled here. Only forward
#: slices are compiled -- a negative step is rare enough that leaving it to the
#: interpreter costs nothing.
fn slice_bounds(
    state: Build,
    node: GraphNode,
    shape: list[int],
    want: list[int]
) -> list[list[int]]? {
    let rank = shape.len()
    var starts = [0 for _index in 0..rank]
    var ends = [shape[index] for index in 0..rank]
    var steps = [1 for _index in 0..rank]

    let given_starts = state.constants.get(node.inputs[1], null)
    let given_ends = if node.inputs.len() > 2 { state.constants.get(node.inputs[2], null) } else { null }
    if given_starts is null or given_ends is null { return null }
    var axes: list[int] = []
    if node.inputs.len() > 3 and node.inputs[3].len() > 0 {
        let given = state.constants.get(node.inputs[3], null)
        if given is null { return null }
        axes = as_ints(given)
    } else {
        for index in 0..given_starts.len() { axes.push(index) }
    }
    if node.inputs.len() > 4 and node.inputs[4].len() > 0 {
        let given = state.constants.get(node.inputs[4], null)
        if given is null { return null }
        for value in as_ints(given) {
            if value != 1 { return null }
        }
    }

    for (slot, raw_axis) in axes.enumerate() {
        let axis = normalize_axis(raw_axis, rank, node.label())
        let width = shape[axis]
        var first = int(given_starts[slot])
        var last = int(given_ends[slot])
        if first < 0 { first += width }
        if last < 0 { last += width }
        if first < 0 { first = 0 }
        if first > width { first = width }
        if last < 0 { last = 0 }
        if last > width { last = width }
        starts[axis] = first
        ends[axis] = last
    }
    #: The survey already knows what came out; if these bounds disagree with
    #: it, something has been read wrongly and the interpreter should have it.
    for index in 0..rank {
        if ends[index] - starts[index] != want[index] { return null }
    }
    return [starts, ends, steps]
}

#: A convolution or pooling node's four padding values, or nothing when the
#: node asks for a padding this compiler does not express.
fn resolved_pads(
    node: GraphNode,
    spatial: list[int],
    window: list[int],
    strides: list[int],
    dilations: list[int]
) -> list[int]? {
    let mode = attr_str(node, "auto_pad", "NOTSET")
    if mode == "NOTSET" {
        let pads = attr_ints(node, "pads", [0, 0, 0, 0])
        if pads.len() != 4 { return null }
        return pads
    }
    if mode == "VALID" { return [0, 0, 0, 0] }
    if mode != "SAME_UPPER" and mode != "SAME_LOWER" { return null }
    var pads = [0, 0, 0, 0]
    for axis in 0..2 {
        let reach = (window[axis] - 1) * dilations[axis] + 1
        let out = (spatial[axis] + strides[axis] - 1) // strides[axis]
        let total = max((out - 1) * strides[axis] + reach - spatial[axis], 0)
        let head = if mode == "SAME_UPPER" { total // 2 } else { total - total // 2 }
        pads[axis] = head
        pads[axis + 2] = total - head
    }
    return pads
}

#: Compile `graph` for the shapes `feeds` has, or nothing when it cannot be.
#:
#: The survey run is a real execution of the model, so this costs about what
#: one inference costs; it happens once, and every run after it is the compiled
#: graph. `null` is not a failure to report -- it means the caller should keep
#: using the interpreter, which is always correct.
pub fn compile_plan(graph: Graph, feeds: dict[str, Tensor]) -> Plan? {
    if graph.outputs.len() == 0 { return null }
    let order = graph.schedule()
    let known = static_values(graph, order)

    #: Ask the survey for every static value, which is what the folding needs.
    var wanted: dict[str, list[float]] = {}
    for node in order {
        for name in node.outputs {
            if name.len() > 0 and known.get(name, false) { wanted[name] = [] }
        }
    }
    var shapes: dict[str, list[int]] = {}
    var produced: dict[str, Tensor] = {}
    try {
        produced = run_graph(graph, feeds, shapes, wanted)
    } catch _failure: Error {
        return null
    }
    for (_name, tensor) in produced.items() { tensor.free() }

    #: Weights are constants too, and the survey does not report them because
    #: nothing produced them.
    for (name, weight) in graph.initializers.items() {
        if weight.size <= FOLD_LIMIT { wanted[name] = weight.download() }
    }

    var oversized: list[str] = []
    for (name, values) in wanted.items() {
        if values.len() > FOLD_LIMIT { oversized.push(name) }
    }
    for name in oversized { wanted.remove(name) }

    let builder = __prim__.graph_new()
    var state = Build(builder, shapes, wanted, known)

    #: Weights too large to fold are still constants in the graph, just ones
    #: whose numbers are handed over rather than reasoned about.
    for (name, weight) in graph.initializers.items() {
        if state.constants.get(name, null) is not null { continue }
        state.ids[name] = __prim__.graph_constant(
            builder,
            weight.download(),
            dims_of(weight.shape)
        )
    }
    var inputs: list[int] = []
    for name in graph.inputs {
        let shape = shapes.get(name, null)
        if shape is null {
            __prim__.graph_free(builder)
            return null
        }
        let id = __prim__.graph_input(builder, dims_of(shape))
        state.ids[name] = id
        inputs.push(id)
    }

    for node in order {
        #: A node every one of whose outputs is already a folded constant has
        #: nothing to contribute to the graph.
        var settled = node.outputs.len() > 0
        for name in node.outputs {
            if name.len() == 0 { continue }
            if state.constants.get(name, null) is null { settled = false }
        }
        if settled { continue }
        var ok = false
        try {
            ok = emit(state, node)
        } catch _failure: Error {
            ok = false
        }
        if not ok {
            #: Worth saying which operator stopped it: the answer is still
            #: right, but a model that falls back is running several times
            #: slower than it could, and this is the one fact needed to fix it.
            if env("JAICV_GRAPH_PLAN_REPORT") is not null {
                var told: list[str] = []
                for name in node.inputs {
                    let fixed = state.constants.get(name, null) is not null
                    let shape = state.shapes.get(name, null)
                    told.push(f"{name}{if fixed { ' fixed' } else { ' live' }} {shape}")
                }
                print(f"[plan] {node.label()} is not one this compiler expresses")
                for line in told { print(f"[plan]     {line}") }
            }
            __prim__.graph_free(builder)
            return null
        }
    }

    var outputs: list[int] = []
    for name in graph.outputs {
        if env("JAICV_GRAPH_PLAN_REPORT") is not null and state.id_of(name) < 0 {
            print(f"[plan] the output {name} has no value in the compiled graph")
        }
        #: `id_of` rather than a lookup, so that a graph asked for a value the
        #: folding settled at compile time still compiles -- it becomes a
        #: constant in the graph instead of a hole in it.
        let id = state.id_of(name)
        if id < 0 {
            __prim__.graph_free(builder)
            return null
        }
        outputs.push(id)
    }
    let plan = __prim__.graph_compile(builder, inputs, outputs)
    __prim__.graph_free(builder)
    if plan is null { return null }

    var out_shapes: list[list[int]] = []
    for index in 0..graph.outputs.len() {
        out_shapes.push(__prim__.graph_plan_output_shape(plan, index))
    }
    var in_shapes: list[list[int]] = []
    for name in graph.inputs {
        let shape = shapes.get(name, null)
        in_shapes.push(if shape is null { [] } else { shape })
    }
    return Plan(plan, graph.inputs, in_shapes, graph.outputs, out_shapes)
}


#: The plan already compiled for a graph, by the graph's number.
var _plans: dict[int, Plan] = {}
var _seen: dict[int, int] = {}
var _refused: dict[int, bool] = {}

#: Run a model whichever way is faster for it.
#:
#: The first call interprets: compiling costs about what a couple of hundred
#: inferences do, and a model run once should not pay for it. From the second
#: call on, the graph is compiled and the plan is used for every run after --
#: which is the shape of every real use, since a detector runs its model on
#: frame after frame.
#:
#: A graph the compiler will not take is remembered as such and interpreted
#: from then on without trying again. `JAICV_GRAPH_PLAN=0` keeps it on the
#: interpreter throughout, which is how the two are compared.
pub fn run_best(graph: Graph, feeds: dict[str, Tensor]) -> dict[str, Tensor] {
    if env("JAICV_GRAPH_PLAN") == "0" { return run_graph(graph, feeds) }
    let held = _plans.get(graph.id, null)
    if held is not null {
        if held.fits(feeds) { return held.run(feeds) }
        #: The shapes moved, so the plan is no longer about this call.
        held.free()
        _plans.remove(graph.id)
    }
    if _refused.get(graph.id, false) { return run_graph(graph, feeds) }
    let seen = _seen.get(graph.id, 0) + 1
    _seen[graph.id] = seen
    if seen < 2 { return run_graph(graph, feeds) }
    let built = compile_plan(graph, feeds)
    if built is null {
        _refused[graph.id] = true
        return run_graph(graph, feeds)
    }
    _plans[graph.id] = built
    return built.run(feeds)
}
