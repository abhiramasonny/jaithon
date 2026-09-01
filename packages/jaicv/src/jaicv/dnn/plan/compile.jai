#: Building a plan for a graph, and choosing between it and the interpreter.
from jaitensor import Tensor
from std.os import env
from .build import Build, dims_of, static_values
from .codes import FOLD_LIMIT
from .emit import emit
from .plan import Plan
from ..graph import Graph, run_graph

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
    var state = Build(builder, shapes, wanted, known, graph.initializers)

    #: Weights too large to fold are still constants in the graph, just ones
    #: whose numbers are handed over rather than reasoned about.
    for (name, weight) in graph.initializers.items() {
        if state.constants.get(name, null) is not null { continue }
        state.ids[name] = __prim__.graph_constant(builder, weight.download(), dims_of(weight.shape))
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
        let shape = __prim__.graph_plan_output_shape(plan, index)
        #: A shape the compiler could not settle, or a scalar -- there is no
        #: rank-zero tensor here to receive one. Either way the interpreter
        #: should have this graph.
        if shape.len() == 0 {
            __prim__.graph_plan_free(plan)
            return null
        }
        out_shapes.push(shape)
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
