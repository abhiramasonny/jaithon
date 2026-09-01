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
from .plan.compile import compile_plan, run_best
from .plan.plan import Plan

export { compile_plan, Plan, run_best }
