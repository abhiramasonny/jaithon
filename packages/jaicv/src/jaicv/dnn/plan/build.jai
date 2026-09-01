#: What a single compilation carries while it runs, and what it may treat as
#: constant.
from jaitensor import Tensor
from .codes import FOLDABLE
from ..graph import Graph, GraphNode

#: Which values a compiler may treat as constant.
#:
#: An initializer is one, and so is anything every one of whose inputs is one.
#: A graph input is not, and neither is anything downstream of one -- that is
#: the whole point, since a reshape whose target came from the data would be
#: baked at whatever size the survey happened to see.
pub fn static_values(graph: Graph, order: list[GraphNode]) -> dict[str, bool] {
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
        let constant = node.op == "Shape" or
            node.op == "Constant" or
            all_known and FOLDABLE.get(node.op, false)
        if not constant { continue }
        for name in node.outputs {
            if name.len() > 0 { known[name] = true }
        }
    }
    return known
}

#: Every dimension of `shape` as one flat list, for a builder that takes them
#: that way.
pub fn dims_of(shape: list[int]) -> list[int] { return [value for value in shape] }

#: A one-element constant, for the operators written as arithmetic against a
#: fixed number.
pub fn scalar(state: Build, value: float) -> int {
    return __prim__.graph_constant(state.builder, [value], [1])
}

pub fn as_ints(values: list[float]) -> list[int] {
    var out: list[int] = []
    for value in values { out.push(int(value)) }
    return out
}

#: The state a single compilation carries. Kept in one place so the operator
#: cases below read as translations rather than as bookkeeping.
pub class Build {
    pub var builder: int
    pub var ids: dict[str, int]
    pub var shapes: dict[str, list[int]]
    pub var constants: dict[str, list[float]]
    pub var known: dict[str, bool]
    pub var weights: dict[str, Tensor]

    pub fn init(
        self,
        builder: int,
        shapes: dict[str, list[int]],
        constants: dict[str, list[float]],
        known: dict[str, bool],
        weights: dict[str, Tensor]
    ) {
        self.builder = builder
        self.ids = {}
        self.shapes = shapes
        self.constants = constants
        self.known = known
        self.weights = weights
    }

    #: A value's numbers, whether they were folded or are simply a weight too
    #: large to have been worth folding. Some operators have to rearrange their
    #: weights at compile time and so need the numbers either way.
    pub fn numbers_of(self, name: str) -> list[float]? {
        let folded = self.constants.get(name, null)
        if folded is not null { return folded }
        let weight = self.weights.get(name, null)
        if weight is null { return null }
        return weight.download()
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
