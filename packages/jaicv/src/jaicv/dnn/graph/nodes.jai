#: What an imported network is made of: an attribute, a node, the graph the
#: two build up, and the readers that pull a typed value back off a node.
from jaitensor import Tensor

from .tensors import host_ints

#: One attribute on a node, in whichever of ONNX's seven shapes it arrived.
#:
#: `kind` says which field to read. An importer that guesses wrong writes a
#: field the executor never looks at, so the readers below check `kind` and
#: name the node when it does not match.
pub class Attr {
    #: `"int"`, `"float"`, `"str"`, `"ints"`, `"floats"`, `"strs"` or `"tensor"`.
    pub var kind: str
    pub var i: int
    pub var f: float
    pub var s: str
    pub var ints: list[int]
    pub var floats: list[float]
    pub var strs: list[str]
    pub var tensor: Tensor?

    pub fn init(
        self,
        kind: str,
        i: int = 0,
        f: float = 0.0,
        s: str = "",
        ints: list[int] = [],
        floats: list[float] = [],
        strs: list[str] = [],
        tensor: Tensor? = null
    ) {
        self.kind = kind
        self.i = i
        self.f = f
        self.s = s
        self.ints = [value for value in ints]
        self.floats = [value for value in floats]
        self.strs = [value for value in strs]
        self.tensor = tensor
    }

    pub static fn of_int(value: int) -> Attr { return Attr("int", i: value) }

    pub static fn of_float(value: float) -> Attr { return Attr("float", f: value) }

    pub static fn of_str(value: str) -> Attr { return Attr("str", s: value) }

    pub static fn of_ints(values: list[int]) -> Attr { return Attr("ints", ints: values) }

    pub static fn of_floats(values: list[float]) -> Attr { return Attr("floats", floats: values) }

    pub static fn of_strs(values: list[str]) -> Attr { return Attr("strs", strs: values) }

    pub static fn of_tensor(value: Tensor) -> Attr { return Attr("tensor", tensor: value) }

    fn __str__(self) -> str { return f"Attr({self.kind})" }
}

#: One operator application. `op` is an ONNX operator name whatever the file
#: it came out of, and `inputs` and `outputs` are the names it reads and
#: writes. An empty input name is ONNX's way of skipping an optional input,
#: and is kept rather than dropped because position is what identifies an
#: argument.
pub class GraphNode {
    pub var op: str
    pub var name: str
    pub var inputs: list[str]
    pub var outputs: list[str]
    pub var attrs: dict[str, Attr]
    #: Work an operator can do once and reuse on every later run of the graph.
    #:
    #: A convolution's weights are an initializer, so the layout swap from
    #: ONNX's `[M, C/g, kH, kW]` to jaitensor's `[kH, kW, C/g, M]` produces the
    #: same tensor every frame -- redoing it measured 12.5 ms a frame on
    #: YOLOv8n, a tenth of the whole graph, for a result that never changed.
    #: `cached` owns its tensors and `free_cached` releases them. `cached_of`
    #: holds the input each entry was computed from, so a caller who feeds a
    #: different tensor next time gets a fresh result rather than a stale one;
    #: those are borrowed, never freed here.
    pub var cached: dict[str, Tensor]
    pub var cached_of: dict[str, Tensor]

    pub fn init(
        self,
        op: str,
        name: str = "",
        inputs: list[str] = [],
        outputs: list[str] = [],
        attrs: dict[str, Attr] = {}
    ) {
        self.op = op
        self.name = name
        self.inputs = [value for value in inputs]
        self.outputs = [value for value in outputs]
        self.attrs = {key: value for (key, value) in attrs.items()}
        self.cached = {}
        self.cached_of = {}
    }

    #: Drop anything an operator memoised on this node. Only `cached` is
    #: freed; `cached_of` refers to tensors the graph owns.
    pub fn free_cached(self) -> void {
        for (_key, tensor) in self.cached.items() { tensor.free() }
        self.cached = {}
        self.cached_of = {}
    }

    #: How this node is named in an error. Nodes are often anonymous in an
    #: exported file, so the first output stands in -- it is unique and it is
    #: what the rest of the graph calls this node's result.
    pub fn label(self) -> str {
        if self.name.len() > 0 { return f"operator {self.op} at node {self.name}" }
        if self.outputs.len() > 0 { return f"operator {self.op} at node {self.outputs[0]}" }
        return f"operator {self.op} at an unnamed node"
    }

    fn __str__(self) -> str { return self.label() }
}

#: A whole network: the nodes, the names fed in, the names read out, and the
#: weights that came with the file.
#: Numbers the graphs, so that a compiled form can be kept for one without
#: holding a reference that would keep it alive.
var _next_graph_id = 0

pub class Graph {
    #: This graph's own number, stable for its lifetime.
    pub let id: int
    pub var nodes: list[GraphNode]
    pub var inputs: list[str]
    pub var outputs: list[str]
    pub var initializers: dict[str, Tensor]
    var plan: list[GraphNode]?

    pub fn init(
        self,
        nodes: list[GraphNode] = [],
        inputs: list[str] = [],
        outputs: list[str] = [],
        initializers: dict[str, Tensor] = {}
    ) {
        _next_graph_id += 1
        self.id = _next_graph_id
        self.nodes = [node for node in nodes]
        self.inputs = [name for name in inputs]
        self.outputs = [name for name in outputs]
        self.initializers = {key: value for (key, value) in initializers.items()}
        self.plan = null
    }

    #: The nodes in an order where every input is produced before it is read.
    #:
    #: Raises `ValueError` naming the node and the name when something is
    #: consumed that nothing produces, and naming every node left over when
    #: what remains is a cycle.
    pub fn topological(self) -> list[GraphNode] {
        var producer: dict[str, int] = {}
        for (index, node) in self.nodes.enumerate() {
            for name in node.outputs {
                if name.len() == 0 { continue }
                let earlier = producer.get(name, null)
                if earlier is not null {
                    throw ValueError(
                        f"{node.label()} and {self.nodes[earlier].label()} both produce {name}"
                    )
                }
                producer[name] = index
            }
        }

        var ready: dict[str, bool] = {}
        for name in self.inputs { ready[name] = true }
        for (name, _weight) in self.initializers.items() { ready[name] = true }

        #: How many inputs each node is still waiting for, and which nodes are
        #: waiting on each name. Indexing the second direction once is what
        #: keeps this linear in the edges rather than quadratic in the nodes.
        var pending: list[int] = []
        var waiting_on: dict[str, list[int]] = {}
        for (index, node) in self.nodes.enumerate() {
            var count = 0
            for name in node.inputs {
                if name.len() == 0 or ready.get(name, false) { continue }
                if producer.get(name, null) is null {
                    throw ValueError(
                        f"{node.label()} reads {name}, which nothing in the graph produces"
                    )
                }
                count += 1
                if waiting_on.get(name, null) is null { waiting_on[name] = [] }
                waiting_on[name].push(index)
            }
            pending.push(count)
        }

        var frontier: list[int] = []
        for (index, count) in pending.enumerate() {
            if count == 0 { frontier.push(index) }
        }
        var order: list[GraphNode] = []
        var emitted = [false for _node in self.nodes]
        var cursor = 0
        while cursor < frontier.len() {
            let index = frontier[cursor]
            cursor += 1
            emitted[index] = true
            let node = self.nodes[index]
            order.push(node)
            for name in node.outputs {
                if name.len() == 0 { continue }
                let blocked = waiting_on.get(name, null)
                if blocked is null { continue }
                for other in blocked {
                    pending[other] -= 1
                    if pending[other] == 0 { frontier.push(other) }
                }
            }
        }

        if order.len() != self.nodes.len() {
            var stuck: list[str] = []
            for (index, node) in self.nodes.enumerate() {
                if not emitted[index] { stuck.push(node.label()) }
            }
            throw ValueError(f"these nodes form a cycle: {'; '.join(stuck)}")
        }
        return order
    }

    #: The order `run_graph` actually walks: topologically sorted, then
    #: rewritten by `fuse_activations`.
    #:
    #: Held onto because neither step depends on the values flowing through --
    #: only on the shape of the graph -- and redoing them per frame cost more
    #: than some of the operators do. A caller who edits `nodes`, `inputs`,
    #: `outputs` or `initializers` afterwards has to call `invalidate`.
    pub fn schedule(self) -> list[GraphNode] {
        let ready = self.plan
        if ready is not null { return ready }
        let built = fuse_activations(self.topological(), self.outputs)
        self.plan = built
        return built
    }

    #: Forget the schedule and everything the operators memoised, so the next
    #: run rebuilds both. Needed after the graph is edited, and harmless
    #: otherwise.
    pub fn invalidate(self) -> void {
        self.plan = null
        for node in self.nodes { node.free_cached() }
    }

    fn __str__(self) -> str {
        return f"Graph({self.nodes.len()} nodes, in={self.inputs}, out={self.outputs})"
    }
}

#: Rewrite `Mul(x, Sigmoid(x))` to a single `Silu` node.
#:
#: Exporters have no choice but to spell SiLU that way -- ONNX has no operator
#: for it -- so a detector that uses it as its activation pays for two full
#: passes over every activation tensor and an intermediate the size of one.
#: On YOLOv8n that pair was 35% of the graph's warm time.
#:
#: The rewrite only fires when the Sigmoid's result goes nowhere else: another
#: reader, or the graph itself naming it as an output, means the value is
#: wanted on its own and folding it away would lose it. The Mul's operands may
#: be either way round, since multiplication does not care and exporters
#: disagree.
fn fuse_activations(order: list[GraphNode], outputs: list[str]) -> list[GraphNode] {
    let readers = count_readers(order, outputs)

    #: Which Sigmoid produced each name, so the Mul that follows can find it
    #: without a second sweep.
    var sigmoid_of: dict[str, GraphNode] = {}
    var folded: dict[str, bool] = {}
    var fused: list[GraphNode] = []
    for node in order {
        if node.op == "Sigmoid" and node.inputs.len() == 1 and node.outputs.len() == 1 {
            let produced = node.outputs[0]
            if produced.len() > 0 and readers.get(produced, 0) == 1 {
                sigmoid_of[produced] = node
            }
            fused.push(node)
            continue
        }
        if node.op == "Mul" and node.inputs.len() == 2 and node.outputs.len() == 1 {
            let left = node.inputs[0]
            let right = node.inputs[1]
            var source = sigmoid_of.get(right, null)
            var other = left
            if source is null {
                source = sigmoid_of.get(left, null)
                other = right
            }
            if source is not null and source.inputs[0] == other {
                folded[source.outputs[0]] = true
                fused.push(
                    GraphNode(
                        op: "Silu",
                        name: node.name,
                        inputs: [other],
                        outputs: [node.outputs[0]]
                    )
                )
                continue
            }
        }
        fused.push(node)
    }
    var kept: list[GraphNode] = []
    for node in fused {
        if node.op == "Sigmoid" and folded.get(node.outputs[0], false) { continue }
        kept.push(node)
    }
    return kept
}

#: How many places each name is consumed, counting the graph's own outputs as
#: a reader -- a value the caller asked for is wanted whatever else reads it.
fn count_readers(order: list[GraphNode], outputs: list[str]) -> dict[str, int] {
    var readers: dict[str, int] = {}
    for node in order {
        for name in node.inputs {
            if name.len() > 0 { readers[name] = readers.get(name, 0) + 1 }
        }
    }
    for name in outputs { readers[name] = readers.get(name, 0) + 1 }
    return readers
}

pub fn attr_int(node: GraphNode, name: str, fallback: int) -> int {
    let found = node.attrs.get(name, null)
    if found is null { return fallback }
    if found.kind != "int" {
        throw ValueError(f"{node.label()} carries {name} as {found.kind}, expected int")
    }
    return found.i
}

pub fn attr_float(node: GraphNode, name: str, fallback: float) -> float {
    let found = node.attrs.get(name, null)
    if found is null { return fallback }
    if found.kind == "int" { return float(found.i) }
    if found.kind != "float" {
        throw ValueError(f"{node.label()} carries {name} as {found.kind}, expected float")
    }
    return found.f
}

pub fn attr_str(node: GraphNode, name: str, fallback: str) -> str {
    let found = node.attrs.get(name, null)
    if found is null { return fallback }
    if found.kind != "str" {
        throw ValueError(f"{node.label()} carries {name} as {found.kind}, expected str")
    }
    return found.s
}

pub fn attr_ints(node: GraphNode, name: str, fallback: list[int]) -> list[int] {
    let found = node.attrs.get(name, null)
    if found is null { return fallback }
    if found.kind != "ints" {
        throw ValueError(f"{node.label()} carries {name} as {found.kind}, expected ints")
    }
    return found.ints
}

pub fn attr_floats(node: GraphNode, name: str, fallback: list[float]) -> list[float] {
    let found = node.attrs.get(name, null)
    if found is null { return fallback }
    if found.kind != "floats" {
        throw ValueError(f"{node.label()} carries {name} as {found.kind}, expected floats")
    }
    return found.floats
}

pub fn maybe(ins: list[Tensor?], index: int) -> Tensor? {
    if index < 0 or index >= ins.len() { return null }
    return ins[index]
}

pub fn need(ins: list[Tensor?], index: int, node: GraphNode) -> Tensor {
    let value = maybe(ins, index)
    if value is null { throw ValueError(f"{node.label()} needs an input at position {index}") }
    return value
}

#: Integers from an input if the graph passes them as one, otherwise from the
#: attribute an older opset put them in.
pub fn ints_from(node: GraphNode, ins: list[Tensor?], index: int, name: str) -> list[int]? {
    let supplied = maybe(ins, index)
    if supplied is not null { return host_ints(supplied) }
    let found = node.attrs.get(name, null)
    if found is null { return null }
    if found.kind != "ints" {
        throw ValueError(f"{node.label()} carries {name} as {found.kind}, expected ints")
    }
    return found.ints
}
