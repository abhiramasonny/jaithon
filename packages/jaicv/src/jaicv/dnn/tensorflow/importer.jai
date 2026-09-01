#: What the translation carries: a rank for every name, and the nodes emitted.
from ..graph import Attr, GraphNode
from jaitensor import Tensor
from .graphdef import TfNode
from .layout import NCHW_AXIS, TF_AXIS, permuted, reordered
from .tensors import TfTensor

#: The translation, one TensorFlow node at a time.
#:
#: Nodes are walked in topological order so that the rank of every name is
#: known before the node that needs to renumber its axes is reached. Constants
#: never become nodes: a weight becomes an initializer, and a shape, an axis
#: list or a pad width is read at import and folded into an attribute, which
#: is why nothing downstream has to evaluate a `Const`.
pub class Importer {
    pub var ranks: dict[str, int]
    pub var consts: dict[str, TfTensor]
    pub var dropped: dict[str, bool]
    pub var produced: list[GraphNode]
    pub var weights: dict[str, Tensor]
    pub var graph_inputs: list[str]

    pub fn init(self) {
        self.ranks = {}
        self.consts = {}
        self.dropped = {}
        self.produced = []
        self.weights = {}
        self.graph_inputs = []
    }

    #: Release every initializer built so far. Called when translation throws
    #: part of the way through, so that a rejected file leaves no tensors on
    #: the GPU.
    pub fn discard(self) -> void {
        for (_name, tensor) in self.weights.items() { tensor.free() }
        self.weights = {}
    }

    pub fn emit(self, op: str, name: str, ins: list[str], attrs: dict[str, Attr] = {}) -> str {
        self.produced.push(GraphNode(op, name: name, inputs: ins, outputs: [name], attrs: attrs))
        return name
    }

    #: An initializer under a name derived from the constant and the shape it
    #: was needed in, so that one `Const` feeding a convolution and an addition
    #: becomes two tensors rather than one that is wrong for one of them.
    pub fn intern(self, key: str, shape: list[int], values: list[float]) -> str {
        if key in self.weights { return key }
        self.weights[key] = Tensor(shape, values)
        return key
    }

    pub fn const_of(self, name: str) -> TfTensor? { return self.consts.get(name, null) }

    #: The rank the file gives a name, refusing to guess when it has none.
    pub fn rank_of(self, name: str, node: TfNode) -> int {
        if self.dropped.get(name, false) {
            throw ValueError(f"{node.label()} reads {name}, which carries no value")
        }
        let found = self.ranks.get(name, -2)
        if found == -2 {
            throw ValueError(f"{node.label()} reads {name}, which nothing in the file produces")
        }
        if found < 0 {
            throw ValueError(
                f"{node.label()} reads {name}, whose rank the file does not say -- and this " +
                    "importer needs it to know which axes are which"
            )
        }
        return found
    }

    #: The rank of a name, or -1 when it is not known. For the places that can
    #: carry on without it.
    pub fn loose_rank(self, name: str) -> int { return self.ranks.get(name, -1) }

    #: A constant read as exact integers, for an axis list or a shape.
    pub fn const_ints(self, name: str, node: TfNode, what: str) -> list[int] {
        let held = self.const_of(name)
        if held is null {
            throw ValueError(
                f"{node.label()} computes its {what} rather than holding it as a constant, " +
                    "which this importer does not follow"
            )
        }
        let tensor: TfTensor = held
        return tensor.ints(f"{node.label()}'s {what}")
    }

    #: The name to read a value input by.
    #:
    #: A constant becomes an initializer laid out to the invariant: rank four
    #: goes to `NCHW`, and TensorFlow's rank-zero scalar becomes the `[1]` that
    #: the executor uses in its place.
    pub fn value_input(self, name: str, node: TfNode) -> str {
        let held = self.const_of(name)
        if held is null {
            if self.dropped.get(name, false) {
                throw ValueError(f"{node.label()} reads {name}, which carries no value")
            }
            if self.ranks.get(name, -2) == -2 {
                throw ValueError(f"{node.label()} reads {name}, which nothing in the file produces")
            }
            return name
        }
        let tensor: TfTensor = held
        #: One role, one name: a constant has one rank, so the shape this
        #: hands back is the same every time and interning it once is safe.
        #: A constant read as weights instead goes under a name of its own,
        #: which is what keeps one `Const` feeding a convolution and an
        #: addition from having to be laid out two ways at once.
        if tensor.rank() == 4 {
            return self.intern(
                name + "$data",
                reordered(tensor.shape, TF_AXIS),
                permuted(tensor.values(), tensor.shape, TF_AXIS)
            )
        }
        if tensor.rank() == 0 { return self.intern(name + "$data", [1], tensor.values()) }
        return self.intern(name + "$data", tensor.shape, tensor.values())
    }

    pub fn to_nhwc(self, name: str, tag: str) -> str {
        return self.emit("Transpose", tag, [name], {"perm": Attr.of_ints(NCHW_AXIS)})
    }

    pub fn to_nchw(self, name: str, tag: str) -> str {
        return self.emit("Transpose", tag, [name], {"perm": Attr.of_ints(TF_AXIS)})
    }

    #: An operand of an elementwise node, put where numpy broadcasting will
    #: line it up once every operand agrees on which of the two regimes the
    #: node's result -- rank `want` -- is in.
    #:
    #: When `want` is four the result is `NCHW`, and TensorFlow aligns a
    #: shorter shape against the *trailing* axes, so a bias of `[C]` meets
    #: `[N, H, W, C]` on the channels. Stored `NCHW`, the channels are axis
    #: one, so the bias has to become `[1, C, 1, 1]`. Padding the shape with
    #: leading ones and then permuting it is the same rule for every rank, and
    #: for a constant it is done to the initializer rather than with nodes.
    #:
    #: When `want` is not four the result stays in TensorFlow's own order, and
    #: the executor's own broadcasting already aligns trailing axes exactly
    #: the way TensorFlow does -- so every operand can be handed over as
    #: `value_input` gives it, MINUS the one case `value_input` gets backwards
    #: for this purpose: a rank-four operand, which `value_input` always
    #: permutes into `NCHW` on the assumption it is going somewhere that wants
    #: that. Here it is not, so a rank-four operand has to be put back into
    #: TensorFlow's order first, the same conversion `to_nhwc` does for an
    #: intermediate value, or read straight out of the constant's own
    #: un-permuted shape.
    pub fn broadcast_input(self, name: str, node: TfNode, want: int) -> str {
        let rank = self.rank_of(name, node)
        if want != 4 {
            if rank != 4 { return self.value_input(name, node) }
            let held = self.const_of(name)
            if held is not null {
                let tensor: TfTensor = held
                return self.intern(name + "$flat", tensor.shape, tensor.values())
            }
            let source = self.value_input(name, node)
            return self.to_nhwc(source, f"{node.name}$flat${name}")
        }
        if rank == 4 { return self.value_input(name, node) }
        if rank > 4 {
            throw ValueError(
                f"{node.label()} broadcasts a rank-{rank} operand against a rank-four one, " +
                    "which this importer's layout does not cover"
            )
        }
        #: The executor has no rank-zero tensor, so a scalar is already `[1]`
        #: and needs one fewer axis inserted than its TensorFlow rank suggests.
        let stored = max(rank, 1)
        let held = self.const_of(name)
        if held is not null {
            let tensor: TfTensor = held
            var padded = [1 for _slot in 0..4 - stored]
            for dimension in tensor.shape { padded.push(dimension) }
            if rank == 0 { padded.push(1) }
            return self.intern(
                name + "$bcast",
                reordered(padded, TF_AXIS),
                permuted(tensor.values(), padded, TF_AXIS)
            )
        }
        let source = self.value_input(name, node)
        let axes = [axis for axis in 0..4 - stored]
        let grown = self.emit(
            "Unsqueeze",
            f"{node.name}$grow${name}",
            [source],
            {"axes": Attr.of_ints(axes)}
        )
        return self.to_nchw(grown, f"{node.name}$align${name}")
    }
}
