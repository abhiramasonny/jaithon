#: The state a translation carries: one layer at a time, and the graph so far.
from jaitensor import Tensor
from ..graph import Attr, GraphNode
from .blobs import Blob
from .message import Message

#: One Caffe layer, or a `BatchNorm` and the `Scale` folded into it.
pub class Step {
    pub let layer: Message
    pub let kind: str
    pub let name: str
    pub let bottoms: list[str]
    pub let tops: list[str]
    pub let scale: Message?

    pub fn init(
        self,
        layer: Message,
        kind: str,
        name: str,
        bottoms: list[str],
        tops: list[str],
        scale: Message? = null
    ) {
        self.layer = layer
        self.kind = kind
        self.name = name
        self.bottoms = bottoms
        self.tops = tops
        self.scale = scale
    }

    #: How this layer is named in an error.
    pub fn label(self) -> str {
        if self.name.len() > 0 { return f"the {self.kind} layer {self.name}" }
        return f"an unnamed {self.kind} layer"
    }
}

#: The graph as it is being put together.
pub class Reading {
    pub var nodes: list[GraphNode]
    pub var weights: dict[str, Tensor]
    pub var inputs: list[str]
    pub var live: dict[str, str]
    pub var shapes: dict[str, list[int]]
    pub var taken: dict[str, bool]
    pub var consumed: dict[str, bool]
    pub var written: dict[str, int]
    pub var expected: dict[str, int]
    pub var declared: dict[str, bool]
    pub var finished: list[str]

    pub fn init(self) {
        self.nodes = []
        self.weights = {}
        self.inputs = []
        self.live = {}
        self.shapes = {}
        self.taken = {}
        self.consumed = {}
        self.written = {}
        self.expected = {}
        self.declared = {}
        self.finished = []
    }

    #: Free everything allocated so far. Called when the import throws, so
    #: that a half-built graph does not leak a model's worth of GPU memory.
    pub fn discard(self) -> void {
        for (_name, tensor) in self.weights.items() { tensor.free() }
        self.weights = {}
    }

    #: A name nothing else has taken. Blob names are reserved up front, so the
    #: only thing this bumps is a name this importer invented.
    pub fn fresh(self, wanted: str) -> str {
        if not wanted in self.taken {
            self.taken[wanted] = true
            return wanted
        }
        var counter = 1
        loop {
            let candidate = f"{wanted}:{counter}"
            if not candidate in self.taken {
                self.taken[candidate] = true
                return candidate
            }
            counter += 1
        }
    }

    pub fn hold(self, name: str, shape: list[int], values: list[float]) -> void {
        self.weights[name] = Tensor(shape, values)
    }

    pub fn emit(
        self,
        op: str,
        name: str,
        ins: list[str],
        outs: list[str],
        attrs: dict[str, Attr] = {}
    ) -> void {
        self.nodes.push(GraphNode(op, name: name, inputs: ins, outputs: outs, attrs: attrs))
    }

    #: The graph name currently holding a Caffe blob.
    pub fn read(self, step: Step, blob: str) -> str {
        let name = self.live.get(blob, null)
        if name is null {
            throw ValueError(
                f"{step.label()} reads {blob}, which no earlier layer writes and the net " +
                    "does not declare as an input"
            )
        }
        let held: str = name
        self.consumed[held] = true
        return held
    }

    #: The graph name a Caffe blob's next value goes to. The last value a blob
    #: takes keeps the blob's own name, so an in-place chain leaves Caffe's
    #: name on the thing a caller asks for -- unless the net declared that name
    #: as an input, in which case nothing may produce it and every version has
    #: to be renamed.
    pub fn write(self, blob: str, shape: list[int]?) -> str {
        let seen = self.written.get(blob, 0) + 1
        self.written[blob] = seen
        let last = seen >= self.expected.get(blob, 1)
        let name = if last and not blob in self.declared {
            self.finished.push(blob)
            blob
        } else {
            let invented = self.fresh(f"{blob}:{seen}")
            if last {
                self.finished.push(invented)
            }
            invented
        }
        if shape is not null {
            let dims: list[int] = shape
            self.shapes[name] = dims
        }
        self.live[blob] = name
        return name
    }

    #: The shape of a graph name, when it was possible to work one out.
    pub fn shape_of(self, name: str) -> list[int]? { return self.shapes.get(name, null) }

    #: The shape of a layer's first input, or `null`.
    pub fn input_shape(self, step: Step, index: int = 0) -> list[int]? {
        if index >= step.bottoms.len() { return null }
        let name = self.live.get(step.bottoms[index], null)
        if name is null { return null }
        let held: str = name
        return self.shapes.get(held, null)
    }

    #: A constant this importer invented -- pad widths, a reshape's dimensions,
    #: a coefficient. Always an initializer, never an input.
    pub fn constant(self, stem: str, shape: list[int], values: list[float]) -> str {
        let name = self.fresh(stem)
        self.hold(name, shape, values)
        return name
    }

    #: One of a layer's weight blobs, reshaped to what the operator needs.
    #:
    #: A prototxt on its own carries no weights, and a graph that names them
    #: anyway is more useful than one that refuses to be built: the name goes
    #: into `graph.inputs` instead, so the structure is complete and the caller
    #: supplies the numbers.
    pub fn weight(
        self,
        step: Step,
        blobs: list[Blob],
        index: int,
        stem: str,
        shape: list[int]?
    ) -> str {
        let name = self.fresh(f"{step.name}/{stem}")
        if index >= blobs.len() {
            self.inputs.push(name)
            return name
        }
        let blob = blobs[index]
        var dims = shape ?? blob.shape
        if dims.len() == 0 { dims = [blob.values.len()] }
        var volume = 1
        for dimension in dims { volume *= dimension }
        if volume != blob.values.len() {
            throw ValueError(
                f"{step.label()} has {blob.values.len()} numbers in blob {index}, and this " +
                    f"layer needs {volume} laid out as {dims}"
            )
        }
        self.hold(name, dims, blob.values)
        return name
    }
}

pub fn resolve_axis(axis: int, rank: int, step: Step) -> int {
    let resolved = if axis < 0 { rank + axis } else { axis }
    if resolved < 0 or resolved >= rank {
        throw ValueError(f"{step.label()} names axis {axis}, which is outside rank {rank}")
    }
    return resolved
}

pub fn need_shape(reading: Reading, step: Step, index: int = 0) -> list[int] {
    let found = reading.input_shape(step, index)
    if found is null {
        throw ValueError(
            f"{step.label()} needs the shape of {step.bottoms[index]}, which this file does " +
                "not declare and nothing above it fixes"
        )
    }
    let dims: list[int] = found
    return dims
}

pub fn floats_of_ints(values: list[int]) -> list[float] {
    return [float(value) for value in values]
}
