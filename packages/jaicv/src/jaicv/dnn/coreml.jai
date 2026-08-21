#: Running a model on the neural accelerator, which is the one piece of this
#: machine nothing else here reaches.
#:
#: It is worth being clear about why. The accelerator is NOT faster than the
#: GPU at running a network: a model compiled by `dnn.plan` beat it on every
#: one measured -- YOLOv8n at 3.2 ms against 5.8. What it is, is a separate
#: piece of silicon that runs at the same time. Measured on an M2 Max, 0.45 ms
#: of GPU work and 0.73 ms of accelerator work took 0.80 ms together rather
#: than the 1.18 they would take one after the other.
#:
#: So this is not the fast path for a model on its own. It is the way to run a
#: model WHILE the GPU is doing something else -- a detector on the frame that
#: just arrived while the GPU letterboxes the next one, or an evaluation pass
#: while training continues. Used that way a pipeline costs the slower of its
#: two halves instead of their sum.
#:
#: Only CoreML can reach the accelerator, so the model has to be one of its
#: own: a `.mlmodel`, a `.mlpackage`, or an already-compiled `.mlmodelc`. The
#: first two are compiled on open, which takes a noticeable fraction of a
#: second and is worth doing once.
#:
#: ```
#: let model = CoreMLModel("detector.mlpackage", units: NEURAL_ENGINE)
#: let out = model.run({model.inputs[0]: frame})
#: ```
import jaitensor as jt
from jaitensor import Tensor

#: Where CoreML may place the work. It decides per operator within whatever it
#: is allowed, so these are permissions rather than instructions.
pub let CPU_ONLY = 0
pub let CPU_AND_GPU = 1
pub let EVERYTHING = 2
#: The one that admits the accelerator without also admitting the GPU, which is
#: what makes a model leave the GPU free for other work.
pub let NEURAL_ENGINE = 3

#: A compiled CoreML model, held open.
pub class CoreMLModel {
    let handle: int
    #: What the model calls its inputs and outputs, in a settled order.
    pub let inputs: list[str]
    pub let outputs: list[str]
    pub let input_shapes: list[list[int]]
    pub let output_shapes: list[list[int]]
    var freed: bool

    #: Open and compile a model. Raises `RuntimeError` when there is no CoreML,
    #: when the file is not a model, or when the model will not load for the
    #: units asked for.
    pub fn init(self, path: str, units: int = EVERYTHING) {
        self.handle = __prim__.coreml_open(path, units)
        self.freed = false
        self.inputs = __prim__.coreml_names(self.handle, 0)
        self.outputs = __prim__.coreml_names(self.handle, 1)
        var wanted: list[list[int]] = []
        for index in 0..self.inputs.len() {
            wanted.push(__prim__.coreml_shape(self.handle, 0, index))
        }
        self.input_shapes = wanted
        var produced: list[list[int]] = []
        for index in 0..self.outputs.len() {
            produced.push(__prim__.coreml_shape(self.handle, 1, index))
        }
        self.output_shapes = produced
    }

    #: Run the model, by the names it gave its own inputs.
    #:
    #: The tensors are read where they already are -- device memory is shared
    #: with the host, so nothing is copied in. What comes back is copied out of
    #: CoreML's own arrays, which is small beside the work that made it.
    pub fn run(self, feeds: dict[str, Tensor]) -> dict[str, Tensor] {
        if self.freed { throw RuntimeError("this CoreMLModel has already been closed") }
        var handles: list[int] = []
        var shapes: list[list[int]] = []
        for (index, name) in self.inputs.enumerate() {
            let fed = feeds.get(name, null)
            if fed is null { throw ValueError(f"the model's input {name} was not fed") }
            handles.push(fed.buffer().handle_of())
            #: The model's own declared shape when it has one, since CoreML
            #: checks it; otherwise whatever the tensor is.
            let declared = self.input_shapes[index]
            shapes.push(if declared.len() > 0 { declared } else { fed.shape })
        }

        var produced: dict[str, Tensor] = {}
        var out_handles: list[int] = []
        var out_shapes: list[list[int]] = []
        for (index, name) in self.outputs.enumerate() {
            let shape = self.output_shapes[index]
            if shape.len() == 0 {
                throw RuntimeError(f"the model's output {name} has no fixed shape to receive")
            }
            let tensor = Tensor(shape)
            produced[name] = tensor
            out_handles.push(tensor.buffer().handle_of())
            out_shapes.push(shape)
        }
        __prim__.coreml_run(self.handle, handles, shapes, out_handles, out_shapes)
        return produced
    }

    #: Start the model and come back for it later.
    #:
    #: Between `start` and `wait` the caller can queue as much GPU work as it
    #: likes: the accelerator and the GPU are separate and run at once, so a
    #: pipeline that does both costs the slower of the two rather than their
    #: sum. The tensors handed in must not be written to in between, and the
    #: ones handed back hold nothing until `wait` returns.
    pub fn start(self, feeds: dict[str, Tensor]) -> tuple[int, dict[str, Tensor]] {
        if self.freed { throw RuntimeError("this CoreMLModel has already been closed") }
        var handles: list[int] = []
        var shapes: list[list[int]] = []
        for (index, name) in self.inputs.enumerate() {
            let fed = feeds.get(name, null)
            if fed is null { throw ValueError(f"the model's input {name} was not fed") }
            handles.push(fed.buffer().handle_of())
            let declared = self.input_shapes[index]
            shapes.push(if declared.len() > 0 { declared } else { fed.shape })
        }
        var produced: dict[str, Tensor] = {}
        var out_handles: list[int] = []
        var out_shapes: list[list[int]] = []
        for (index, name) in self.outputs.enumerate() {
            let shape = self.output_shapes[index]
            if shape.len() == 0 {
                throw RuntimeError(f"the model's output {name} has no fixed shape to receive")
            }
            let tensor = Tensor(shape)
            produced[name] = tensor
            out_handles.push(tensor.buffer().handle_of())
            out_shapes.push(shape)
        }
        let ticket = __prim__.coreml_start(self.handle, handles, shapes, out_handles, out_shapes)
        return (ticket, produced)
    }

    #: Wait for a prediction `start` began. The tensors it returned hold their
    #: results once this comes back.
    pub fn wait(self, ticket: int) -> void { __prim__.coreml_wait(ticket) }

    pub fn close(self) -> void {
        if self.freed { return }
        self.freed = true
        __prim__.coreml_close(self.handle)
    }
}

#: Whether this machine has CoreML at all.
pub fn is_available() -> bool {
    try {
        let probe = CoreMLModel("", units: CPU_ONLY)
        probe.close()
        return true
    } catch failure: RuntimeError {
        #: An empty path always fails, but it fails differently with and
        #: without the framework: without it there is nothing to refuse with.
        return not failure.message.contains("no CoreML on this machine")
    }
}
