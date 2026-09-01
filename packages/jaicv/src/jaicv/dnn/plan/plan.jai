#: The compiled graph a caller holds on to, and feeding it.
from jaitensor import Tensor

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
