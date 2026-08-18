#: Shape-specialized execution metadata recorded at `Sequential.compile`.
pub class ExecutionPlan {
    #: Feature width the plan was traced for.
    pub let input_units: int
    #: Batch size the plan was traced for.
    pub let batch_size: int
    #: Output width of each layer, including the classification head.
    pub let layer_widths: list[int]
    #: Whether every layer supports the static forward/backward path.
    pub let static_step: bool

    pub fn init(
        self,
        input_units: int,
        batch_size: int,
        layer_widths: list[int],
        static_step: bool
    ) {
        self.input_units = input_units
        self.batch_size = batch_size
        self.layer_widths = layer_widths
        self.static_step = static_step
    }
}
