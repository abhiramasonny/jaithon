#: Optimizers used by `Sequential.compile`.

#: Configuration shared by GPU parameter-update kernels.
pub trait Optimizer {
    #: Backend update algorithm identifier.
    fn kind(self) -> int

    #: Numeric settings copied to one small device buffer.
    fn settings(self) -> list[float]

    #: Short name used in summaries.
    fn name(self) -> str
}

#: Minibatch stochastic gradient descent.
pub class SGD: Optimizer {
    #: Step size applied to each minibatch gradient.
    pub let learning_rate: float
    #: Fraction of the previous update retained as velocity.
    pub let momentum: float
    #: Whether to evaluate the update at the momentum look-ahead point.
    pub let nesterov: bool

    #: Configure SGD, optionally with momentum and Nesterov acceleration.
    pub fn init(self, learning_rate: float = 0.01, momentum: float = 0.0, nesterov: bool = false) {
        if learning_rate <= 0.0 {
            throw ValueError(f"learning_rate must be positive, got {learning_rate}")
        }
        if momentum < 0.0 or momentum >= 1.0 {
            throw ValueError(f"momentum must be in [0, 1), got {momentum}")
        }
        if nesterov and momentum == 0.0 {
            throw ValueError("Nesterov SGD needs non-zero momentum")
        }
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
    }

    #: Select the SGD update in the shared Metal kernel.
    pub fn kind(self) -> int { return 0 }

    #: Encode learning rate, momentum, and Nesterov mode for the GPU.
    pub fn settings(self) -> list[float] {
        return [self.learning_rate, self.momentum, if self.nesterov { 1.0 } else { 0.0 }]
    }

    #: Optimizer name used in model output.
    pub fn name(self) -> str { return "SGD" }
}

#: Adam with first- and second-moment bias correction.
pub class Adam: Optimizer {
    #: Base step size.
    pub let learning_rate: float
    #: Exponential decay for the first moment.
    pub let beta1: float
    #: Exponential decay for the second moment.
    pub let beta2: float
    #: Denominator stabilizer.
    pub let epsilon: float

    #: Configure Adam with common defaults.
    pub fn init(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-7
    ) {
        if learning_rate <= 0.0 {
            throw ValueError(f"learning_rate must be positive, got {learning_rate}")
        }
        if beta1 < 0.0 or beta1 >= 1.0 { throw ValueError(f"beta1 must be in [0, 1), got {beta1}") }
        if beta2 < 0.0 or beta2 >= 1.0 { throw ValueError(f"beta2 must be in [0, 1), got {beta2}") }
        if epsilon <= 0.0 { throw ValueError(f"epsilon must be positive, got {epsilon}") }
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
    }

    #: Select the Adam update in the shared Metal kernel.
    pub fn kind(self) -> int { return 1 }

    #: Encode learning rate, decay rates, and epsilon for the GPU.
    pub fn settings(self) -> list[float] {
        return [self.learning_rate, self.beta1, self.beta2, self.epsilon]
    }

    #: Optimizer name used in model output.
    pub fn name(self) -> str { return "Adam" }
}
