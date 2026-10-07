#: A trait, two classes that implement it, and a method whose parameter is
#: typed by the trait. trait_boundary_unimported.jai imports the classes
#: and the method's class but never the trait, which is the shape that made a
#: boundary guard name a type its caller could not resolve.
pub trait Stepper {
    fn name(self) -> str
}

pub class Plain: Stepper {
    pub let rate: float

    pub fn init(self, rate: float) { self.rate = rate }

    pub fn name(self) -> str { return "plain" }
}

pub class Moment: Stepper {
    pub let rate: float

    pub fn init(self, rate: float) { self.rate = rate }

    pub fn name(self) -> str { return "moment" }
}

#: A subclass, so the walk up the superclass chain is exercised too.
pub class Nesterov extends Moment {
    pub fn init(self, rate: float) { super.init(rate) }

    pub fn name(self) -> str { return "nesterov" }
}

pub class Trainer {
    pub var chosen: str

    pub fn init(self) { self.chosen = "" }

    pub fn compile(self, stepper: Stepper, label: str = "") -> str {
        self.chosen = stepper.name()
        return self.chosen
    }
}

pub fn pick(stepper: Stepper) -> str { return stepper.name() }
