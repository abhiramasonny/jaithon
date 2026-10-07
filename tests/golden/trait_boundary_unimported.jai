#: A value whose static type is joined from two branches, handed to another
#: module's trait-typed parameter, from a module that never imported the trait.
#:
#: The checker accepts the call and the emitter puts a boundary guard in front
#: of it, naming the parameter's declared type. That guard runs in THIS module,
#: which has no binding for the name, and the runtime used to fall back to
#: comparing the value's own type name with it: `Plain` is not spelled
#: `Stepper`, so `Sequential.compile(optimizer: if ... { Adam() } else { SGD() })`
#: threw "expected 'Optimizer' but got 'SGD'" though `jaithon check` passed.
from .trait_boundary.optim import Moment, Nesterov, Plain, Trainer, pick

fn wide(flag: bool) -> bool { return flag }

fn mixed_by_keyword(flag: bool) -> str {
    let trainer = Trainer()
    let stepper = if wide(flag) { Moment(0.001) } else { Plain(0.08) }
    return trainer.compile(stepper: stepper, label: "k")
}

fn mixed_positionally(flag: bool) -> str {
    let trainer = Trainer()
    let stepper = if wide(flag) { Moment(0.001) } else { Plain(0.08) }
    return trainer.compile(stepper)
}

fn both_the_same_class(flag: bool) -> str {
    let trainer = Trainer()
    let stepper = if wide(flag) { Plain(0.001) } else { Plain(0.08) }
    return f"{trainer.compile(stepper: stepper)} {trainer.compile(stepper)}"
}

fn a_subclass(flag: bool) -> str {
    let stepper = if wide(flag) { Nesterov(0.01) } else { Moment(0.08) }
    return f"{Trainer().compile(stepper)} {pick(stepper)}"
}

#: The fallback matches by name, so it has to keep refusing what does not
#: implement the trait at all. `any` is how a wrong value reaches the guard
#: past the checker.
fn refused(value: any) -> str {
    try {
        return Trainer().compile(value)
    } catch failure: TypeError {
        return "TypeError"
    }
}

fn main() -> int {
    for flag in [true, false] {
        print(mixed_by_keyword(flag))
        print(mixed_positionally(flag))
        print(both_the_same_class(flag))
        print(a_subclass(flag))
    }
    print(refused("not a stepper"))
    print(refused(Trainer()))
    print(refused(Plain(1.0)))

    #: Enough calls that a JIT tier compiles the callers, so the compiled
    #: guard is held to the same answer.
    var count = 0
    for index in 0..4000 {
        if index % 3 == 0 {
            if mixed_positionally(true) == "moment" { count += 1 }
        } else {
            if mixed_by_keyword(false) == "plain" { count += 1 }
        }
    }
    print(count)
    return 0
}
