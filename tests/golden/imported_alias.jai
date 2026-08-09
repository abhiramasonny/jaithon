#: What crossing a module boundary buys the checker (spec §8, Phase 5).
#:
#: Every line here is a decision the checker cannot make without reading the
#: other module's declarations, and each one shows up in the bytecode:
#:
#: - constructing an imported class answers that class, so the guard below is
#:   the one the C emits rather than nothing at all
#: - the guard names the class the way this file's globals do — `Crate`, the
#:   alias — while the type itself is interned as `cross_module.shapes.Box`
#: - an imported enum's variants have tags here, so a pattern compiles to the
#:   tag test and not to the by-name fallback
#: - an imported function's defaults are its own, so a call may omit them
from .cross_module.shapes import Box as Crate, Ink, scaled

fn opaque(side: int) -> any { return Crate(side) }

# `any -> Crate` needs a runtime guard, and the guard's operand is the name
# this module resolves against its own globals.
let held: Crate = opaque(3)
print(held.area())

# `factor` has a default in the module that declares it.
let bigger = scaled(held)
print(bigger.area())
print(scaled(held, 3).area())

# The variant order is the declaration's, which lives in the other file.
fn describe(ink: Ink) -> str {
    return match ink {
        Ink.Solid => "solid",
        Ink.Dotted => "dotted",
    }
}

print(describe(held.ink))
print(describe(Ink.Dotted))
