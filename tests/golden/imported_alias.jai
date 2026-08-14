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

let held: Crate = opaque(3)
print(held.area())

let bigger = scaled(held)
print(bigger.area())
print(scaled(held, 3).area())

fn describe(ink: Ink) -> str {
    return match ink {
        Ink.Solid => "solid",
        Ink.Dotted => "dotted",
    }
}

print(describe(held.ink))
print(describe(Ink.Dotted))
