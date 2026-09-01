#: An EMPTY initializer must hand back the object, not null.
#:
#: `fn init(self) {}` compiles to a chunk that is exactly one `OP_RETURN_NULL`,
#: and the tier has a fast path for precisely that shape: emit `mov w0, #0; ret`
#: and let `jaiJitEnter` write `NULL_VAL` over `slotBase[0]`. That is right for
#: an ordinary function and wrong for an initializer, whose `OP_RETURN_NULL`
#: the interpreter rewrites to `frame->slots[0]` -- the receiver. It is what
#: makes `Point(1, 2)` an expression, and the fast path skipped it, so
#: `Empty()` returned null from the compiled call onward.
#:
#: WHAT THESE TESTS DO AND DO NOT COVER. They pin the contract, and they will
#: not reproduce the original defect on their own -- read this before treating
#: them as the regression test.
#:
#: `compileReturnNull` is a FALLBACK: `jaiJitEnter` only reaches it once the
#: whole-function tier has declined the body first. An empty initializer passes
#: every check that tier makes, so the only decline it can reach is `arenaEmit`
#: returning NULL -- the shared 1 MB code arena being full. That is a property
#: of how much the PROCESS has already compiled, not of the initializer, so it
#: is threshold-independent in principle: any long-running program that fills
#: the arena and then first constructs an empty-init class gets null back.
#:
#: Reaching it deliberately takes about a megabyte of already-compiled code.
#: `JAITHON_JIT_THRESHOLD=1` gets there quickly because every body compiles on
#: its first call; that is how this surfaced, as `packages/jaiplot`'s
#: `LinearScale`. `Axes.init` builds two on consecutive lines, so `x_scale` was
#: an object and `y_scale` was null -- the first construction is always correct,
#: since the call that COMPILES a body does not use it -- and the program died
#: hundreds of frames later in `_fallback` with `'null' object has no method
#: 'accepts'`.
#:
#: A file that does fill the arena and does fail without the fix takes ~1600
#: lines of soak functions plus a chain of 400 empty classes, and it stops
#: reproducing the moment anything upstream changes how much code compiles --
#: it would go quietly green rather than fail, which is worse than not having
#: it. So the arena-full case is named at its refusal instead (see
#: `the code arena is full` in jit_func.c), and what is checked here is the
#: contract every empty initializer has to keep whichever tier compiles it.
from std.test import assert_eq, assert_true

#: The shape that broke: no fields, no statements, no arguments.
class Empty {
    pub fn init(self) {
    }
}

#: The same empty body on a class that also declares fields, so the check does
#: not depend on the instance being featureless.
class EmptyWithFields {
    pub var count: int

    pub fn init(self) {
    }
}

#: An empty initializer taking arguments, whose chunk is still a lone
#: `OP_RETURN_NULL` because it stores none of them.
class EmptyWithArgs {
    pub fn init(self, a: int, b: str) {
    }
}

fn test_an_empty_initializer_returns_the_instance_every_time() -> void {
    var nulls = 0
    var first_null = -1
    for index in 0..400 {
        let made = Empty()
        if made == null {
            nulls += 1
            if first_null < 0 { first_null = index }
        }
    }
    assert_eq(nulls, 0, f"an empty initializer first returned null at construction {first_null}")
}

fn test_an_empty_initializer_on_a_class_with_fields_returns_the_instance() -> void {
    var nulls = 0
    for _index in 0..400 {
        if EmptyWithFields() == null { nulls += 1 }
    }
    assert_eq(nulls, 0, "an empty initializer on a class with fields returned null")
}

fn test_an_empty_initializer_taking_arguments_returns_the_instance() -> void {
    var nulls = 0
    for index in 0..400 {
        if EmptyWithArgs(index, "x") == null { nulls += 1 }
    }
    assert_eq(nulls, 0, "an empty initializer taking arguments returned null")
}

#: Two constructions on consecutive lines, which is how the defect actually
#: presented: the first was an object and the second was null.
fn test_two_consecutive_constructions_both_yield_instances() -> void {
    var wrong = 0
    for _index in 0..400 {
        let a = Empty()
        let b = Empty()
        if a == null or b == null { wrong += 1 }
    }
    assert_eq(wrong, 0, "a second construction on the next line returned null")
}

#: The instance a compiled empty initializer hands back has to be usable, not
#: merely non-null: a wrong-but-non-null value would pass every check above.
fn test_the_instance_from_an_empty_initializer_is_usable() -> void {
    var stored = 0
    for index in 0..400 {
        let made = EmptyWithFields()
        made.count = index
        stored += made.count
    }
    assert_eq(stored, 79800, "a field written through a compiled empty initializer's result")
    assert_true(Empty() is Empty, "the value is an instance of the class that made it")
}
