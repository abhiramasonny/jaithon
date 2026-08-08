#: Fixture for the `--dump-sema` golden test.
#:
#: One of every decision the dump records. `folded` is a constant fold;
#: `take` returns an `any` where `int` is declared, which is a guard; `widened`
#: is an `int` binding stored where `float` is declared, which is a widening.
#: The widening needs a binding and not a literal: an int *literal* in a float
#: context becomes a float literal outright and no cast is inserted, so a
#: fixture written with `let widened: float = 1` exercises nothing.
let counted = 2
let widened: float = counted
let folded = 2 + 3

fn take(value: any) -> int {
    return value
}

fn main() -> void {
    print(widened)
    print(folded)
    print(take(1))
}
