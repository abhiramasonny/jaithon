#: A bool returned by a function in ANOTHER module, read from a compiled body.
#:
#: Same mistake as `test_jit_bool_field`, one level out. `BOOL_VAL` writes only
#: the one-byte member of the Value union and leaves the other seven
#: indeterminate, so a returned bool has to come out of the call descriptor
#: with `ldrb`. Read eight bytes wide it carries whatever the result slot held
#: before, and every test downstream is against zero, so it reads true.
#:
#: Only a CROSS-MODULE callee reaches it: a callee in the caller's own module
#: goes through the direct-branch path, which takes the callee's clean x0. A
#: same-module copy of the identical function is therefore right while the
#: imported one is wrong, which is what made this hard to see -- and why both
#: are checked here.
#:
#: It cost the whole of `jaiframe`: `Column.from_floats` decides which rows are
#: missing with `values.map(|v| is_nan(v))`, that came back all-true over a
#: list holding no NaN, so every column arrived fully null and 72 tests across
#: seven modules failed on answers that were `nan` or zero.
from std.math import NAN, is_nan
from std.test import assert_eq, assert_true

fn local_is_nan(x: float) -> bool { return x != x }

#: A named function whose whole body is the imported call, which is the shape
#: a lambda passed to `map` compiles to.
fn imported_flag(x: float) -> bool { return is_nan(x) }

fn warm_imported(values: list[float]) -> int {
    var hits = 0
    for value in values {
        if is_nan(value) { hits += 1 }
    }
    return hits
}

fn imported_flags(values: list[float]) -> list[bool] { return values.map(|v| is_nan(v)) }

fn local_flags(values: list[float]) -> list[bool] { return values.map(|v| local_is_nan(v)) }

#: The callee has to be compiled BEFORE the caller that returns its bool, or
#: the caller takes the descriptor path's untyped arm and the load never
#: happens. Warming it here is what puts the two in that order.
fn test_a_bool_returned_across_a_module_survives_the_tier() -> void {
    var data: list[float] = []
    for index in 0..2000 { data.push(float(index)) }
    for _round in 0..200 { warm_imported(data) }

    var wrong = 0
    for _round in 0..400 {
        let flags = imported_flags([1.0, 2.0])
        if flags[0] or flags[1] { wrong += 1 }
    }
    assert_eq(wrong, 0, "an imported `is_nan` came back true through `map`")

    var local_wrong = 0
    for _round in 0..400 {
        let flags = local_flags([1.0, 2.0])
        if flags[0] or flags[1] { local_wrong += 1 }
    }
    assert_eq(local_wrong, 0, "a same-module `is_nan` came back true through `map`")
}

#: The same return, used by the compiled caller rather than handed to a
#: builtin, so the defect is caught whether or not `map` is in the picture.
fn test_a_bool_returned_across_a_module_is_used_correctly() -> void {
    var data: list[float] = []
    for index in 0..2000 { data.push(float(index)) }
    for _round in 0..200 { warm_imported(data) }

    var hits = 0
    for _round in 0..400 {
        for value in [1.0, 2.0, 3.0] {
            if imported_flag(value) { hits += 1 }
        }
    }
    assert_eq(hits, 0, "an imported `is_nan` read true inside a compiled caller")

    var found = 0
    for _round in 0..400 {
        for value in [1.0, NAN, 3.0] {
            if imported_flag(value) { found += 1 }
        }
    }
    assert_eq(found, 400, "a real NaN stopped being seen")
    assert_true(imported_flag(NAN), "the cold path disagrees with the hot one")
}
