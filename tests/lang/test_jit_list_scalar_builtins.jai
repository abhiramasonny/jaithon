#: `sum`, `min` and `max` over a LIST, in a loop hot enough to compile.
#:
#: Every other row in `kNativeResults` states a result kind that is a property
#: of the FUNCTION. These three are a property of the ARGUMENT: an int list sums
#: to an int and a float list to a float, so the table cannot hold them and the
#: kind is read off the list's own element instead -- from a live sample if the
#: model has one, otherwise from what `OP_BUILD_LIST` recorded.
#:
#: Worth an arm because they are CHEAP next to the loop around them, which is
#: the test a builtin row has to pass. `sorted` is not, and its rows were built
#: and thrown away for measuring zero
#: (docs/research/FALSIFIED-list-returning-builtins.md). A five-element `sum` is
#: five adds: `docs/probes/p20_sum_min_max.jai` ran 17,000,327 interpreted
#: instructions -- the whole loop -- and 26,522 after, 1.12x on the clock
#: (0.28s to 0.25s, five runs each, no spread, measured in one binary through
#: `JAITHON_JIT_LIST_SCALAR`).
# jit-compiles: total_ints, total_floats, spread_of
from std.test import assert_eq, assert_true

let CALLS = 100

fn total_ints(xs: list[int], rounds: int) -> int {
    var t = 0
    var i = 0
    while i < rounds {
        t += sum(xs)
        i += 1
    }
    return t
}

fn total_floats(xs: list[float], rounds: int) -> float {
    var t = 0.0
    var i = 0
    while i < rounds {
        t += sum(xs)
        i += 1
    }
    return t
}

fn spread_of(xs: list[int], rounds: int) -> int {
    var t = 0
    var i = 0
    while i < rounds {
        t += max(xs) - min(xs)
        i += 1
    }
    return t
}

pub fn test_an_int_list_sums_to_an_int() -> void {
    let xs = [5, 3, 9, 1, 7]
    var total = 0
    var call = 0
    while call < CALLS {
        total += total_ints(xs, 10)
        call += 1
    }
    assert_eq(total, 25 * 10 * CALLS)
}

pub fn test_a_float_list_sums_to_a_float() -> void {
    let xs = [1.5, 2.5, 3.0]
    var total = 0.0
    var call = 0
    while call < CALLS {
        total += total_floats(xs, 10)
        call += 1
    }
    assert_true(total > 6999.0 and total < 7001.0, "7.0 x 10 x CALLS")
}

pub fn test_min_and_max_keep_the_element_kind() -> void {
    let xs = [5, 3, 9, 1, 7]
    var total = 0
    var call = 0
    while call < CALLS {
        total += spread_of(xs, 10)
        call += 1
    }
    assert_eq(total, 8 * 10 * CALLS)
}

pub fn test_a_mixed_list_is_refused_not_guessed() -> void {
    #: No exemplar the arm will take, so the body is declined and the
    #: interpreter answers. Here to prove the refusal is a refusal.
    let xs: list[any] = [1, 2.5, 3]
    var total = 0.0
    var call = 0
    while call < CALLS {
        total += sum(xs)
        call += 1
    }
    assert_true(total > 649.0 and total < 651.0, "6.5 x CALLS")
}
