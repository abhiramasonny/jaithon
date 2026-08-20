#: A function with default parameters, compiled.
#:
#: Defaults used to make a function categorically ineligible for the function
#: tier, which ruled out most library code. Entry is safe because every path
#: into compiled code already requires the arguments in full -- so what these
#: check is the other half: that a call which OMITS an argument still gets the
#: default, and still gets the right one after the body has compiled.
from std.test import assert_eq, assert_true

fn scaled(value: int, factor: int = 3, offset: int = 1) -> int {
    return value * factor + offset
}

fn tagged(value: int, label: str = "d", flag: bool = true) -> int {
    if flag { return value + label.len() }
    return value
}

fn test_a_full_call_and_a_defaulted_one_agree_once_hot() -> void {
    #: Warm on the full-argument form, which is the one that compiles.
    var full = 0
    for index in 0..40000 { full += scaled(index % 97, 3, 1) }

    #: Then check the defaulted forms still produce the defaults.
    assert_eq(scaled(10), 31, "both defaults")
    assert_eq(scaled(10, 5), 51, "one default")
    assert_eq(scaled(10, 5, 2), 52, "no defaults")

    var cold = 0
    for index in 0..97 { cold += scaled(index, 3, 1) }
    var hot = 0
    for index in 0..97 { hot += scaled(index) }
    assert_eq(hot, cold, "omitting the defaults changed the answer")
}

fn test_interleaving_full_and_defaulted_calls() -> void {
    #: The two go through different entries -- full arguments can reach the
    #: compiled body, a short call cannot -- so alternating is what would
    #: catch the compiled one reading an unfilled slot.
    var total = 0
    for index in 0..40000 {
        total += scaled(index % 50, 2, 7)
        total += scaled(index % 50)
    }
    var expected = 0
    for index in 0..40000 {
        expected += index % 50 * 2 + 7
        expected += index % 50 * 3 + 1
    }
    assert_eq(total, expected, "a defaulted call read the wrong slot")
}

fn test_defaults_of_other_kinds() -> void {
    var total = 0
    for index in 0..40000 { total += tagged(index % 20, "xy", true) }
    assert_eq(tagged(5), 6, "a str default")
    assert_eq(tagged(5, "abc"), 8, "a supplied str")
    assert_eq(tagged(5, "abc", false), 5, "a bool default overridden")
    assert_true(total > 0, "the warm loop did nothing")
}
