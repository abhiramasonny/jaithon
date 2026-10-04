#: A declared `str` boundary on a value the compiled tier holds only as "some
#: object": one type check, and a deopt that hands a non-string back to the
#: interpreter, which raises the TypeError a wrong return is owed. Before the
#: arm the guard declined the whole body around it.
# jit-compiles: guard_str_return, guard_str_param
from std.test import assert_eq, assert_throws

let CALLS = 100

fn guard_str_return(x: any) -> str {
    return x
}

fn guard_str_param(names: list[any], i: int) -> int {
    let s: str = names[i]
    return s.len()
}

pub fn test_a_string_passes_the_guard() -> void {
    var total = 0
    var call = 0
    while call < CALLS {
        total += guard_str_return(f"s{call % 10}").len()
        call += 1
    }
    assert_eq(total, 2 * CALLS)
}

pub fn test_anything_else_still_raises() -> void {
    var call = 0
    while call < CALLS {
        let _ = guard_str_return("ok")
        call += 1
    }
    assert_throws(TypeError, fn() -> void {
        let _ = guard_str_return(42)
    }, "an int through a `-> str` boundary")
    assert_throws(TypeError, fn() -> void {
        let _ = guard_str_return([1])
    }, "a list through a `-> str` boundary")
}

pub fn test_a_local_annotation() -> void {
    let names: list[any] = ["ab", "cde"]
    var total = 0
    var call = 0
    while call < CALLS {
        total += guard_str_param(names, call % 2)
        call += 1
    }
    assert_eq(total, 250)
    assert_throws(TypeError, fn() -> void {
        let _ = guard_str_param([7], 0)
    }, "an int into a `str` local")
}
