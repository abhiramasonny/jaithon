#: `for k in d` in compiled code: OP_GET_ITER builds the ITER_DICT_KEYS
#: iterator and OP_FOR_ITER_BIND steps it inline, binding the key alone.
#: Before, a function holding one declined whole.
#:
#: Each helper is called more than JAI_JIT_THRESHOLD times so the
#: whole-function tier compiles it on call count alone; `JAITHON_NO_JIT=1`
#: runs the same file, and every answer is the interpreter's.
# jit-compiles: keys_sum, keys_len, keys_int, keys_mutate
from std.test import assert_eq, assert_throws

let CALLS = 100

fn keys_sum(d: dict[str, int]) -> int {
    var t = 0
    for k in d {
        t += d[k]
    }
    return t
}

fn keys_len(d: dict[str, int]) -> int {
    var t = 0
    for k in d {
        t += k.len()
    }
    return t
}

fn keys_int(d: dict[int, str]) -> int {
    var t = 0
    for k in d {
        t += k * 10 + d[k].len()
    }
    return t
}

fn keys_mutate(d: dict[str, int], grow: bool) -> int {
    var n = 0
    for k in d {
        if grow { d[f"{k}+"] = 0 }
        n += 1
    }
    return n
}

pub fn test_string_keys() -> void {
    let d = {"a": 1, "bb": 2, "ccc": 3}
    var t = 0
    var l = 0
    var call = 0
    while call < CALLS {
        t += keys_sum(d)
        l += keys_len(d)
        call += 1
    }
    assert_eq(t, 6 * CALLS)
    assert_eq(l, 6 * CALLS)
}

pub fn test_holes_left_by_deletes() -> void {
    var d = {"a": 1, "b": 2, "c": 3, "d": 4}
    let _ = d.pop("b")
    let _ = d.pop("d")
    var call = 0
    var t = 0
    while call < CALLS {
        t += keys_sum(d)
        call += 1
    }
    assert_eq(t, 4 * CALLS)
}

pub fn test_int_keys() -> void {
    let d = {1: "x", 20: "yy"}
    var t = 0
    var call = 0
    while call < CALLS {
        t += keys_int(d)
        call += 1
    }
    assert_eq(t, (10 + 1 + 200 + 2) * CALLS)
}

pub fn test_a_dict_that_grows_under_the_loop_still_raises() -> void {
    var call = 0
    while call < CALLS {
        let _ = keys_mutate({"a": 1, "b": 2}, false)
        call += 1
    }
    assert_throws(RuntimeError, fn() -> void {
        let _ = keys_mutate({"a": 1, "b": 2}, true)
    }, "a dict grown inside `for k in d`")
}

pub fn test_an_empty_dict() -> void {
    var call = 0
    while call < CALLS {
        assert_eq(keys_sum({}), 0)
        call += 1
    }
}
