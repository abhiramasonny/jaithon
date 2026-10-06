#: Destructuring takes two paths in the interpreter: a target list without a
#: rest copies the items straight over the stack, and one with a rest builds
#: the rest list out of line. Both must agree on order, storage kinds and the
#: errors a wrong shape raises.
from std.test import assert_eq

fn _dyn(value: any) -> any {
    return value
}

fn _pair(i: int) -> (int, int) {
    return (i, i * 2)
}

fn _attempt(value: any) -> str {
    try {
        let (a, b) = _dyn(value)
        return f"pair {a} {b}"
    } catch e: ValueError {
        return f"ValueError {e}"
    } catch e: TypeError {
        return f"TypeError {e}"
    }
}

fn _attempt_rest(value: any) -> str {
    try {
        let [a, ...middle, z] = _dyn(value)
        return f"rest {a} {middle} {z}"
    } catch e: ValueError {
        return f"ValueError {e}"
    } catch e: TypeError {
        return f"TypeError {e}"
    }
}

fn test_without_a_rest_targets_bind_in_source_order() -> void {
    var total = 0
    for i in 0..50 {
        let (a, b) = _pair(i)
        let [c, d, e] = [a, b, a + b]
        let ((x, y), z) = ((a, c), e)
        total += a * 1 + b * 3 + c * 5 + d * 7 + e * 11 + x * 13 + y * 17 + z * 19
    }
    assert_eq(total, 178850)
}

fn test_unboxed_list_storage_unpacks() -> void {
    let floats = [1.5, 2.5, 4.0]
    let [p, q, r] = floats
    assert_eq(p + q * 10.0 + r * 100.0, 426.5)
    let ints = [3, 4]
    let [m, n] = ints
    assert_eq(m * 10 + n, 34)
}

fn test_dict_items_unpack() -> void {
    var seen: list[str] = []
    for (key, value) in {"a": 1, "b": 2}.items() { seen.push(f"{key}={value}") }
    assert_eq(seen, ["a=1", "b=2"])
}

fn test_with_a_rest_the_middle_is_a_fresh_list() -> void {
    assert_eq(_attempt_rest([1, 2, 3, 4]), "rest 1 [2, 3] 4")
    assert_eq(_attempt_rest((7, 8)), "rest 7 [] 8")
    assert_eq(_attempt_rest([1.5, 2.5, 3.5]), "rest 1.5 [2.5] 3.5")
}

fn test_a_wrong_shape_raises() -> void {
    assert_eq(_attempt((1, 2)), "pair 1 2")
    assert_eq(_attempt(["s", 3]), "pair s 3")
    assert_eq(_attempt((1, 2, 3)), "ValueError cannot unpack 3 values into 2 targets")
    assert_eq(_attempt([1]), "ValueError cannot unpack 1 value into 2 targets")
    assert_eq(_attempt(5), "TypeError cannot destructure a 'int' value")
    assert_eq(_attempt_rest([1]), "ValueError cannot unpack 1 value into 3 targets")
    assert_eq(_attempt_rest("ab"), "TypeError cannot destructure a 'str' value")
}
