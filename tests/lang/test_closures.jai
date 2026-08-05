#: Closure capture semantics (spec §6): `var` by reference, `let` by value.
from std.test import assert_eq

fn test_var_is_captured_by_reference() -> void {
    var count = 0
    let bump = fn() { count += 1 }
    bump()
    bump()
    assert_eq(count, 2)
}

fn test_let_is_captured_by_value() -> void {
    let base = 10
    let add = |x| x + base
    assert_eq(add(5), 15)
}

fn test_each_closure_gets_its_own_environment() -> void {
    fn counter() -> fn() -> int {
        var n = 0
        return fn() -> int {
            n += 1
            return n
        }
    }
    let a = counter()
    let b = counter()
    assert_eq(a(), 1)
    assert_eq(a(), 2)
    assert_eq(b(), 1)
}

fn test_loop_variables_are_fresh_per_iteration() -> void {
    # The classic trap: in a language that shares one loop variable, every
    # closure here would see the final value.
    var fns = []
    for i in 0..3 { fns.push(|| i) }
    assert_eq(fns.map(|f| f()), [0, 1, 2])
}

fn test_upvalue_survives_the_frame() -> void {
    fn make_pair() -> list[any] {
        var shared = 0
        let get = || shared
        let set = fn(v) { shared = v }
        return [get, set]
    }
    let pair = make_pair()
    let get = pair[0]
    let set = pair[1]
    set(42)
    assert_eq(get(), 42)
}

fn test_deep_capture_through_two_levels() -> void {
    fn outer() -> fn() -> int {
        var x = 1
        fn middle() -> fn() -> int {
            fn inner() -> int {
                x += 1
                return x
            }
            return inner
        }
        return middle()
    }
    let f = outer()
    assert_eq(f(), 2)
    assert_eq(f(), 3)
}

fn test_recursive_closure_via_named_function() -> void {
    fn fib(n: int) -> int {
        if n < 2 { return n }
        return fib(n - 1) + fib(n - 2)
    }
    assert_eq(fib(20), 6765)
}

fn test_defaults_are_evaluated_per_call() -> void {
    fn append(x: int, into: list[int] = []) -> list[int] {
        into.push(x)
        return into
    }
    assert_eq(append(1), [1])
    assert_eq(append(2), [2])
}
