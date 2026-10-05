#: A body that opens with `if <int param> <cmp> <literal> { return <param or
#: literal> }` is compiled with that test ahead of its frame
#: (JAITHON_JIT_SHRINK_WRAP), and the body proper starts past it.
#:
#: Each comparison, both kinds of returned value, a returned parameter that is
#: not the tested one, recursion through the frameless arm, and the body that
#: follows it all have to agree with the interpreter.
from std.test import assert_eq

fn fib(n: int) -> int {
    if n < 2 { return n }
    return fib(n - 1) + fib(n - 2)
}

fn floor_at_zero(n: int) -> int {
    if n <= 0 { return 0 }
    return n
}

fn above(n: int, cap: int) -> int {
    if n > 100 { return cap }
    return n * 2
}

fn at_least(n: int) -> int {
    if n >= 7 { return -1 }
    return n + 1
}

fn when_three(n: int, other: int) -> int {
    if n == 3 { return other }
    return n
}

fn unless_five(n: int) -> int {
    if n != 5 { return n }
    return 500
}

#: Recursion all the way down through the arm, so most calls never build a
#: frame at all.
fn count_down(n: int) -> int {
    if n <= 0 { return 0 }
    return 1 + count_down(n - 1)
}

#: The body after the arm overflows; the raise must come from there.
fn double_until(n: int, x: int) -> int {
    if n == 0 { return x }
    return double_until(n - 1, x * 2)
}

fn test_fib_through_the_arm() -> void {
    var total = 0
    for n in 0..25 { total += fib(n) }
    assert_eq(total, 121392)
    assert_eq(fib(25), 75025)
}

fn test_each_comparison() -> void {
    var a = 0
    var b = 0
    var c = 0
    var d = 0
    var e = 0
    for i in -50..150 {
        a += floor_at_zero(i)
        b += above(i, 7)
        c += at_least(i % 10)
        d += when_three(i % 6, 1000)
        e += unless_five(i % 8)
    }
    assert_eq(a, 11175)
    assert_eq(b, 7893)
    assert_eq(c, 500)
    assert_eq(d, 33405)
    assert_eq(e, 13075)
}

fn test_deep_recursion_through_the_arm() -> void {
    var got = 0
    for _ in 0..200 { got = count_down(500) }
    assert_eq(got, 500)
}

fn test_overflow_in_the_body_after_the_arm() -> void {
    for _ in 0..200 { double_until(10, 1) }
    assert_eq(double_until(10, 1), 1024)
    var raised = false
    try {
        let _unused = double_until(70, 1)
    } catch _failure {
        raised = true
    }
    assert_eq(raised, true)
}

class Tree {
    pub var left: Tree?
    pub var right: Tree?
    pub var value: int

    fn init(self, left: Tree?, right: Tree?, value: int) {
        self.left = left
        self.right = right
        self.value = value
    }
}

fn grow(depth: int) -> Tree? {
    if depth == 0 { return null }
    return Tree(grow(depth - 1), grow(depth - 1), depth)
}

#: The null test, ahead of the frame: half the calls are the null children.
fn weight(node: Tree?) -> int {
    if node == null { return 0 }
    return node.value + weight(node.left) + weight(node.right)
}

#: The negated form: the arm runs for every present node.
fn present(node: Tree?) -> int {
    if node != null { return 1 }
    return 0
}

fn test_a_null_test_ahead_of_the_frame() -> void {
    let tree = grow(12)
    var got = 0
    for _ in 0..40 { got = weight(tree) }
    assert_eq(got, 8178)
    var count = 0
    for _ in 0..2000 {
        count = present(tree) + present(null)
    }
    assert_eq(count, 1)
}

#: A bool literal returned from the arm.
fn holds(node: Tree?, v: int) -> bool {
    if node == null { return false }
    if node.value == v { return true }
    return holds(node.left, v) or holds(node.right, v)
}

fn is_small(n: int) -> bool {
    if n < 10 { return true }
    return n % 2 == 0
}

var g_visits = 0

#: A bare `return` from a function with nothing to return.
fn visit(node: Tree?) -> void {
    if node == null { return }
    g_visits += 1
    visit(node.left)
    visit(node.right)
}

fn test_bool_and_void_arms() -> void {
    let tree = grow(10)
    var found = 0
    for v in 0..30 {
        if holds(tree, v) { found += 1 }
    }
    assert_eq(found, 10)
    var small = 0
    for i in 0..400 {
        if is_small(i) { small += 1 }
    }
    assert_eq(small, 205)
    g_visits = 0
    for _ in 0..20 { visit(tree) }
    assert_eq(g_visits, 20460)
}

#: Two int parameters compared against each other, the base case of a
#: divide-and-conquer recursion.
fn spans(lo: int, hi: int) -> int {
    if lo >= hi { return 0 }
    let mid = (lo + hi) // 2
    return 1 + spans(lo, mid) + spans(mid + 1, hi)
}

fn clamp_low(x: int, floor: int) -> int {
    if x < floor { return floor }
    return x
}

fn test_two_parameter_tests() -> void {
    var got = 0
    for _ in 0..20 { got = spans(0, 5000) }
    assert_eq(got, 5000)
    var total = 0
    for i in -200..200 { total += clamp_low(i, 17) }
    assert_eq(total, 23453)
}
