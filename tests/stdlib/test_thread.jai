#: `std.thread`'s list operations. The module had NO tests and no callers, and
#: that is exactly how it came to ship broken: `parallel_map`, `parallel_for`
#: and `parallel_reduce` each split their input first and took a sequential path
#: only when the whole input landed in ONE chunk, then handed a Jaithon closure
#: to `__prim__.thread_spawn`, which accepts native primitives only. So every
#: call above the split threshold raised
#: `TypeError: the worker must be a native primitive, not fn` -- while the
#: module's own four-element docstring examples passed.
#:
#: Every case below therefore runs BOTH sides of that old threshold: a handful
#: of elements, and several thousand. A test that only does the former is the
#: test that was missing.
from std.thread import parallel_map, parallel_for, parallel_reduce
from std.thread import Mutex, Atomic

from std.test import assert_eq, assert_true

fn _range_list(n: int) -> list[any] {
    var xs: list[any] = []
    for i in 0..n { xs.push(i) }
    return xs
}

fn test_parallel_map_small_and_large() -> void {
    assert_eq(parallel_map([1, 2, 3, 4], |x| x * x), [1, 4, 9, 16])

    # Above the old split threshold, where this used to raise.
    let big = _range_list(5000)
    let doubled = parallel_map(big, |x| x * 2)
    assert_eq(doubled.len(), 5000)
    assert_eq(doubled[0], 0)
    assert_eq(doubled[4999], 9998)
}

fn test_parallel_map_matches_a_plain_map() -> void {
    let big = _range_list(4000)
    assert_eq(parallel_map(big, |x| x * x), big.map(|x| x * x))
}

fn test_parallel_reduce_small_and_large() -> void {
    assert_eq(parallel_reduce([1, 2, 3, 4], 0, |a, b| a + b), 10)
    assert_eq(parallel_reduce([1, 2, 3, 4], 1, |a, b| a * b), 24)

    # n(n-1)/2 for 0..5000.
    assert_eq(parallel_reduce(_range_list(5000), 0, |a, b| a + b), 12497500)
}

fn test_parallel_reduce_on_an_empty_list_is_the_identity() -> void {
    assert_eq(parallel_reduce([], 0, |a, b| a + b), 0)
    assert_eq(parallel_reduce([], 7, |a, b| a + b), 7)
}

fn test_parallel_for_visits_every_element() -> void {
    var seen = 0
    parallel_for([1, 2, 3, 4], fn(x) -> void { seen += x })
    assert_eq(seen, 10)

    var count = 0
    parallel_for(_range_list(5000), fn(_x) -> void { count += 1 })
    assert_eq(count, 5000)
}

fn test_parallel_map_leaves_its_input_alone() -> void {
    let source = [3, 1, 2]
    let _ = parallel_map(source, |x| x + 1)
    assert_eq(source, [3, 1, 2])
}

fn test_mutex_and_atomic_are_usable() -> void {
    let counter = Atomic(0)
    counter.add(5)
    counter.add(3)
    assert_eq(counter.get(), 8)

    let lock = Mutex()
    lock.lock()
    lock.unlock()
    assert_true(lock.try_lock())
    lock.unlock()
}
