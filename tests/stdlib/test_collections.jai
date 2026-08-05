#: `std.collections` and the containers under it. A `Counter` measures and may
#: go negative; a `Bag` counts multiplicities and may not. The tests below turn
#: on exactly that difference, and on the invariants of the heap and the deque.
import std.collections as collections

from std.collections.heap import Heap
from std.collections.ring import Deque
from std.test import assert_eq, assert_true, assert_false, assert_throws

fn test_counter_never_stores_a_zero() -> void {
    let counts = collections.Counter[str].from_list(["a", "b", "a"])
    assert_eq(counts.get("a"), 2)
    assert_eq(counts.get("b"), 1)
    assert_eq(counts.get("absent"), 0)
    counts.add("b", -1)
    assert_eq(counts.get("b"), 0)
    assert_eq(counts.len(), 1)  # the zero entry is gone
}

fn test_counter_total_and_most_common() -> void {
    let counts = collections.Counter[str].from_list(["a", "b", "a", "c", "a", "b"])
    assert_eq(counts.total(), 6)
    assert_eq(counts.most_common(2), [("a", 3), ("b", 2)])
    assert_eq(counts.most_common(0), [])
    assert_eq(counts.most_common(99).len(), 3)  # clamped, not padded
}

fn test_counter_subtract_may_go_negative() -> void {
    let counts = collections.Counter[str].from_list(["a", "a", "b"])
    counts.subtract(collections.Counter[str].from_list(["a", "c"]))
    assert_eq(counts.get("a"), 1)
    assert_eq(counts.get("c"), -1)
    assert_eq(counts.total(), 1)
}

fn test_counter_from_dict_drops_the_zeroes() -> void {
    let counts = collections.Counter[str].from_dict({"a": 2, "b": 0})
    assert_eq(counts.len(), 1)
    assert_eq(counts.get("b"), 0)
}

fn test_bag_counts_multiplicity_and_refuses_to_go_negative() -> void {
    let bag = collections.Bag[str].from_list(["a", "a", "b"])
    assert_eq(bag.len(), 3)
    assert_eq(bag.distinct_count(), 2)
    assert_eq(bag.count_of("a"), 2)
    assert_eq(bag.count_of("absent"), 0)
    assert_throws(KeyError, || bag.remove("a", 5))
    assert_eq(bag.discard("a", 5), 2)  # discard removes what it can
    assert_eq(bag.count_of("a"), 0)
    assert_false(bag.is_empty())
}

fn test_bag_union_and_intersection_use_max_and_min() -> void {
    let left = collections.Bag[str].from_list(["a", "a", "b"])
    let right = collections.Bag[str].from_list(["a", "c"])
    assert_eq(left.union(right).count_of("a"), 2)
    assert_eq(left.union(right).count_of("c"), 1)
    assert_eq(left.intersection(right).count_of("a"), 1)
    assert_eq(left.intersection(right).count_of("b"), 0)
}

fn test_heap_pops_in_order_and_empties_cleanly() -> void {
    let heap = Heap[int].from_list([5, 1, 4, 1, 3])
    assert_eq(heap.len(), 5)
    assert_eq(heap.peek(), 1)
    assert_eq(heap.drain(), [1, 1, 3, 4, 5])
    assert_true(heap.is_empty())
    assert_true(heap.peek() == null)
    assert_throws(IndexError, || heap.pop())
}

fn test_heap_takes_its_own_ordering() -> void {
    let highest = Heap[int].from_list([1, 5, 3], |a, b| a > b)
    assert_eq(highest.drain(), [5, 3, 1])
}

fn test_heap_push_pop_is_not_pop_then_push() -> void {
    let heap = Heap[int].from_list([2, 3])
    # push_pop of a value smaller than the top returns it untouched.
    assert_eq(heap.push_pop(1), 1)
    assert_eq(heap.sorted(), [2, 3])
    # replace always yields the old top, whatever is pushed.
    assert_eq(heap.replace(9), 2)
    assert_eq(heap.sorted(), [3, 9])
}

fn test_deque_works_from_both_ends() -> void {
    let deque = Deque[int].from_list([2, 3])
    deque.push_front(1)
    deque.push_back(4)
    assert_eq(deque.to_list(), [1, 2, 3, 4])
    assert_eq(deque.pop_front(), 1)
    assert_eq(deque.pop_back(), 4)
    assert_eq(deque.len(), 2)
    assert_eq(deque.front(), 2)
    assert_eq(deque.back(), 3)
}

fn test_empty_deque_reports_null_and_raises_on_pop() -> void {
    let deque = Deque[int]()
    assert_true(deque.is_empty())
    assert_true(deque.front() == null)
    assert_true(deque.back() == null)
    assert_throws(IndexError, || deque.pop_front())
    assert_throws(IndexError, || deque.pop_back())
}

fn test_deque_rotation_is_modular_in_both_directions() -> void {
    let deque = Deque[int].from_list([1, 2, 3, 4, 5])
    deque.rotate(2)
    assert_eq(deque.to_list(), [4, 5, 1, 2, 3])
    deque.rotate(-2)
    assert_eq(deque.to_list(), [1, 2, 3, 4, 5])
    deque.rotate(5)
    assert_eq(deque.to_list(), [1, 2, 3, 4, 5])
}

fn test_deque_indexing_is_bounds_checked() -> void {
    let deque = Deque[int].from_list([1, 2, 3])
    assert_eq(deque.get(0), 1)
    assert_eq(deque.get(2), 3)
    assert_throws(IndexError, || deque.get(3))
    deque.set(1, 9)
    assert_eq(deque.to_list(), [1, 9, 3])
}
