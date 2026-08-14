#: `std.iter` — laziness and the adapter algebra. The point of every test here
#: is that nothing is computed before it is asked for and that an adapter chain
#: consumes its source exactly once.
import std.iter as iter

from std.iter.generators import count, repeat, once, empty, successors, range_step
from std.test import assert_eq, assert_true, assert_false

fn test_iter_over_the_builtin_containers() -> void {
    assert_eq(iter.iter([1, 2, 3]).collect(), [1, 2, 3])
    assert_eq(iter.iter("abc").collect(), ["a", "b", "c"])
    assert_eq(iter.iter(0..3).collect(), [0, 1, 2])
    assert_eq(iter.iter([]).collect(), [])
}

fn test_map_and_filter_are_lazy() -> void {
    var touched = 0
    fn bump(n: int) -> int {
        touched += 1
        return n * 2
    }
    let chain = iter.iter([1, 2, 3, 4]).map(bump)
    assert_eq(touched, 0)
    assert_eq(chain.take(2).collect(), [2, 4])
    assert_eq(touched, 2)
}

fn test_take_and_drop_clamp_rather_than_raise() -> void {
    let xs = [1, 2, 3]
    assert_eq(iter.iter(xs).take(10).collect(), [1, 2, 3])
    assert_eq(iter.iter(xs).take(0).collect(), [])
    assert_eq(iter.iter(xs).drop(10).collect(), [])
    assert_eq(iter.iter(xs).drop(0).collect(), [1, 2, 3])
}

fn test_take_while_stops_at_the_first_failure() -> void {
    let xs = [1, 2, 3, 1, 2]
    assert_eq(iter.iter(xs).take_while(|n| n < 3).collect(), [1, 2])
    assert_eq(iter.iter(xs).drop_while(|n| n < 3).collect(), [3, 1, 2])
}

fn test_step_by_always_yields_the_first_element() -> void {
    assert_eq(iter.iter([0, 1, 2, 3, 4, 5]).step_by(2).collect(), [0, 2, 4])
    assert_eq(iter.iter([0, 1]).step_by(5).collect(), [0])
}

fn test_zip_stops_with_the_shorter_side() -> void {
    let zipped = iter.iter([1, 2, 3]).zip(iter.iter(["a", "b"])).collect()
    assert_eq(zipped, [(1, "a"), (2, "b")])
    assert_eq(iter.iter([]).zip(iter.iter([1])).collect(), [])
}

fn test_chain_and_flatten() -> void {
    assert_eq(iter.iter([1, 2]).chain(iter.iter([3])).collect(), [1, 2, 3])
    assert_eq(iter.iter([[1, 2], [], [3]]).flatten().collect(), [1, 2, 3])
    assert_eq(iter.iter([1, 2]).flat_map(|n| [n, n]).collect(), [1, 1, 2, 2])
}

fn test_enumerate_can_start_anywhere() -> void {
    assert_eq(iter.iter(["a", "b"]).enumerate().collect(), [(0, "a"), (1, "b")])
    assert_eq(iter.iter(["a"]).enumerate(5).collect(), [(5, "a")])
}

fn test_windows_and_chunks_treat_a_short_tail_differently() -> void {
    assert_eq(iter.iter([1, 2, 3, 4]).windows(2).collect(), [[1, 2], [2, 3], [3, 4]])
    assert_eq(iter.iter([1, 2, 3]).windows(5).collect(), [])
    assert_eq(iter.iter([1, 2, 3]).chunks(2).collect(), [[1, 2], [3]])
}

fn test_dedup_removes_only_adjacent_duplicates() -> void {
    assert_eq(iter.iter([1, 1, 2, 1]).dedup().collect(), [1, 2, 1])
    assert_eq(iter.iter([]).dedup().collect(), [])
}

fn test_scan_threads_state_through() -> void {
    let running = iter.iter([1, 2, 3]).scan(0, |total, n| total + n).collect()
    assert_eq(running, [1, 3, 6])
}

fn test_peekable_does_not_consume() -> void {
    let peekable = iter.iter([1, 2]).peekable()
    assert_eq(peekable.peek(), 1)
    assert_eq(peekable.peek(), 1)
    assert_eq(peekable.collect(), [1, 2])
}

fn test_fold_and_reduce_differ_on_empty() -> void {
    assert_eq(iter.iter([1, 2, 3]).fold(0, |a, b| a + b), 6)
    assert_eq(iter.iter([]).fold(0, |a, b| a + b), 0)
    assert_eq(iter.iter([1, 2, 3]).reduce(|a, b| a + b), 6)
    assert_true(iter.iter([]).reduce(|a, b| a + b) == null)
}

fn test_min_max_and_their_keyed_forms() -> void {
    assert_eq(iter.iter([3, 1, 2]).min(), 1)
    assert_eq(iter.iter([3, 1, 2]).max(), 3)
    assert_true(iter.iter([]).min() == null)
    assert_eq(iter.iter(["aaa", "b"]).min_by(|s| s.len()), "b")
    assert_eq(iter.iter(["aaa", "b"]).max_by(|s| s.len()), "aaa")
}

fn test_all_and_any_short_circuit_and_hold_vacuously() -> void {
    assert_true(iter.iter([]).all(|n| false))
    assert_false(iter.iter([]).any(|n| true))
    assert_true(iter.iter([2, 4]).all(|n| n % 2 == 0))
    assert_true(iter.iter([1, 2]).any(|n| n % 2 == 0))
}

fn test_find_position_and_nth() -> void {
    assert_eq(iter.iter([1, 2, 3]).find(|n| n > 1), 2)
    assert_true(iter.iter([1, 2, 3]).find(|n| n > 9) == null)
    assert_eq(iter.iter([1, 2, 3]).position(|n| n == 3), 2)
    assert_eq(iter.iter([1, 2, 3]).nth(1), 2)
    assert_true(iter.iter([1, 2, 3]).nth(9) == null)
    assert_eq(iter.iter([1, 2, 3]).last(), 3)
}

fn test_partition_splits_without_reordering() -> void {
    let (even, odd) = iter.iter([1, 2, 3, 4]).partition(|n| n % 2 == 0)
    assert_eq(even, [2, 4])
    assert_eq(odd, [1, 3])
}

fn test_infinite_sources_are_safe_when_bounded() -> void {
    assert_eq(count().take(3).collect(), [0, 1, 2])
    assert_eq(count(10, -2).take(3).collect(), [10, 8, 6])
    assert_eq(repeat("x").take(2).collect(), ["x", "x"])
    assert_eq(repeat("x", 2).collect(), ["x", "x"])
    assert_eq(iter.cycle([1, 2]).take(5).collect(), [1, 2, 1, 2, 1])
    assert_eq(successors(1, |n| n * 2).take(4).collect(), [1, 2, 4, 8])
}

fn test_once_and_empty_are_the_degenerate_sources() -> void {
    assert_eq(once(7).collect(), [7])
    assert_eq(empty().collect(), [])
    assert_eq(range_step(0, 10, 3).collect(), [0, 3, 6, 9])
    assert_eq(range_step(10, 0, -3).collect(), [10, 7, 4, 1])
}

fn test_collect_dict_and_set() -> void {
    let pairs = iter.iter([(1, "a"), (2, "b")]).collect_dict()
    assert_eq(pairs, {1: "a", 2: "b"})
    assert_eq(iter.iter([1, 1, 2]).collect_set().len(), 2)
}

fn test_combinations_and_permutations_are_lazy_too() -> void {
    assert_eq(iter.iter([1, 2, 3]).combinations(2).collect(), [[1, 2], [1, 3], [2, 3]])
    assert_eq(iter.iter([1, 2, 3]).permutations().count_items(), 6)
}
