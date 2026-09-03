#: `std.itertools` — the port is only worth having if the order and the edge
#: cases are CPython's, so most of what is checked here is the awkward end of
#: the surface: `r > n`, `r == 0`, an empty pool, an exhausted `tee` branch,
#: and a `groupby` group read after the walk has moved past it. Laziness is
#: checked the same way `test_iter.jai` checks it — by counting how far into an
#: infinite source a bounded consumer actually reaches.
#:
#: The exhaustive comparison against CPython lives beside this file in
#: `tests/stdlib/py/itertools_diff.py`, which enumerates the same edge cases
#: over generated inputs and diffs both runtimes exactly.
import std.itertools as it

from std.test import assert_eq, assert_true, assert_throws

fn test_count_walks_ints_and_floats() -> void {
    assert_eq(it.count().take(3).collect(), [0, 1, 2])
    assert_eq(it.count(10).take(3).collect(), [10, 11, 12])
    assert_eq(it.count(10, -2).take(3).collect(), [10, 8, 6])
    assert_eq(it.count(2.5, 0.5).take(3).collect(), [2.5, 3.0, 3.5])
    assert_eq(it.count(1, 0).take(3).collect(), [1, 1, 1])
    assert_throws(TypeError, || it.count("a"))
}

fn test_cycle_replays_and_stays_empty() -> void {
    assert_eq(it.cycle([1, 2]).take(5).collect(), [1, 2, 1, 2, 1])
    assert_eq(it.cycle("ab").take(3).collect(), ["a", "b", "a"])
    assert_eq(it.cycle([]).collect(), [])
}

fn test_repeat_counts_down_and_never_raises() -> void {
    assert_eq(it.repeat("x", 3).collect(), ["x", "x", "x"])
    assert_eq(it.repeat("x", 0).collect(), [])
    assert_eq(it.repeat("x", -1).collect(), [])
    assert_eq(it.repeat(7).take(2).collect(), [7, 7])
}

fn test_accumulate_totals_with_and_without_an_initial() -> void {
    assert_eq(it.accumulate([1, 2, 3, 4]).collect(), [1, 3, 6, 10])
    assert_eq(it.accumulate([1, 2, 3], |a, b| a * b).collect(), [1, 2, 6])
    assert_eq(it.accumulate([1, 2, 3], null, 100).collect(), [100, 101, 103, 106])
    assert_eq(it.accumulate([]).collect(), [])
    assert_eq(it.accumulate([], null, 9).collect(), [9])
    assert_eq(it.accumulate([5]).collect(), [5])
}

fn test_chain_concatenates_lazily_in_both_dimensions() -> void {
    assert_eq(it.chain([1, 2], (3,), "45").collect(), [1, 2, 3, "4", "5"])
    assert_eq(it.chain().collect(), [])
    assert_eq(it.chain([], []).collect(), [])
    assert_eq(it.chain.from_iterable(["ab", "cd"]).collect(), ["a", "b", "c", "d"])
    assert_eq(it.chain.from_iterable([]).collect(), [])
    let endless = it.chain.from_iterable(it.repeat([1, 2]))
    assert_eq(endless.take(5).collect(), [1, 2, 1, 2, 1])
}

fn test_compress_stops_with_the_shorter_side() -> void {
    assert_eq(it.compress("abcdef", [1, 0, 1, 0, 1, 1]).collect(), ["a", "c", "e", "f"])
    assert_eq(it.compress("abc", [1]).collect(), ["a"])
    assert_eq(it.compress("a", [1, 1, 1]).collect(), ["a"])
    assert_eq(it.compress([1, 2, 3], ["", "x", []]).collect(), [2])
    assert_eq(it.compress([], []).collect(), [])
}

fn test_dropwhile_and_takewhile_split_at_the_first_failure() -> void {
    assert_eq(it.dropwhile(|n| n < 5, [1, 4, 6, 4, 1]).collect(), [6, 4, 1])
    assert_eq(it.takewhile(|n| n < 5, [1, 4, 6, 4, 1]).collect(), [1, 4])
    assert_eq(it.dropwhile(|n| n < 5, []).collect(), [])
    assert_eq(it.takewhile(|n| n < 5, []).collect(), [])
    assert_eq(it.dropwhile(|n| true, [1, 2]).collect(), [])
    assert_eq(it.takewhile(|n| false, [1, 2]).collect(), [])
}

fn test_filterfalse_keeps_what_filter_would_drop() -> void {
    assert_eq(it.filterfalse(|n| n % 2, [0, 1, 2, 3, 4]).collect(), [0, 2, 4])
    assert_eq(it.filterfalse(null, [1, 0, 2, 0]).collect(), [0, 0])
    assert_eq(it.filterfalse(null, ["", "a"]).collect(), [""])
    assert_eq(it.filterfalse(|n| true, [1, 2]).collect(), [])
}

fn test_groupby_runs_in_order_with_a_key() -> void {
    let runs: list[any] = []
    for pair in it.groupby("AAAABBBCCDAABBB") { runs.push((pair[0], pair[1].collect())) }
    assert_eq(runs.map(|run| run[0]), ["A", "B", "C", "D", "A", "B"])
    assert_eq(runs[0][1], ["A", "A", "A", "A"])
    assert_eq(runs[3][1], ["D"])

    let by_parity: list[any] = []
    for pair in it.groupby([1, 3, 2, 4, 5], |n| n % 2) { by_parity.push(pair[0]) }
    assert_eq(by_parity, [1, 0, 1])
    assert_eq(it.groupby([]).collect(), [])
}

fn test_groupby_groups_are_views_not_copies() -> void {
    let held: list[any] = []
    for pair in it.groupby([1, 1, 2, 2]) { held.push(pair[1]) }
    assert_eq(held.len(), 2)
    assert_eq(held[0].collect(), [])
    assert_eq(held[1].collect(), [])

    let keys: list[any] = []
    for pair in it.groupby([1, 1, 1, 2]) { keys.push(pair[0]) }
    assert_eq(keys, [1, 2])
}

fn test_islice_slices_by_position() -> void {
    assert_eq(it.islice("abcdefg", 2).collect(), ["a", "b"])
    assert_eq(it.islice("abcdefg", 2, 5).collect(), ["c", "d", "e"])
    assert_eq(it.islice("abcdefg", 2, null).collect(), ["c", "d", "e", "f", "g"])
    assert_eq(it.islice("abcdefg", 0, null, 2).collect(), ["a", "c", "e", "g"])
    assert_eq(it.islice("abcdefg", 1, 6, 2).collect(), ["b", "d", "f"])
    assert_eq(it.islice("abc", null).collect(), ["a", "b", "c"])
    assert_eq(it.islice("abc", 5, 9).collect(), [])
    assert_eq(it.islice("abc", 2, 1).collect(), [])
    assert_eq(it.islice(it.count(), 3).collect(), [0, 1, 2])
}

fn test_islice_refuses_what_a_lazy_source_cannot_answer() -> void {
    assert_throws(ValueError, || it.islice("abc", -1).collect())
    assert_throws(ValueError, || it.islice("abc", 0, 2, 0).collect())
    assert_throws(ValueError, || it.islice("abc", 0, 2, -1).collect())
    assert_throws(TypeError, || it.islice("abc").collect())
    assert_throws(TypeError, || it.islice("abc", 0, 1, 1, 1).collect())
}

fn test_islice_leaves_the_rest_of_a_shared_source() -> void {
    let source = it.count()
    assert_eq(it.islice(source, 2).collect(), [0, 1])
    assert_eq(source.take(2).collect(), [2, 3])
}

fn test_pairwise_needs_two_elements() -> void {
    assert_eq(it.pairwise("abcd").collect(), [("a", "b"), ("b", "c"), ("c", "d")])
    assert_eq(it.pairwise([1]).collect(), [])
    assert_eq(it.pairwise([]).collect(), [])
    assert_eq(it.pairwise([1, 2]).collect(), [(1, 2)])
}

fn test_batched_cuts_into_tuples() -> void {
    assert_eq(it.batched("abcdefg", 3).collect(), [("a", "b", "c"), ("d", "e", "f"), ("g",)])
    assert_eq(it.batched("abcdef", 3).collect(), [("a", "b", "c"), ("d", "e", "f")])
    assert_eq(it.batched([], 3).collect(), [])
    assert_eq(it.batched([1, 2], 1).collect(), [(1,), (2,)])
    assert_eq(it.batched("abcdef", 3, true).collect(), [("a", "b", "c"), ("d", "e", "f")])
    assert_throws(ValueError, || it.batched("abcd", 3, true).collect())
    assert_throws(ValueError, || it.batched("abcd", 0))
}

fn test_starmap_spreads_each_element() -> void {
    assert_eq(it.starmap(|a, b| a ** b, [(2, 5), (3, 2)]).collect(), [32, 9])
    assert_eq(it.starmap(|a, b| a + b, [[1, 2]]).collect(), [3])
    assert_eq(it.starmap(|a, b| a + b, []).collect(), [])
}

fn test_tee_branches_run_independently() -> void {
    let pair = it.tee([1, 2, 3])
    assert_eq(pair[0].collect(), [1, 2, 3])
    assert_eq(pair[1].collect(), [1, 2, 3])

    let three = it.tee("abc", 3)
    assert_eq(three.len(), 3)
    assert_eq(three[2].collect(), ["a", "b", "c"])

    assert_eq(it.tee([1], 0), tuple([]))
    assert_eq(it.tee([], 2)[0].collect(), [])
    assert_throws(ValueError, || it.tee([1], -1))
}

fn test_an_exhausted_tee_branch_stays_exhausted() -> void {
    let pair = it.tee([1, 2])
    let ahead = pair[0]
    let behind = pair[1]
    assert_eq(ahead.collect(), [1, 2])
    assert_true(ahead.next() == null)
    assert_eq(behind.collect(), [1, 2])
    assert_true(behind.next() == null)
    assert_true(ahead.next() == null)
}

fn test_tee_interleaves_without_losing_a_branch() -> void {
    let pair = it.tee([1, 2, 3, 4])
    let left = pair[0]
    let right = pair[1]
    assert_eq(left.next(), 1)
    assert_eq(left.next(), 2)
    assert_eq(right.next(), 1)
    assert_eq(left.next(), 3)
    assert_eq(right.next(), 2)
    assert_eq(right.collect(), [3, 4])
    assert_eq(left.collect(), [4])
}

fn test_zip_longest_pads_the_short_sides() -> void {
    assert_eq(it.zip_longest([1, 2], "a").collect(), [(1, "a"), (2, null)])
    assert_eq(it.zip_longest().collect(), [])
    assert_eq(it.zip_longest([1, 2]).collect(), [(1,), (2,)])
    assert_eq(it.zip_longest([], []).collect(), [])
    assert_eq(it.zip_longest.with_fill(0, [1, 2], "a").collect(), [(1, "a"), (2, 0)])
    assert_eq(
        it.zip_longest.with_fill("-", "ab", "xyz").collect(),
        [("a", "x"), ("b", "y"), ("-", "z")]
    )
}

fn test_product_counts_rightmost_first() -> void {
    assert_eq(it.product([1, 2], "ab").collect(), [(1, "a"), (1, "b"), (2, "a"), (2, "b")])
    assert_eq(it.product().collect(), [tuple([])])
    assert_eq(it.product("ab", []).collect(), [])
    assert_eq(it.product("ab").collect(), [("a",), ("b",)])
    assert_eq(
        it.product.repeated(2, "ab").collect(),
        [("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")]
    )
    assert_eq(it.product.repeated(0, "ab").collect(), [tuple([])])
    assert_eq(it.product.repeated(1).collect(), [tuple([])])
    assert_throws(ValueError, || it.product.repeated(-1, "ab"))
}

fn test_permutations_covers_r_zero_and_r_over_n() -> void {
    assert_eq(
        it.permutations([1, 2, 3], 2).collect(),
        [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]
    )
    assert_eq(it.permutations("abc").collect().len(), 6)
    assert_eq(it.permutations("ab", 3).collect(), [])
    assert_eq(it.permutations("ab", 0).collect(), [tuple([])])
    assert_eq(it.permutations("", 0).collect(), [tuple([])])
    assert_eq(it.permutations("").collect(), [tuple([])])
    assert_eq(it.permutations([1, 1]).collect(), [(1, 1), (1, 1)])
}

fn test_combinations_covers_r_zero_and_r_over_n() -> void {
    assert_eq(it.combinations("abc", 2).collect(), [("a", "b"), ("a", "c"), ("b", "c")])
    assert_eq(it.combinations("abcd", 2).collect().len(), 6)
    assert_eq(it.combinations("ab", 3).collect(), [])
    assert_eq(it.combinations("ab", 0).collect(), [tuple([])])
    assert_eq(it.combinations("", 0).collect(), [tuple([])])
    assert_eq(it.combinations("", 1).collect(), [])
}

fn test_combinations_with_replacement_lets_indices_repeat() -> void {
    assert_eq(
        it.combinations_with_replacement("abc", 2).collect(),
        [("a", "a"), ("a", "b"), ("a", "c"), ("b", "b"), ("b", "c"), ("c", "c")]
    )
    assert_eq(
        it.combinations_with_replacement("ab", 3).collect(),
        [("a", "a", "a"), ("a", "a", "b"), ("a", "b", "b"), ("b", "b", "b")]
    )
    assert_eq(it.combinations_with_replacement("ab", 0).collect(), [tuple([])])
    assert_eq(it.combinations_with_replacement("", 0).collect(), [tuple([])])
    assert_eq(it.combinations_with_replacement("", 2).collect(), [])
    assert_throws(ValueError, || it.combinations_with_replacement("ab", -1))
}

fn test_nothing_is_computed_before_it_is_asked_for() -> void {
    var touched = 0
    fn watch(n: int) -> bool {
        touched += 1
        return n < 100
    }
    let bounded = it.takewhile(watch, it.count())
    assert_eq(touched, 0)
    assert_eq(bounded.take(3).collect(), [0, 1, 2])
    assert_eq(touched, 3)

    var built = 0
    fn make(n: int) -> list[any] {
        built += 1
        return [n]
    }
    assert_eq(it.chain.from_iterable(it.count().map(make)).take(2).collect(), [0, 1])
    assert_eq(built, 2)
}

fn test_the_results_are_tuples_where_cpython_yields_tuples() -> void {
    assert_eq(type_of(it.pairwise([1, 2]).collect()[0]), "tuple")
    assert_eq(type_of(it.batched([1, 2], 2).collect()[0]), "tuple")
    assert_eq(type_of(it.product([1], [2]).collect()[0]), "tuple")
    assert_eq(type_of(it.permutations([1, 2]).collect()[0]), "tuple")
    assert_eq(type_of(it.combinations([1, 2], 2).collect()[0]), "tuple")
    assert_eq(type_of(it.combinations_with_replacement([1], 2).collect()[0]), "tuple")
    assert_eq(type_of(it.zip_longest([1], [2]).collect()[0]), "tuple")
    assert_eq(type_of(it.tee([1], 2)), "tuple")
}
