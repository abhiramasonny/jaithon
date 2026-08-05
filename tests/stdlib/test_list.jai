#: `std.list` — the combinators over `list[T]` that the built-in methods leave
#: out. The interesting cases are the empty list and the boundaries, because
#: that is where an off-by-one in a slice hides.
import std.list as list

from std.test import assert_eq, assert_true, assert_false, assert_throws

fn test_partition_keeps_relative_order() -> void {
    let (even, odd) = list.partition([5, 2, 8, 1, 4], |n| n % 2 == 0)
    assert_eq(even, [2, 8, 4])
    assert_eq(odd, [5, 1])
}

fn test_partition_of_empty_yields_two_empties() -> void {
    let (yes, no) = list.partition([], |n| true)
    assert_eq(yes, [])
    assert_eq(no, [])
}

fn test_group_by_preserves_order_within_a_group() -> void {
    let groups = list.group_by([1, 2, 3, 4, 5], |n| n % 2)
    assert_eq(groups[1], [1, 3, 5])
    assert_eq(groups[0], [2, 4])
    assert_eq(groups.len(), 2)
}

fn test_chunk_by_cuts_where_the_predicate_fails() -> void {
    assert_eq(list.chunk_by([1, 2, 4, 5, 7], |a, b| b - a == 1), [[1, 2], [4, 5], [7]])
    assert_eq(list.chunk_by([], |a, b| true), [])
    assert_eq(list.chunk_by([9], |a, b| true), [[9]])
}

fn test_split_at_accepts_both_ends_and_rejects_beyond() -> void {
    let (head, tail) = list.split_at([1, 2, 3], 0)
    assert_eq(head, [])
    assert_eq(tail, [1, 2, 3])

    let (all, none) = list.split_at([1, 2, 3], 3)
    assert_eq(all, [1, 2, 3])
    assert_eq(none, [])

    assert_throws(IndexError, || list.split_at([1, 2, 3], 4))
    assert_throws(IndexError, || list.split_at([1, 2, 3], -1))
}

fn test_split_when_always_yields_one_more_part_than_separators() -> void {
    assert_eq(list.split_when([1, 0, 2, 3, 0], |n| n == 0), [[1], [2, 3], []])
    assert_eq(list.split_when([], |n| n == 0), [[]])
    assert_eq(list.split_when([0], |n| n == 0), [[], []])
}

fn test_rotate_is_modular_in_both_directions() -> void {
    assert_eq(list.rotate([1, 2, 3, 4, 5], 2), [3, 4, 5, 1, 2])
    assert_eq(list.rotate([1, 2, 3, 4, 5], -2), [4, 5, 1, 2, 3])
    assert_eq(list.rotate([1, 2, 3], 3), [1, 2, 3])
    assert_eq(list.rotate([1, 2, 3], 7), [2, 3, 1])
    assert_eq(list.rotate([], 3), [])
}

fn test_interleave_runs_until_every_list_is_spent() -> void {
    assert_eq(list.interleave([[1, 2, 3], [10, 20]]), [1, 10, 2, 20, 3])
    assert_eq(list.interleave([]), [])
    assert_eq(list.interleave([[], [1]]), [1])
}

fn test_transpose_swaps_rows_and_columns() -> void {
    assert_eq(list.transpose([[1, 2, 3], [4, 5, 6]]), [[1, 4], [2, 5], [3, 6]])
    assert_eq(list.transpose([]), [])
}

fn test_run_length_round_trips() -> void {
    let runs = list.run_length_encode([1, 1, 2, 3, 3, 3])
    assert_eq(runs, [(1, 2), (2, 1), (3, 3)])
    assert_eq(list.run_length_decode(runs), [1, 1, 2, 3, 3, 3])
    assert_eq(list.run_length_encode([]), [])
}

fn test_frequency_and_most_common() -> void {
    let counts = list.frequency(["a", "b", "a", "c", "a", "b"])
    assert_eq(counts["a"], 3)
    assert_eq(counts["b"], 2)
    assert_eq(counts["c"], 1)
    assert_eq(list.most_common(["a", "b", "a"], 1), [("a", 2)])
}

fn test_argmin_and_argmax_return_the_first_extreme() -> void {
    assert_eq(list.argmin([3, 1, 4, 1]), 1)
    assert_eq(list.argmax([3, 1, 4, 4]), 2)
}

fn test_cumulative_sum_and_differences_are_inverse() -> void {
    assert_eq(list.cumulative_sum([1, 2, 3, 4]), [1, 3, 6, 10])
    assert_eq(list.differences([1, 3, 6, 10]), [2, 3, 4])
    assert_eq(list.cumulative_sum([]), [])
    assert_eq(list.differences([5]), [])
}

fn test_zip_with_stops_at_the_shorter_input() -> void {
    assert_eq(list.zip_with([1, 2, 3], [10, 20], |a, b| a + b), [11, 22])
    assert_eq(list.zip_with([], [1, 2], |a, b| a), [])
}

fn test_unzip_undoes_a_pair_list() -> void {
    let (lefts, rights) = list.unzip([(1, "a"), (2, "b")])
    assert_eq(lefts, [1, 2])
    assert_eq(rights, ["a", "b"])
}

fn test_scan_reports_every_intermediate_state() -> void {
    # `initial` comes first, so the result is one longer than the input and
    # its last element is the whole fold.
    assert_eq(list.scan([1, 2, 3], 0, |acc, x| acc + x), [0, 1, 3, 6])
    assert_eq(list.scan([], 0, |acc, x| acc + x), [0])
}

fn test_take_while_and_drop_while_split_at_the_same_point() -> void {
    let xs = [1, 2, 3, 1]
    assert_eq(list.take_while(xs, |n| n < 3), [1, 2])
    assert_eq(list.drop_while(xs, |n| n < 3), [3, 1])
    assert_eq(list.take_while(xs, |n| false), [])
    assert_eq(list.drop_while(xs, |n| true), [])
}

fn test_is_sorted_holds_for_empty_and_single() -> void {
    assert_true(list.is_sorted([]))
    assert_true(list.is_sorted([1]))
    assert_true(list.is_sorted([1, 1, 2]))
    assert_false(list.is_sorted([2, 1]))
}

fn test_permutations_and_combinations_count_correctly() -> void {
    assert_eq(list.permutations([1, 2, 3]).len(), 6)
    assert_eq(list.permutations([1, 2, 3], 2).len(), 6)
    assert_eq(list.combinations([1, 2, 3, 4], 2).len(), 6)
    assert_eq(list.combinations([1, 2], 0), [[]])
    assert_eq(list.power_set([1, 2]).len(), 4)
}

fn test_compare_orders_lexicographically() -> void {
    assert_eq(list.compare([1, 2], [1, 3]), -1)
    assert_eq(list.compare([1, 2], [1, 2]), 0)
    assert_eq(list.compare([1, 2, 0], [1, 2]), 1)
    assert_eq(list.compare([], []), 0)
}

fn test_is_permutation_ignores_order_but_not_multiplicity() -> void {
    assert_true(list.is_permutation([1, 2, 2], [2, 1, 2]))
    assert_false(list.is_permutation([1, 2, 2], [1, 2]))
    assert_true(list.is_permutation([], []))
}
