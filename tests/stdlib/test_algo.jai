#: `std.algo.sort` and `std.algo.search`. Every sort here is checked for the
#: same three things: it orders, it is stable where it claims to be, and it
#: leaves its input alone. The searches are checked at the boundaries, which is
#: where a binary search is wrong or nowhere.
import std.algo.search as search
import std.algo.sort as sort

from std.test import assert_eq, assert_true, assert_false, assert_throws

fn test_every_sort_agrees_on_the_same_input() -> void {
    let source = [5, 3, 8, 1, 9, 2, 7]
    let want = [1, 2, 3, 5, 7, 8, 9]
    assert_eq(sort.merge_sort(source), want)
    assert_eq(sort.quick_sort(source), want)
    assert_eq(sort.heap_sort(source), want)
    assert_eq(sort.insertion_sort(source), want)
    assert_eq(sort.tim_sort(source), want)
    assert_eq(sort.counting_sort(source), want)
    assert_eq(sort.radix_sort(source), want)
    assert_eq(source, [5, 3, 8, 1, 9, 2, 7])  # none of them mutate
}

fn test_sorts_handle_empty_and_single_inputs() -> void {
    assert_eq(sort.merge_sort([]), [])
    assert_eq(sort.quick_sort([]), [])
    assert_eq(sort.heap_sort([]), [])
    assert_eq(sort.tim_sort([7]), [7])
    assert_eq(sort.radix_sort([]), [])
    assert_eq(sort.counting_sort([4]), [4])
}

fn test_sorts_handle_duplicates_and_reversed_input() -> void {
    assert_eq(sort.merge_sort([2, 2, 1, 1]), [1, 1, 2, 2])
    assert_eq(sort.quick_sort([5, 4, 3, 2, 1]), [1, 2, 3, 4, 5])
    assert_eq(sort.heap_sort([1, 1, 1]), [1, 1, 1])
}

fn test_radix_and_counting_sort_accept_negatives() -> void {
    assert_eq(sort.radix_sort([3, -1, 0, -5, 2]), [-5, -1, 0, 2, 3])
    assert_eq(sort.counting_sort([3, -1, 0, -5, 2]), [-5, -1, 0, 2, 3])
}

fn test_merge_sort_is_stable() -> void {
    # Equal keys must come out in input order; an unstable sort passes the
    # ordering check above and fails this one.
    let pairs = [(1, "a"), (0, "b"), (1, "c"), (0, "d")]
    let sorted = sort.merge_sort(pairs, |x, y| sort.natural_order(x[0], y[0]))
    assert_eq(sorted, [(0, "b"), (0, "d"), (1, "a"), (1, "c")])
}

fn test_sort_by_key_reads_the_key_once_per_element() -> void {
    var reads = 0
    fn length_of(text: str) -> int {
        reads += 1
        return text.len()
    }
    assert_eq(sort.sort_by_key(["ccc", "a", "bb"], length_of), ["a", "bb", "ccc"])
    assert_eq(reads, 3)
}

fn test_natural_order_compares_across_the_builtin_types() -> void {
    assert_eq(sort.natural_order(1, 2), -1)
    assert_eq(sort.natural_order(2, 2), 0)
    assert_eq(sort.natural_order("b", "a"), 1)
    assert_eq(sort.natural_order(1, 1.0), 0)
}

fn test_bucket_sort_orders_floats() -> void {
    let sorted = sort.bucket_sort([0.5, 0.1, 0.9, 0.3])
    assert_eq(sorted, [0.1, 0.3, 0.5, 0.9])
    assert_eq(sort.bucket_sort([]), [])
}

fn test_partial_sort_and_nth_element() -> void {
    assert_eq(sort.partial_sort([5, 1, 4, 2, 3], 3), [1, 2, 3])
    assert_eq(sort.partial_sort([1, 2], 5), [1, 2])
    assert_eq(sort.partial_sort([1, 2], 0), [])
    assert_eq(sort.nth_element([5, 1, 4, 2, 3], 0), 1)
    assert_eq(sort.nth_element([5, 1, 4, 2, 3], 4), 5)
    assert_throws(IndexError, || sort.nth_element([1, 2], 5))
}

fn test_merge_of_two_sorted_runs() -> void {
    assert_eq(sort.merge([1, 3, 5], [2, 4]), [1, 2, 3, 4, 5])
    assert_eq(sort.merge([], [1]), [1])
    assert_eq(sort.merge([1], []), [1])
    assert_eq(sort.merge([], []), [])
}

fn test_topological_sort_respects_dependencies() -> void {
    let order = sort.topological_sort(["a", "b", "c"], |n| n, |n| n == "c" ? ["a", "b"] : [])
    assert_true(order.index("c") > order.index("a"))
    assert_true(order.index("c") > order.index("b"))
}

fn test_is_sorted_is_vacuously_true_for_short_lists() -> void {
    assert_true(sort.is_sorted([]))
    assert_true(sort.is_sorted([1]))
    assert_true(sort.is_sorted([1, 1, 2]))
    assert_false(sort.is_sorted([2, 1]))
}

fn test_binary_search_finds_and_reports_absence() -> void {
    let xs = [1, 3, 5, 7, 9]
    assert_eq(search.binary_search(xs, 1), 0)
    assert_eq(search.binary_search(xs, 9), 4)
    assert_eq(search.binary_search(xs, 5), 2)
    assert_true(search.binary_search(xs, 4) == null)
    assert_true(search.binary_search([], 1) == null)
}

fn test_bounds_bracket_a_run_of_equal_keys() -> void {
    let xs = [1, 2, 2, 2, 3]
    assert_eq(search.lower_bound(xs, 2), 1)
    assert_eq(search.upper_bound(xs, 2), 4)
    assert_eq(search.equal_range(xs, 2), (1, 4))
    # A key that is not there gives an empty range at the insertion point.
    assert_eq(search.lower_bound(xs, 0), 0)
    assert_eq(search.upper_bound(xs, 9), 5)
    assert_eq(search.equal_range(xs, 4), (5, 5))
}

fn test_binary_search_by_finds_the_partition_point() -> void {
    assert_eq(search.binary_search_by([1, 2, 3, 4], |n| n >= 3), 2)
    assert_eq(search.binary_search_by([1, 2], |n| false), 2)
    assert_eq(search.binary_search_by([1, 2], |n| true), 0)
}

fn test_exponential_and_interpolation_search_agree_with_binary() -> void {
    let xs = [1, 3, 5, 7, 9, 11]
    assert_eq(search.exponential_search(xs, 11), 5)
    assert_true(search.exponential_search(xs, 2) == null)
    assert_eq(search.interpolation_search(xs, 7), 3)
    assert_true(search.interpolation_search(xs, 8) == null)
    assert_true(search.interpolation_search([], 1) == null)
}

fn test_linear_search_and_find_all() -> void {
    assert_eq(search.linear_search([1, 2, 1], 1), 0)
    assert_true(search.linear_search([1], 2) == null)
    assert_eq(search.find_all([1, 2, 1, 1], 1), [0, 2, 3])
    assert_eq(search.find_all([], 1), [])
}

fn test_substring_searches_agree_and_handle_overlaps() -> void {
    let text = "aaaa"
    assert_eq(search.kmp_search(text, "aa"), [0, 1, 2])
    assert_eq(search.boyer_moore_search(text, "aa"), [0, 1, 2])
    assert_eq(search.rabin_karp_search(text, "aa"), [0, 1, 2])
    assert_eq(search.kmp_search("abc", "d"), [])
    assert_eq(search.kmp_search("", "a"), [])
}

fn test_z_algorithm_reports_prefix_matches() -> void {
    assert_eq(search.z_algorithm("aaaa"), [4, 3, 2, 1])
    assert_eq(search.z_algorithm(""), [])
}

fn test_aho_corasick_finds_every_pattern_at_once() -> void {
    let hits = search.aho_corasick("ahishers", ["he", "she", "hers", "his"])
    assert_true(hits.len() == 4)
    assert_true((1, 3) in hits)  # "his" at index 1
}
