#: `std.dict` — the reshaping helpers over `dict[K, V]`, plus the built-in
#: dict's own contract about insertion order and missing keys.
import std.dict as dict

from std.test import assert_eq, assert_true, assert_throws

fn test_builtin_dict_keeps_insertion_order() -> void {
    let entries = {"b": 1, "a": 2, "c": 3}
    assert_eq(entries.keys(), ["b", "a", "c"])
    entries["b"] = 9  # updating must not reorder
    assert_eq(entries.keys(), ["b", "a", "c"])
    entries.remove("a")
    entries["a"] = 4  # reinserting appends
    assert_eq(entries.keys(), ["b", "c", "a"])
}

fn test_missing_key_raises_but_get_does_not() -> void {
    let entries = {"a": 1}
    assert_throws(KeyError, || entries["missing"])
    assert_eq(entries.get("missing", 0), 0)
    assert_true(entries.get("missing") == null)
}

fn test_invert_swaps_keys_and_values() -> void {
    assert_eq(dict.invert({"a": 1, "b": 2}), {1: "a", 2: "b"})
    assert_eq(dict.invert({}), {})
}

fn test_invert_keeps_the_last_key_for_a_duplicated_value() -> void {
    let flipped = dict.invert({"a": 1, "b": 1})
    assert_eq(flipped.len(), 1)
    assert_eq(flipped[1], "b")
}

fn test_group_values_keeps_every_key_of_a_shared_value() -> void {
    let grouped = dict.group_values({"a": 1, "b": 1, "c": 2})
    assert_eq(grouped[1], ["a", "b"])
    assert_eq(grouped[2], ["c"])
}

fn test_merge_with_only_calls_the_combiner_on_collisions() -> void {
    var calls = 0
    fn combine(key: str, left: int, right: int) -> int {
        calls += 1
        return left + right
    }
    let merged = dict.merge_with({"a": 1, "b": 2}, {"b": 10, "c": 3}, combine)
    assert_eq(merged, {"a": 1, "b": 12, "c": 3})
    assert_eq(calls, 1)
}

fn test_deep_merge_descends_into_nested_dicts() -> void {
    let merged = dict.deep_merge({"x": {"a": 1, "b": 2}, "y": 1}, {"x": {"b": 3, "c": 4}, "z": 2})
    assert_eq(merged["x"], {"a": 1, "b": 3, "c": 4})
    assert_eq(merged["y"], 1)
    assert_eq(merged["z"], 2)
}

fn test_filter_and_map_leave_the_original_alone() -> void {
    let source = {"a": 1, "bb": 2, "ccc": 3}
    assert_eq(dict.filter_keys(source, |k| k.len() > 1), {"bb": 2, "ccc": 3})
    assert_eq(dict.filter_values(source, |v| v % 2 == 1), {"a": 1, "ccc": 3})
    assert_eq(dict.map_values(source, |v| v * 2), {"a": 2, "bb": 4, "ccc": 6})
    assert_eq(dict.map_keys(source, |k| k.len()), {1: 1, 2: 2, 3: 3})
    assert_eq(source, {"a": 1, "bb": 2, "ccc": 3})
}

fn test_pick_and_omit_are_complementary() -> void {
    let source = {"a": 1, "b": 2, "c": 3}
    assert_eq(dict.pick(source, ["a", "c"]), {"a": 1, "c": 3})
    assert_eq(dict.omit(source, ["a", "c"]), {"b": 2})
    assert_eq(dict.pick(source, ["nope"]), {})
    assert_eq(dict.omit(source, []), source)
}

fn test_pairs_round_trip() -> void {
    let source = {"a": 1, "b": 2}
    let pairs = dict.to_pairs(source)
    assert_eq(pairs, [("a", 1), ("b", 2)])
    assert_eq(dict.from_pairs(pairs), source)
    assert_eq(dict.from_pairs([]), {})
}

fn test_default_get_inserts_only_once() -> void {
    let cache: dict[str, list[int]] = {}
    var built = 0
    fn make() -> list[int] {
        built += 1
        return [1]
    }
    let first = dict.default_get(cache, "k", make)
    first.push(2)
    let second = dict.default_get(cache, "k", make)
    assert_eq(built, 1)
    assert_eq(second, [1, 2])
}

fn test_increment_creates_then_accumulates() -> void {
    let counts: dict[str, int] = {}
    assert_eq(dict.increment(counts, "a"), 1)
    assert_eq(dict.increment(counts, "a", 4), 5)
    assert_eq(dict.increment(counts, "a", -5), 0)
    assert_eq(counts["a"], 0)
}

fn test_with_default_never_raises() -> void {
    let counts = dict.with_default({"a": 1}, 0)
    assert_eq(counts.get("a"), 1)
    assert_eq(counts.get("absent"), 0)
}
