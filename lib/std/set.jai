#: Set algebra as free functions over any number of sets, plus multisets.
#:
#: The built-in `set` has `add`/`del`/`len`/`in` but no algebra; this module adds
#: it in n-ary form, so "intersect all of these" is one call, not a fold.
from std.list import power_set as list_power_set

#: An element-to-count mapping. Counts are always positive; absent means zero.
pub type Multiset[T] = dict[T, int]

#: Every element belonging to at least one of `sets`.
#:
#: ```
#: union([{1, 2}, {2, 3}]) == {1, 2, 3}    # true
#: ```
pub fn union[T](sets: list[set[T]]) -> set[T] {
    let result: set[T] = set()
    for part in sets {
        for item in part { result.add(item) }
    }
    return result
}

#: Every element belonging to all of `sets`.
#:
#: Raises `ValueError` on an empty argument list, for which the intersection is
#: not defined.
#:
#: ```
#: intersection([{1, 2, 3}, {2, 3}, {3}]) == {3}    # true
#: ```
pub fn intersection[T](sets: list[set[T]]) -> set[T] {
    if sets.len() == 0 { throw ValueError("intersection of no sets is undefined") }
    # Walking the smallest member keeps the cost proportional to it rather than
    # to the largest set.
    var smallest = sets[0]
    for part in sets {
        if part.len() < smallest.len() { smallest = part }
    }
    let result: set[T] = set()
    for item in smallest {
        var in_all = true
        for part in sets {
            if item not in part {
                in_all = false
                break
            }
        }
        if in_all { result.add(item) }
    }
    return result
}

#: Every element of `base` that appears in none of `others`.
#:
#: ```
#: difference({1, 2, 3}, [{2}, {3}]) == {1}    # true
#: ```
pub fn difference[T](base: set[T], others: list[set[T]]) -> set[T] {
    let result: set[T] = set()
    for item in base {
        var found = false
        for part in others {
            if item in part {
                found = true
                break
            }
        }
        if not found { result.add(item) }
    }
    return result
}

#: Every element belonging to an odd number of `sets`.
#:
#: For two sets this is the usual symmetric difference, and for more it is what
#: folding the two-set operation from the left produces.
#:
#: ```
#: symmetric_difference([{1, 2}, {2, 3}]) == {1, 3}    # true
#: ```
pub fn symmetric_difference[T](sets: list[set[T]]) -> set[T] {
    let parity: dict[T, int] = {}
    for part in sets {
        for item in part { parity[item] = parity.get(item, 0) + 1 }
    }
    let result: set[T] = set()
    for (item, count) in parity.items() {
        if count % 2 == 1 { result.add(item) }
    }
    return result
}

#: Report whether no two of `sets` share an element.
#:
#: Runs in one pass over every element rather than comparing all pairs.
#:
#: ```
#: is_disjoint([{1}, {2}, {3}])    # true
#: is_disjoint([{1}, {2}, {1}])    # false
#: ```
pub fn is_disjoint[T](sets: list[set[T]]) -> bool {
    let seen: set[T] = set()
    for part in sets {
        for item in part {
            if item in seen { return false }
            seen.add(item)
        }
    }
    return true
}

#: Report whether every element of `a` is also in `b`.
#:
#: ```
#: is_subset({1, 2}, {1, 2, 3})    # true
#: ```
pub fn is_subset[T](a: set[T], b: set[T]) -> bool {
    if a.len() > b.len() { return false }
    for item in a {
        if item not in b { return false }
    }
    return true
}

#: Report whether `a` contains every element of `b`.
#:
#: ```
#: is_superset({1, 2, 3}, {1, 2})    # true
#: ```
pub fn is_superset[T](a: set[T], b: set[T]) -> bool {
    return is_subset(b, a)
}

#: Every subset of `items`, from the empty set to `items` itself.
#:
#: ```
#: power_set({1, 2}).len()    # 4
#: ```
pub fn power_set[T](items: set[T]) -> list[set[T]] {
    let ordered = [item for item in items]
    return [{item for item in subset} for subset in list_power_set(ordered)]
}

#: Every `(left, right)` pair with `left` from `a` and `right` from `b`.
#:
#: ```
#: cartesian_product({1}, {"a"}) == {(1, "a")}    # true
#: ```
pub fn cartesian_product[T, U](a: set[T], b: set[U]) -> set[tuple[T, U]] {
    let pairs: set[tuple[T, U]] = set()
    for left in a {
        for right in b { pairs.add((left, right)) }
    }
    return pairs
}

#: The multiset union: every element keeps the larger of its two counts.
#:
#: ```
#: multiset_union({"a": 3}, {"a": 1, "b": 2}) == {"a": 3, "b": 2}    # true
#: ```
pub fn multiset_union[T](a: Multiset[T], b: Multiset[T]) -> Multiset[T] {
    let out: Multiset[T] = {}
    for (key, count) in a.items() { _put(out, key, count) }
    for (key, count) in b.items() { _put(out, key, max(count, out.get(key, 0))) }
    return out
}

#: The multiset sum: counts of shared elements add together.
#:
#: ```
#: multiset_sum({"a": 3}, {"a": 1}) == {"a": 4}    # true
#: ```
pub fn multiset_sum[T](a: Multiset[T], b: Multiset[T]) -> Multiset[T] {
    let out: Multiset[T] = {}
    for (key, count) in a.items() { _put(out, key, count) }
    for (key, count) in b.items() { _put(out, key, out.get(key, 0) + count) }
    return out
}

#: The multiset intersection: every element keeps the smaller of its two counts.
#:
#: ```
#: multiset_intersection({"a": 3, "b": 1}, {"a": 1}) == {"a": 1}    # true
#: ```
pub fn multiset_intersection[T](a: Multiset[T], b: Multiset[T]) -> Multiset[T] {
    let out: Multiset[T] = {}
    for (key, count) in a.items() { _put(out, key, min(count, b.get(key, 0))) }
    return out
}

#: The multiset difference `a - b`, with counts clamped at zero.
#:
#: ```
#: multiset_difference({"a": 3}, {"a": 1}) == {"a": 2}    # true
#: multiset_difference({"a": 1}, {"a": 5}) == {}          # true
#: ```
pub fn multiset_difference[T](a: Multiset[T], b: Multiset[T]) -> Multiset[T] {
    let out: Multiset[T] = {}
    for (key, count) in a.items() { _put(out, key, count - b.get(key, 0)) }
    return out
}

#: Report whether every element of `a` occurs in `b` at least as often.
#:
#: ```
#: multiset_is_subset({"a": 1}, {"a": 2})    # true
#: multiset_is_subset({"a": 3}, {"a": 2})    # false
#: ```
pub fn multiset_is_subset[T](a: Multiset[T], b: Multiset[T]) -> bool {
    for (key, count) in a.items() {
        if count > b.get(key, 0) { return false }
    }
    return true
}

#: Expand a multiset into a list, repeating each element by its count.
#:
#: ```
#: multiset_to_list({"a": 2}) == ["a", "a"]    # true
#: ```
pub fn multiset_to_list[T](counts: Multiset[T]) -> list[T] {
    let out: list[T] = []
    for (key, count) in counts.items() {
        if count < 0 {
            throw ValueError(f"count for {key} must be non-negative, got {count}")
        }
        for _repeat in 0..count { out.push(key) }
    }
    return out
}

#: Store `count` under `key`, dropping the entry when it is not positive.
fn _put[T](counts: Multiset[T], key: T, count: int) -> void {
    if count > 0 { counts[key] = count }
}
