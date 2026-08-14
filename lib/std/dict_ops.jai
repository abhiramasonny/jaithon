#: Algorithms over dicts that the built-in dict methods do not cover.
#:
#: Every function here returns a new dict unless its name says otherwise.

#: A view over a dict that materialises missing entries on first read.
pub class WithDefault[K, V] {
    let entries: dict[K, V]
    let factory: fn(K) -> V

    #: Wrap `entries`, filling misses with `factory(key)`.
    pub fn init(self, entries: dict[K, V], factory: fn(K) -> V) {
        self.entries = entries
        self.factory = factory
    }

    #: The value under `key`, creating and storing it when absent.
    pub fn get(self, key: K) -> V {
        if key not in self.entries { self.entries[key] = self.factory(key) }
        return self.entries[key]
    }

    #: The value under `key`, or null — a read that never inserts.
    pub fn peek(self, key: K) -> V? {
        return self.entries.get(key, null)
    }

    #: The number of entries that have actually been materialised.
    pub fn len(self) -> int {
        return self.entries.len()
    }

    #: The wrapped dict itself, not a copy.
    pub fn to_dict(self) -> dict[K, V] {
        return self.entries
    }

    fn __getitem__(self, key: K) -> V {
        return self.get(key)
    }

    fn __setitem__(self, key: K, value: V) -> void {
        self.entries[key] = value
    }

    fn __contains__(self, key: K) -> bool {
        return key in self.entries
    }

    fn __len__(self) -> int {
        return self.entries.len()
    }

    fn __str__(self) -> str {
        return f"WithDefault({self.entries})"
    }
}

#: Swap keys and values.
#:
#: When two keys share a value the last one visited wins; use `group_values`
#: when that would lose information.
#:
#: ```
#: invert({"a": 1, "b": 2}) == {1: "a", 2: "b"}    # true
#: ```
pub fn invert[K, V](entries: dict[K, V]) -> dict[V, K] {
    let flipped: dict[V, K] = {}
    for (key, value) in entries.items() { flipped[value] = key }
    return flipped
}

#: Invert `entries`, collecting every key that shares a value into a list.
#:
#: ```
#: group_values({"a": 1, "b": 1}) == {1: ["a", "b"]}    # true
#: ```
pub fn group_values[K, V](entries: dict[K, V]) -> dict[V, list[K]] {
    let groups: dict[V, list[K]] = {}
    for (key, value) in entries.items() {
        if value not in groups { groups[value] = [] }
        groups[value].push(key)
    }
    return groups
}

#: Merge `right` into `left`, calling `resolve(key, left_value, right_value)`
#: for every key they share.
#:
#: ```
#: merge_with({"a": 1}, {"a": 2}, |_key, x, y| x + y) == {"a": 3}    # true
#: ```
pub fn merge_with[K, V](
    left: dict[K, V],
    right: dict[K, V],
    resolve: fn(K, V, V) -> V
) -> dict[K, V] {
    let merged: dict[K, V] = {}
    for (key, value) in left.items() { merged[key] = value }
    for (key, value) in right.items() {
        if key in merged {
            merged[key] = resolve(key, merged[key], value)
        } else {
            merged[key] = value
        }
    }
    return merged
}

#: Merge `right` into `left`, recursing wherever both sides hold a dict.
#:
#: Anywhere the two disagree on shape, the value from `right` replaces the one
#: from `left` outright.
#:
#: ```
#: deep_merge({"a": {"x": 1}}, {"a": {"y": 2}}) == {"a": {"x": 1, "y": 2}}    # true
#: ```
pub fn deep_merge(left: dict[any, any], right: dict[any, any]) -> dict[any, any] {
    let merged: dict[any, any] = {}
    for (key, value) in left.items() { merged[key] = value }
    for (key, value) in right.items() {
        let existing = merged.get(key, null)
        if isinstance(existing, dict) and isinstance(value, dict) {
            merged[key] = deep_merge(existing, value)
        } else {
            merged[key] = value
        }
    }
    return merged
}

#: Keep only the entries whose key satisfies `pred`.
#:
#: ```
#: filter_keys({"a": 1, "bb": 2}, |key| key.len() == 1) == {"a": 1}    # true
#: ```
pub fn filter_keys[K, V](entries: dict[K, V], pred: fn(K) -> bool) -> dict[K, V] {
    return {key: value for (key, value) in entries.items() if pred(key)}
}

#: Keep only the entries whose value satisfies `pred`.
#:
#: ```
#: filter_values({"a": 1, "b": 2}, |value| value > 1) == {"b": 2}    # true
#: ```
pub fn filter_values[K, V](entries: dict[K, V], pred: fn(V) -> bool) -> dict[K, V] {
    return {key: value for (key, value) in entries.items() if pred(value)}
}

#: Rewrite every key with `transform`, keeping the values.
#:
#: Raises `ValueError` when two keys collide, because silently dropping an
#: entry is never what the caller meant.
#:
#: ```
#: map_keys({"a": 1}, |key| key.upper()) == {"A": 1}    # true
#: ```
pub fn map_keys[K, V, K2](entries: dict[K, V], transform: fn(K) -> K2) -> dict[K2, V] {
    let mapped: dict[K2, V] = {}
    for (key, value) in entries.items() {
        let new_key = transform(key)
        if new_key in mapped {
            throw ValueError(f"transform maps two keys onto {new_key}")
        }
        mapped[new_key] = value
    }
    return mapped
}

#: Rewrite every value with `transform`, keeping the keys.
#:
#: ```
#: map_values({"a": 1, "b": 2}, |value| value * 10) == {"a": 10, "b": 20}    # true
#: ```
pub fn map_values[K, V, V2](entries: dict[K, V], transform: fn(V) -> V2) -> dict[K, V2] {
    return {key: transform(value) for (key, value) in entries.items()}
}

#: The sub-dict holding only `keys`, skipping any that are absent.
#:
#: ```
#: pick({"a": 1, "b": 2}, ["a", "z"]) == {"a": 1}    # true
#: ```
pub fn pick[K, V](entries: dict[K, V], keys: list[K]) -> dict[K, V] {
    let chosen: dict[K, V] = {}
    for key in keys {
        if key in entries { chosen[key] = entries[key] }
    }
    return chosen
}

#: The sub-dict holding everything except `keys`.
#:
#: ```
#: omit({"a": 1, "b": 2}, ["a"]) == {"b": 2}    # true
#: ```
pub fn omit[K, V](entries: dict[K, V], keys: list[K]) -> dict[K, V] {
    let dropped = {key for key in keys}
    return {key: value for (key, value) in entries.items() if key not in dropped}
}

#: Build a dict from `(key, value)` pairs; later pairs win.
#:
#: ```
#: from_pairs([("a", 1), ("a", 2)]) == {"a": 2}    # true
#: ```
pub fn from_pairs[K, V](pairs: list[tuple[K, V]]) -> dict[K, V] {
    let entries: dict[K, V] = {}
    for (key, value) in pairs { entries[key] = value }
    return entries
}

#: Snapshot the entries as a list of `(key, value)` tuples.
#:
#: Unlike `items()`, the result is an ordinary list, so it survives mutation of
#: the dict and can be sorted, sliced and reordered.
#:
#: ```
#: from_pairs(to_pairs({"a": 1})) == {"a": 1}    # true
#: ```
pub fn to_pairs[K, V](entries: dict[K, V]) -> list[tuple[K, V]] {
    return [(key, value) for (key, value) in entries.items()]
}

#: The value stored under `key`, inserting `make()` first when it is absent.
#:
#: `make` is called only on a miss, so it is safe for it to be expensive or to
#: allocate a fresh container per key.
#:
#: ```
#: let index = {}
#: default_get(index, "a", fn() { return [] }).push(1)
#: index == {"a": [1]}    # true
#: ```
pub fn default_get[K, V](entries: dict[K, V], key: K, make: fn() -> V) -> V {
    if key not in entries { entries[key] = make() }
    return entries[key]
}

#: Add `amount` to the count stored under `key` and return the new total.
#:
#: A missing key counts as zero, so this turns any dict into a tally.
#:
#: ```
#: let counts = {}
#: increment(counts, "a")        # 1
#: increment(counts, "a", 4)     # 5
#: ```
pub fn increment[K](counts: dict[K, int], key: K, amount: int = 1) -> int {
    let total = counts.get(key, 0) + amount
    counts[key] = total
    return total
}

#: Wrap `entries` so that reading a missing key yields `default` and stores it.
#:
#: Writes through to `entries`; every missing key gets the *same* `default`
#: object, so for mutable defaults construct `WithDefault` with a factory instead.
#:
#: ```
#: let counts = with_default({}, 0)
#: counts["a"] = counts["a"] + 1
#: counts.to_dict() == {"a": 1}    # true
#: ```
pub fn with_default[K, V](entries: dict[K, V], default: V) -> WithDefault[K, V] {
    return WithDefault(entries, |_key| default)
}
