#: Dictionary variants: `DefaultDict`, `OrderedDict`, `MultiMap`, and the
#: immutable, hashable `FrozenDict`.
#:
#: `OrderedDict` is a hash map paired with a doubly linked list of keys, so
#: insertion order and `move_to_end` cost O(1) rather than an O(n) rebuild.
#: All four are re-exported from `std.collections`.
from std.core import Iterable, Iterator
from std.collections.linked import DoublyLinkedList, DoublyNode

#: A dictionary that fills a missing key from a factory on first lookup.
pub class DefaultDict[K, V]: Iterable[K] {
    var entries: dict[K, V]
    let factory: fn() -> V

    #: Create an empty dictionary that fills missing keys from `factory`.
    #:
    #: `factory` is called once per missing key looked up through `get`. O(1).
    #:
    #: ```
    #: let groups = DefaultDict[str, list[int]](fn() -> list[int] { return [] })
    #: groups.get("odd").push(1)
    #: groups.get("odd").push(3)
    #: groups.get("odd")                   # [1, 3]
    #: ```
    pub fn init(self, factory: fn() -> V) {
        self.entries = {}
        self.factory = factory
    }

    #: Return the value for `key`, inserting a fresh default when absent. O(1).
    pub fn get(self, key: K) -> V {
        if key not in self.entries { self.entries[key] = self.factory() }
        return self.entries[key]
    }

    #: Return the value for `key`, or null, without inserting anything. O(1).
    pub fn peek(self, key: K) -> V? {
        if key not in self.entries { return null }
        return self.entries[key]
    }

    #: Store `value` under `key`. O(1).
    pub fn set(self, key: K, value: V) -> void {
        self.entries[key] = value
    }

    #: Remove `key` and return its value. O(1).
    #:
    #: Raises `KeyError` when `key` is absent.
    pub fn remove(self, key: K) -> V {
        if key not in self.entries { throw KeyError(f"key not found: {key}") }
        let value = self.entries[key]
        self.entries.remove(key)
        return value
    }

    #: Report whether `key` has a stored value. O(1).
    pub fn contains(self, key: K) -> bool {
        return key in self.entries
    }

    #: Number of stored keys. O(1).
    pub fn len(self) -> int {
        return self.entries.len()
    }

    #: The stored keys. O(n).
    pub fn keys(self) -> list[K] {
        return self.entries.keys()
    }

    #: The stored values. O(n).
    pub fn values(self) -> list[V] {
        return self.entries.values()
    }

    #: The stored key-value pairs. O(n).
    pub fn items(self) -> list[tuple[K, V]] {
        return self.entries.items()
    }

    #: Copy the entries into a plain dictionary. O(n).
    pub fn to_dict(self) -> dict[K, V] {
        return {key: value for (key, value) in self.entries.items()}
    }

    #: Discard every entry, keeping the factory. O(1).
    pub fn clear(self) -> void {
        self.entries = {}
    }

    #: Iterate the stored keys. O(n).
    pub fn iter(self) -> Iterator[K] {
        return self.entries.keys().iter()
    }

    #: Number of stored keys. O(1).
    fn __len__(self) -> int {
        return self.entries.len()
    }

    #: Return the value for `key`, inserting a fresh default when absent. O(1).
    fn __getitem__(self, key: K) -> V {
        return self.get(key)
    }

    #: Store `value` under `key`. O(1).
    fn __setitem__(self, key: K, value: V) -> void {
        self.entries[key] = value
    }

    #: Report whether `key` has a stored value, without inserting one. O(1).
    fn __contains__(self, key: K) -> bool {
        return key in self.entries
    }

    #: Two default dictionaries are equal when their entries are; the factories
    #: are not compared. O(n).
    fn __eq__(self, other: DefaultDict[K, V]) -> bool {
        return self.entries == other.entries
    }

    #: Render the stored entries. O(n).
    fn __str__(self) -> str {
        return f"DefaultDict({self.entries})"
    }
}

#: A dictionary that iterates in insertion order and can reorder in O(1).
pub class OrderedDict[K, V]: Iterable[K] {
    var entries: dict[K, V]
    var order: DoublyLinkedList[K]
    var nodes: dict[K, DoublyNode[K]]

    #: Create an empty dictionary that remembers insertion order. O(1).
    #:
    #: Re-assigning an existing key keeps its original position; remove and
    #: re-insert, or call `move_to_end`, to change it.
    #:
    #: ```
    #: let scores = OrderedDict[str, int]()
    #: scores.set("b", 2)
    #: scores.set("a", 1)
    #: scores.keys()                       # ["b", "a"]
    #: ```
    pub fn init(self) {
        self.entries = {}
        self.order = DoublyLinkedList[K]()
        self.nodes = {}
    }

    #: Build a dictionary from `pairs`, keeping their order. O(n).
    pub static fn from_pairs(pairs: list[tuple[K, V]]) -> OrderedDict[K, V] {
        let result = OrderedDict[K, V]()
        for (key, value) in pairs { result.set(key, value) }
        return result
    }

    #: Return the value for `key`, or null when absent. O(1).
    pub fn get(self, key: K) -> V? {
        if key not in self.entries { return null }
        return self.entries[key]
    }

    #: Store `value` under `key`, appending new keys at the end. O(1).
    pub fn set(self, key: K, value: V) -> void {
        if key not in self.entries { self.nodes[key] = self.order.push_back(key) }
        self.entries[key] = value
    }

    #: Remove `key` and return its value. O(1).
    #:
    #: Raises `KeyError` when `key` is absent.
    pub fn remove(self, key: K) -> V {
        if key not in self.entries { throw KeyError(f"key not found: {key}") }
        let value = self.entries[key]
        self.entries.remove(key)
        self.order.remove_node(self.nodes[key])
        self.nodes.remove(key)
        return value
    }

    #: Move `key` to the end of the order, or to the front when `last` is
    #: false. O(1).
    #:
    #: Raises `KeyError` when `key` is absent.
    #:
    #: ```
    #: let scores = OrderedDict[str, int].from_pairs([("a", 1), ("b", 2)])
    #: scores.move_to_end("a")
    #: scores.keys()                       # ["b", "a"]
    #: ```
    pub fn move_to_end(self, key: K, last: bool = true) -> void {
        if key not in self.nodes { throw KeyError(f"key not found: {key}") }
        let node = self.nodes[key]
        if last { self.order.move_to_back(node) } else { self.order.move_to_front(node) }
    }

    #: Remove and return the last pair, or the first when `last` is false.
    #: O(1).
    #:
    #: Raises `KeyError` when the dictionary is empty. This is what makes an
    #: `OrderedDict` usable as a stack or a queue of entries.
    pub fn popitem(self, last: bool = true) -> tuple[K, V] {
        var node = self.order.first_node()
        if last { node = self.order.last_node() }
        if node == null { throw KeyError("popitem from an empty OrderedDict") }
        let key = node.value
        return (key, self.remove(key))
    }

    #: Report whether `key` is present. O(1).
    pub fn contains(self, key: K) -> bool {
        return key in self.entries
    }

    #: Number of entries. O(1).
    pub fn len(self) -> int {
        return self.entries.len()
    }

    #: The keys in insertion order. O(n).
    pub fn keys(self) -> list[K] {
        return self.order.to_list()
    }

    #: The values in insertion order. O(n).
    pub fn values(self) -> list[V] {
        return [self.entries[key] for key in self.order.to_list()]
    }

    #: The pairs in insertion order. O(n).
    pub fn items(self) -> list[tuple[K, V]] {
        return [(key, self.entries[key]) for key in self.order.to_list()]
    }

    #: Discard every entry. O(n).
    pub fn clear(self) -> void {
        self.entries = {}
        self.order.clear()
        self.nodes = {}
    }

    #: Iterate the keys in insertion order. O(n).
    pub fn iter(self) -> Iterator[K] {
        return self.order.to_list().iter()
    }

    #: Number of entries. O(1).
    fn __len__(self) -> int {
        return self.entries.len()
    }

    #: Return the value for `key`. Raises `KeyError` when absent. O(1).
    fn __getitem__(self, key: K) -> V {
        if key not in self.entries { throw KeyError(f"key not found: {key}") }
        return self.entries[key]
    }

    #: Store `value` under `key`. O(1).
    fn __setitem__(self, key: K, value: V) -> void {
        self.set(key, value)
    }

    #: Report whether `key` is present. O(1).
    fn __contains__(self, key: K) -> bool {
        return key in self.entries
    }

    #: Two ordered dictionaries are equal when they hold the same pairs in the
    #: same order. O(n).
    fn __eq__(self, other: OrderedDict[K, V]) -> bool {
        if self.entries.len() != other.entries.len() { return false }
        return self.items() == other.items()
    }

    #: Render the pairs in insertion order. O(n).
    fn __str__(self) -> str {
        return f"OrderedDict({self.items()})"
    }
}

#: A dictionary from each key to a list of values, duplicates kept.
pub class MultiMap[K, V]: Iterable[K] {
    var entries: dict[K, list[V]]
    var pair_count: int

    #: Create an empty map from keys to lists of values. O(1).
    #:
    #: A key exists only while it has at least one value: removing the last
    #: value removes the key.
    #:
    #: ```
    #: let index = MultiMap[str, int]()
    #: index.add("a", 1)
    #: index.add("a", 2)
    #: index.get("a")                      # [1, 2]
    #: index.total()                       # 2
    #: ```
    pub fn init(self) {
        self.entries = {}
        self.pair_count = 0
    }

    #: Build a map from key-value pairs, keeping duplicates. O(n).
    pub static fn from_pairs(pairs: list[tuple[K, V]]) -> MultiMap[K, V] {
        let result = MultiMap[K, V]()
        for (key, value) in pairs { result.add(key, value) }
        return result
    }

    #: Append `value` to the values of `key`. Amortised O(1).
    pub fn add(self, key: K, value: V) -> void {
        if key not in self.entries { self.entries[key] = [] }
        self.entries[key].push(value)
        self.pair_count += 1
    }

    #: Append every element of `values` to the values of `key`. O(m).
    pub fn add_all(self, key: K, values: list[V]) -> void {
        for value in values { self.add(key, value) }
    }

    #: Copy the values stored under `key`, or an empty list. O(m).
    #:
    #: The copy is deliberate: mutating it cannot corrupt the pair count. Use
    #: `count_of` or `first` when you only need a cheap answer.
    pub fn get(self, key: K) -> list[V] {
        if key not in self.entries { return [] }
        return [value for value in self.entries[key]]
    }

    #: The first value stored under `key`, or null. O(1).
    pub fn first(self, key: K) -> V? {
        if key not in self.entries { return null }
        return self.entries[key][0]
    }

    #: Number of values stored under `key`. O(1).
    pub fn count_of(self, key: K) -> int {
        if key not in self.entries { return 0 }
        return self.entries[key].len()
    }

    #: Report whether `key` has at least one value. O(1).
    pub fn contains_key(self, key: K) -> bool {
        return key in self.entries
    }

    #: Report whether `value` is stored under `key`. O(m).
    pub fn contains_pair(self, key: K, value: V) -> bool {
        if key not in self.entries { return false }
        return value in self.entries[key]
    }

    #: Remove `key` and return everything stored under it. O(m).
    pub fn remove_key(self, key: K) -> list[V] {
        if key not in self.entries { return [] }
        let values = self.entries[key]
        self.pair_count -= values.len()
        self.entries.remove(key)
        return values
    }

    #: Remove the first `value` stored under `key`; report whether it was
    #: there. O(m).
    pub fn remove_pair(self, key: K, value: V) -> bool {
        if key not in self.entries { return false }
        let values = self.entries[key]
        var found = -1
        for (position, existing) in values.enumerate() {
            if existing == value {
                found = position
                break
            }
        }
        if found < 0 { return false }
        var cursor = found
        while cursor < values.len() - 1 {
            values[cursor] = values[cursor + 1]
            cursor += 1
        }
        values.pop()
        self.pair_count -= 1
        if values.len() == 0 { self.entries.remove(key) }
        return true
    }

    #: Number of keys. O(1).
    pub fn len(self) -> int {
        return self.entries.len()
    }

    #: Number of key-value pairs. O(1).
    pub fn total(self) -> int {
        return self.pair_count
    }

    #: The keys that have at least one value. O(k).
    pub fn keys(self) -> list[K] {
        return self.entries.keys()
    }

    #: Every value in the map, grouped by key. O(n).
    pub fn values(self) -> list[V] {
        var out: list[V] = []
        for (_key, values) in self.entries.items() {
            for value in values { out.push(value) }
        }
        return out
    }

    #: Every key-value pair, grouped by key. O(n).
    pub fn pairs(self) -> list[tuple[K, V]] {
        var out: list[tuple[K, V]] = []
        for (key, values) in self.entries.items() {
            for value in values { out.push((key, value)) }
        }
        return out
    }

    #: Discard every entry. O(1).
    pub fn clear(self) -> void {
        self.entries = {}
        self.pair_count = 0
    }

    #: Iterate the keys. O(k).
    pub fn iter(self) -> Iterator[K] {
        return self.entries.keys().iter()
    }

    #: Number of keys, not of pairs; use `total` for the pair count. O(1).
    fn __len__(self) -> int {
        return self.entries.len()
    }

    #: Copy the values stored under `key`, or an empty list. O(m).
    fn __getitem__(self, key: K) -> list[V] {
        return self.get(key)
    }

    #: Report whether `key` has at least one value. O(1).
    fn __contains__(self, key: K) -> bool {
        return key in self.entries
    }

    #: Two maps are equal when the same keys carry the same values in the same
    #: order. O(n).
    fn __eq__(self, other: MultiMap[K, V]) -> bool {
        return self.entries == other.entries
    }

    #: Render the keys with their value lists. O(n).
    fn __str__(self) -> str {
        return f"MultiMap({self.entries})"
    }
}

#: An immutable dictionary, and therefore one that can itself be hashed and
#: used as a key or a set member.
pub class FrozenDict[K, V]: Iterable[K] {
    var entries: dict[K, V]
    var hash_value: int
    var hash_ready: bool

    #: Create an immutable dictionary holding a copy of `source`. O(n).
    #:
    #: The copy means later changes to `source` cannot alter the frozen view,
    #: which is what lets a `FrozenDict` be used as a dictionary key or a set
    #: member.
    #:
    #: ```
    #: let point = FrozenDict[str, int]({"x": 1, "y": 2})
    #: point.get("x")                      # 1
    #: point == FrozenDict[str, int]({"y": 2, "x": 1})     # true
    #: ```
    pub fn init(self, source: dict[K, V]) {
        self.entries = {key: value for (key, value) in source.items()}
        self.hash_value = 0
        self.hash_ready = false
    }

    #: Build a frozen dictionary from `pairs`; later pairs win. O(n).
    pub static fn from_pairs(pairs: list[tuple[K, V]]) -> FrozenDict[K, V] {
        var built: dict[K, V] = {}
        for (key, value) in pairs { built[key] = value }
        return FrozenDict[K, V]._wrap(built)
    }

    #: Return the value for `key`, or null when absent. O(1).
    pub fn get(self, key: K) -> V? {
        if key not in self.entries { return null }
        return self.entries[key]
    }

    #: Report whether `key` is present. O(1).
    pub fn contains(self, key: K) -> bool {
        return key in self.entries
    }

    #: Number of entries. O(1).
    pub fn len(self) -> int {
        return self.entries.len()
    }

    #: The keys. O(n).
    pub fn keys(self) -> list[K] {
        return self.entries.keys()
    }

    #: The values. O(n).
    pub fn values(self) -> list[V] {
        return self.entries.values()
    }

    #: The key-value pairs. O(n).
    pub fn items(self) -> list[tuple[K, V]] {
        return self.entries.items()
    }

    #: Copy the entries into a mutable dictionary. O(n).
    pub fn to_dict(self) -> dict[K, V] {
        return self._copy_entries()
    }

    #: Return a new frozen dictionary with `key` set to `value`. O(n).
    pub fn with_entry(self, key: K, value: V) -> FrozenDict[K, V] {
        let next = self._copy_entries()
        next[key] = value
        return FrozenDict[K, V]._wrap(next)
    }

    #: Return a new frozen dictionary without `key`. O(n).
    #:
    #: An absent key is not an error; the result is an equal dictionary.
    pub fn without(self, key: K) -> FrozenDict[K, V] {
        let next = self._copy_entries()
        if key in next { next.remove(key) }
        return FrozenDict[K, V]._wrap(next)
    }

    #: Return a new frozen dictionary with `other`'s entries laid over this
    #: one's. O(n + m).
    pub fn merged(self, other: FrozenDict[K, V]) -> FrozenDict[K, V] {
        let next = self._copy_entries()
        for (key, value) in other.entries.items() { next[key] = value }
        return FrozenDict[K, V]._wrap(next)
    }

    #: Iterate the keys. O(n).
    pub fn iter(self) -> Iterator[K] {
        return self.entries.keys().iter()
    }

    #: Number of entries. O(1).
    fn __len__(self) -> int {
        return self.entries.len()
    }

    #: Return the value for `key`. Raises `KeyError` when absent. O(1).
    fn __getitem__(self, key: K) -> V {
        if key not in self.entries { throw KeyError(f"key not found: {key}") }
        return self.entries[key]
    }

    #: Report whether `key` is present. O(1).
    fn __contains__(self, key: K) -> bool {
        return key in self.entries
    }

    #: Two frozen dictionaries are equal when they hold the same entries,
    #: whatever order they were built in. O(n).
    fn __eq__(self, other: FrozenDict[K, V]) -> bool {
        return self.entries == other.entries
    }

    #: Hash the entries. O(n) on the first call, O(1) afterwards.
    #:
    #: The hash is computed lazily so that a dictionary holding unhashable
    #: values can still be built and compared; only hashing it fails.
    fn __hash__(self) -> int {
        if self.hash_ready { return self.hash_value }
        var combined = 0
        for (key, value) in self.entries.items() {
            combined = combined ^ (hash(key) *% 1000003 ^ hash(value))
        }
        self.hash_value = combined ^ self.entries.len() *% 31
        self.hash_ready = true
        return self.hash_value
    }

    #: Render the entries. O(n).
    fn __str__(self) -> str {
        return f"FrozenDict({self.entries})"
    }

    static fn _wrap(entries: dict[K, V]) -> FrozenDict[K, V] {
        let result = FrozenDict[K, V]({})
        result.entries = entries
        return result
    }

    fn _copy_entries(self) -> dict[K, V] {
        return {key: value for (key, value) in self.entries.items()}
    }
}
