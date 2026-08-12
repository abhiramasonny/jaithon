#: General-purpose collections: the `Counter` and `Bag` multisets, plus every
#: other collection in the package re-exported.
#:
#: Submodules — `heap`, `linked`, `ring`, `dicts`, `lru` — stay split so a
#: program wanting only a heap doesn't compile the rest, and no file holds more
#: than one family of structures.
from std.core import Iterable, Iterator
from std.collections.heap import Heap, IndexedHeap, merge, nlargest, nsmallest
from std.collections.linked import DoublyLinkedList, DoublyNode
from std.collections.linked import SinglyLinkedList, SinglyNode
from std.collections.ring import CircularBuffer, Deque
from std.collections.dicts import DefaultDict, FrozenDict, MultiMap, OrderedDict
from std.collections.lru import LRUCache

export { Heap, IndexedHeap, merge, nlargest, nsmallest }
export { DoublyLinkedList, DoublyNode, SinglyLinkedList, SinglyNode }
export { CircularBuffer, Deque, DefaultDict, FrozenDict, MultiMap, OrderedDict, LRUCache }

#: A multiset over a dictionary, mapping each key to an integer count.
pub class Counter[T]: Iterable[T] {
    var counts: dict[T, int]

    #: Create an empty counter. O(1).
    #:
    #: Counts may be negative (that's what makes `subtract` total); a zero count
    #: is never stored, so `len`/`keys`/`in` stay consistent. Use `Bag` instead
    #: when counts are multiplicities rather than measurements.
    pub fn init(self) {
        self.counts = {}
    }

    #: Count the occurrences of each element of `items`. O(n).
    #:
    #: ```
    #: let counts = Counter[str].from_list(["a", "b", "a", "c", "a", "b"])
    #: counts.get("a")                     # 3
    #: counts.most_common(2)               # [("a", 3), ("b", 2)]
    #: ```
    pub static fn from_list(items: list[T]) -> Counter[T] {
        let result = Counter[T]()
        for item in items { result.add(item) }
        return result
    }

    #: Build a counter from explicit counts, dropping the zeroes. O(n).
    pub static fn from_dict(counts: dict[T, int]) -> Counter[T] {
        let result = Counter[T]()
        for (key, count) in counts.items() { result.set_count(key, count) }
        return result
    }

    #: The count recorded for `key`, or zero. O(1).
    pub fn get(self, key: T) -> int {
        return self.counts.get(key, 0)
    }

    #: Add `amount` to the count for `key`. O(1).
    pub fn add(self, key: T, amount: int = 1) -> void {
        self.set_count(key, self.get(key) + amount)
    }

    #: Set the count for `key`, dropping the entry when it reaches zero. O(1).
    pub fn set_count(self, key: T, count: int) -> void {
        if count == 0 {
            if key in self.counts { self.counts.remove(key) }
        } else {
            self.counts[key] = count
        }
    }

    #: Drop `key` and return the count it had, or zero. O(1).
    pub fn remove(self, key: T) -> int {
        let count = self.get(key)
        if key in self.counts { self.counts.remove(key) }
        return count
    }

    #: The sum of every count, which may be negative. O(k) for k distinct keys.
    pub fn total(self) -> int {
        var sum = 0
        for count in self.counts.values() { sum += count }
        return sum
    }

    #: The `limit` highest-counted pairs, highest first; all of them when
    #: `limit` is null. O(k + limit·log k).
    #:
    #: Ties are broken arbitrarily. A heap is used rather than a sort because
    #: the interesting case is a small `limit` over many keys.
    pub fn most_common(self, limit: int? = null) -> list[tuple[T, int]] {
        var take = limit ?? self.counts.len()
        if take > self.counts.len() { take = self.counts.len() }
        if take <= 0 { return [] }
        let ranked = Heap[tuple[T, int]].from_list(self.counts.items(), |a, b| a[1] > b[1])
        var out: list[tuple[T, int]] = []
        for _position in 0..take { out.push(ranked.pop()) }
        return out
    }

    #: Add every count in `other` to this counter. O(k of other).
    pub fn update(self, other: Counter[T]) -> void {
        for (key, count) in other.counts.items() { self.add(key, count) }
    }

    #: Subtract every count in `other` from this counter. O(k of other).
    #:
    #: Counts are allowed to go negative; that is the difference between a
    #: `Counter` and a `Bag`.
    #:
    #: ```
    #: let counts = Counter[str].from_list(["a", "a", "b"])
    #: counts.subtract(Counter[str].from_list(["a", "c"]))
    #: counts.get("a")                     # 1
    #: counts.get("c")                     # -1
    #: ```
    pub fn subtract(self, other: Counter[T]) -> void {
        for (key, count) in other.counts.items() { self.add(key, -count) }
    }

    #: Return a counter holding the greater of each pair of counts.
    #: O(k1 + k2).
    #:
    #: ```
    #: let left = Counter[str].from_list(["a", "a", "b"])
    #: let right = Counter[str].from_list(["a", "c"])
    #: left.union(right).get("a")          # 2
    #: left.intersection(right).get("a")   # 1
    #: ```
    pub fn union(self, other: Counter[T]) -> Counter[T] {
        let result = Counter[T]()
        for (key, count) in self.counts.items() { result.set_count(key, count) }
        for (key, count) in other.counts.items() {
            if count > result.get(key) { result.set_count(key, count) }
        }
        return result
    }

    #: Return a counter holding the lesser of each pair of counts, for keys in
    #: both. O(k1 + k2).
    pub fn intersection(self, other: Counter[T]) -> Counter[T] {
        let result = Counter[T]()
        for (key, count) in self.counts.items() {
            if key not in other.counts { continue }
            let paired = other.counts[key]
            result.set_count(key, if count < paired { count } else { paired })
        }
        return result
    }

    #: Expand into a list repeating each key by its count. O(total).
    #:
    #: Keys with a negative count are skipped, since they cannot be repeated a
    #: negative number of times.
    pub fn elements(self) -> list[T] {
        var out: list[T] = []
        for (key, count) in self.counts.items() {
            if count < 0 { continue }
            for _repeat in 0..count { out.push(key) }
        }
        return out
    }

    #: Number of distinct keys with a non-zero count. O(1).
    pub fn len(self) -> int {
        return self.counts.len()
    }

    #: The keys with a non-zero count. O(k).
    pub fn keys(self) -> list[T] {
        return self.counts.keys()
    }

    #: The key-count pairs, in no particular order. O(k).
    pub fn items(self) -> list[tuple[T, int]] {
        return self.counts.items()
    }

    #: Discard every count. O(1).
    pub fn clear(self) -> void {
        self.counts = {}
    }

    #: Iterate the keys with a non-zero count. O(k).
    pub fn iter(self) -> Iterator[T] {
        return self.counts.keys().iter()
    }

    #: Number of distinct keys with a non-zero count. O(1).
    fn __len__(self) -> int {
        return self.counts.len()
    }

    #: The count recorded for `key`, or zero. O(1).
    fn __getitem__(self, key: T) -> int {
        return self.counts.get(key, 0)
    }

    #: Set the count for `key`. O(1).
    fn __setitem__(self, key: T, count: int) -> void {
        self.set_count(key, count)
    }

    #: Report whether `key` has a non-zero count. O(1).
    fn __contains__(self, key: T) -> bool {
        return key in self.counts
    }

    #: Two counters are equal when every key has the same count. O(k).
    fn __eq__(self, other: Counter[T]) -> bool {
        return self.counts == other.counts
    }

    #: Render the key-count pairs. O(k).
    fn __str__(self) -> str {
        return f"Counter({self.counts})"
    }
}

#: An unordered multiset in which every element has a positive multiplicity.
pub class Bag[T]: Iterable[T] {
    var counts: dict[T, int]
    var size: int

    #: Create an empty bag. O(1).
    #:
    #: `len` is the total number of elements rather than the number of distinct
    #: ones, and iteration yields each element as many times as it occurs.
    #:
    #: ```
    #: let bag = Bag[str].from_list(["a", "b", "a"])
    #: bag.len()                           # 3
    #: bag.distinct_count()                # 2
    #: bag.count_of("a")                   # 2
    #: ```
    pub fn init(self) {
        self.counts = {}
        self.size = 0
    }

    #: Build a bag holding every element of `items`, duplicates included. O(n).
    pub static fn from_list(items: list[T]) -> Bag[T] {
        let result = Bag[T]()
        for item in items { result.add(item) }
        return result
    }

    #: Add `count` copies of `value`. O(1).
    #:
    #: Raises `ValueError` when `count` is negative; use `remove` or `discard`
    #: to take elements out.
    pub fn add(self, value: T, count: int = 1) -> void {
        if count < 0 { throw ValueError(f"count must not be negative, got {count}") }
        if count == 0 { return }
        self.counts[value] = self.count_of(value) + count
        self.size += count
    }

    #: Remove `count` copies of `value`. O(1).
    #:
    #: Raises `KeyError` when the bag holds fewer than `count` copies, so that
    #: a bag can never reach a negative multiplicity — use `discard` when a
    #: shortfall is ordinary — and `ValueError` when `count` is negative.
    pub fn remove(self, value: T, count: int = 1) -> void {
        if count < 0 { throw ValueError(f"count must not be negative, got {count}") }
        let held = self.count_of(value)
        if held < count {
            throw KeyError(f"bag holds {held} copies of {value}, cannot remove {count}")
        }
        self._take(value, count)
    }

    #: Remove up to `count` copies of `value` and return how many went. O(1).
    pub fn discard(self, value: T, count: int = 1) -> int {
        if count < 0 { throw ValueError(f"count must not be negative, got {count}") }
        let held = self.count_of(value)
        let taken = if held < count { held } else { count }
        self._take(value, taken)
        return taken
    }

    #: The number of copies of `value` held. O(1).
    pub fn count_of(self, value: T) -> int {
        return self.counts.get(value, 0)
    }

    #: Total number of elements, counting duplicates. O(1).
    pub fn len(self) -> int {
        return self.size
    }

    #: Number of distinct elements. O(1).
    pub fn distinct_count(self) -> int {
        return self.counts.len()
    }

    #: The distinct elements, in no particular order. O(k).
    pub fn distinct(self) -> list[T] {
        return self.counts.keys()
    }

    #: Report whether the bag holds no elements. O(1).
    pub fn is_empty(self) -> bool {
        return self.size == 0
    }

    #: Return a bag holding the greater multiplicity of each element.
    #: O(k1 + k2).
    pub fn union(self, other: Bag[T]) -> Bag[T] {
        let result = Bag[T]()
        for (value, count) in self.counts.items() { result.add(value, count) }
        for (value, count) in other.counts.items() {
            let shortfall = count - result.count_of(value)
            if shortfall > 0 { result.add(value, shortfall) }
        }
        return result
    }

    #: Return a bag holding the lesser multiplicity of each element.
    #: O(k1 + k2).
    pub fn intersection(self, other: Bag[T]) -> Bag[T] {
        let result = Bag[T]()
        for (value, count) in self.counts.items() {
            let paired = other.count_of(value)
            let shared = if count < paired { count } else { paired }
            result.add(value, shared)
        }
        return result
    }

    #: Return a bag with `other`'s multiplicities taken away, floored at zero.
    #: O(k1 + k2).
    #:
    #: ```
    #: let left = Bag[str].from_list(["a", "a", "b"])
    #: let right = Bag[str].from_list(["a", "c"])
    #: left.difference(right).count_of("a")        # 1
    #: left.difference(right).count_of("c")        # 0
    #: ```
    pub fn difference(self, other: Bag[T]) -> Bag[T] {
        let result = Bag[T]()
        for (value, count) in self.counts.items() {
            let remaining = count - other.count_of(value)
            if remaining > 0 { result.add(value, remaining) }
        }
        return result
    }

    #: Return a bag holding the sum of the two multiplicities of each element.
    #: O(k1 + k2).
    pub fn combined(self, other: Bag[T]) -> Bag[T] {
        let result = Bag[T]()
        for (value, count) in self.counts.items() { result.add(value, count) }
        for (value, count) in other.counts.items() { result.add(value, count) }
        return result
    }

    #: Report whether every element occurs in `other` at least as often.
    #: O(k).
    pub fn is_subset(self, other: Bag[T]) -> bool {
        for (value, count) in self.counts.items() {
            if other.count_of(value) < count { return false }
        }
        return true
    }

    #: Copy the elements into a list, each repeated by its multiplicity. O(n).
    pub fn to_list(self) -> list[T] {
        var out: list[T] = []
        for (value, count) in self.counts.items() {
            for _repeat in 0..count { out.push(value) }
        }
        return out
    }

    #: Discard every element. O(1).
    pub fn clear(self) -> void {
        self.counts = {}
        self.size = 0
    }

    #: Iterate the elements, each repeated by its multiplicity. O(n), and
    #: allocates a snapshot.
    pub fn iter(self) -> Iterator[T] {
        return self.to_list().iter()
    }

    #: Total number of elements, counting duplicates. O(1).
    fn __len__(self) -> int {
        return self.size
    }

    #: Report whether the bag holds at least one `value`. O(1).
    fn __contains__(self, value: T) -> bool {
        return value in self.counts
    }

    #: Two bags are equal when every element occurs equally often in both.
    #: O(k).
    fn __eq__(self, other: Bag[T]) -> bool {
        return self.counts == other.counts
    }

    #: Render the elements with their multiplicities. O(k).
    fn __str__(self) -> str {
        return f"Bag({self.counts})"
    }

    fn _take(self, value: T, count: int) -> void {
        if count == 0 { return }
        let remaining = self.counts[value] - count
        if remaining == 0 {
            self.counts.remove(value)
        } else {
            self.counts[value] = remaining
        }
        self.size -= count
    }
}
