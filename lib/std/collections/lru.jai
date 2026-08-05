#: A capacity-bounded least-recently-used cache, re-exported from
#: `std.collections`.
from std.core import Iterable, Iterator
from std.collections.linked import DoublyLinkedList, DoublyNode

#: A cache of bounded size that evicts the least recently used entry.
#:
#: A hash map finds an entry and a doubly linked list in most-recently-used
#: order finds the eviction victim, so `get`, `put` and `remove` are all O(1).
pub class LRUCache[K, V]: Iterable[K] {
    var nodes: dict[K, DoublyNode[tuple[K, V]]]
    var order: DoublyLinkedList[tuple[K, V]]
    let limit: int
    var hit_count: int
    var miss_count: int

    #: Create a cache holding at most `capacity` entries. O(1).
    #:
    #: Raises `ValueError` when `capacity` is not positive.
    #:
    #: ```
    #: let cache = LRUCache[str, int](2)
    #: cache.put("a", 1)
    #: cache.put("b", 2)
    #: cache.get("a")                      # 1
    #: cache.put("c", 3)
    #: cache.contains("b")                 # false
    #: cache.keys()                        # ["c", "a"]
    #: ```
    pub fn init(self, capacity: int) {
        if capacity < 1 { throw ValueError(f"capacity must be positive, got {capacity}") }
        self.nodes = {}
        self.order = DoublyLinkedList[tuple[K, V]]()
        self.limit = capacity
        self.hit_count = 0
        self.miss_count = 0
    }

    #: Return the value for `key` and mark it most recently used. O(1).
    #:
    #: Returns null and counts a miss when `key` is absent.
    pub fn get(self, key: K) -> V? {
        if key not in self.nodes {
            self.miss_count += 1
            return null
        }
        self.hit_count += 1
        let node = self.nodes[key]
        self.order.move_to_front(node)
        let (_key, value) = node.value
        return value
    }

    #: Store `value` under `key` and return the entry it evicted, or null.
    #: O(1).
    #:
    #: The evicted pair is returned rather than dropped so that a write-back
    #: cache can flush it.
    pub fn put(self, key: K, value: V) -> tuple[K, V]? {
        if key in self.nodes {
            let node = self.nodes[key]
            node.value = (key, value)
            self.order.move_to_front(node)
            return null
        }
        self.nodes[key] = self.order.push_front((key, value))
        if self.order.len() <= self.limit { return null }
        let evicted = self.order.pop_back()
        let (evicted_key, _evicted_value) = evicted
        self.nodes.remove(evicted_key)
        return evicted
    }

    #: Return the value for `key` without reordering or counting a lookup.
    #: O(1).
    pub fn peek(self, key: K) -> V? {
        if key not in self.nodes { return null }
        let (_key, value) = self.nodes[key].value
        return value
    }

    #: Report whether `key` is cached, without reordering or counting a lookup.
    #: O(1).
    pub fn contains(self, key: K) -> bool {
        return key in self.nodes
    }

    #: Remove `key` and return its value, or null when it was absent. O(1).
    pub fn remove(self, key: K) -> V? {
        if key not in self.nodes { return null }
        let (_key, value) = self.order.remove_node(self.nodes[key])
        self.nodes.remove(key)
        return value
    }

    #: Number of cached entries. O(1).
    pub fn len(self) -> int {
        return self.nodes.len()
    }

    #: The greatest number of entries the cache will hold. O(1).
    pub fn capacity(self) -> int {
        return self.limit
    }

    #: Number of `get` calls that found their key. O(1).
    pub fn hits(self) -> int {
        return self.hit_count
    }

    #: Number of `get` calls that did not find their key. O(1).
    pub fn misses(self) -> int {
        return self.miss_count
    }

    #: The fraction of `get` calls that hit, or 0.0 when there were none. O(1).
    pub fn hit_rate(self) -> float {
        let lookups = self.hit_count + self.miss_count
        if lookups == 0 { return 0.0 }
        return float(self.hit_count) / float(lookups)
    }

    #: The cached keys, most recently used first. O(n).
    pub fn keys(self) -> list[K] {
        return [key for (key, _value) in self.order.to_list()]
    }

    #: The cached pairs, most recently used first. O(n).
    pub fn items(self) -> list[tuple[K, V]] {
        return self.order.to_list()
    }

    #: Discard every entry, leaving the statistics alone. O(n).
    #:
    #: Call `reset_stats` as well to start a fresh measurement.
    pub fn clear(self) -> void {
        self.nodes = {}
        self.order.clear()
    }

    #: Set the hit and miss counters back to zero. O(1).
    pub fn reset_stats(self) -> void {
        self.hit_count = 0
        self.miss_count = 0
    }

    #: Iterate the keys, most recently used first. O(n).
    pub fn iter(self) -> Iterator[K] {
        return self.keys().iter()
    }

    #: Number of cached entries. O(1).
    fn __len__(self) -> int {
        return self.nodes.len()
    }

    #: Report whether `key` is cached, without counting a lookup. O(1).
    fn __contains__(self, key: K) -> bool {
        return key in self.nodes
    }

    #: Two caches are equal when they have the same capacity and hold the same
    #: entries in the same recency order. Statistics are ignored. O(n).
    fn __eq__(self, other: LRUCache[K, V]) -> bool {
        if self.limit != other.limit { return false }
        if self.nodes.len() != other.nodes.len() { return false }
        return self.order.to_list() == other.order.to_list()
    }

    #: Render the entries, most recently used first. O(n).
    fn __str__(self) -> str {
        return f"LRUCache({self.order.to_list()})"
    }
}
