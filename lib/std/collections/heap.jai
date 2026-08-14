#: Binary heaps: a comparator-parameterised `Heap`, and an `IndexedHeap` whose
#: `decrease_key` runs in O(log n).
#:
#: `IndexedHeap` maintains a key-to-position map alongside the heap array. That
#: is what Dijkstra's algorithm and A* actually need: without it, lowering the
#: priority of a key already in the frontier costs an O(n) scan.
from std.core import Iterable, Iterator

#: A binary heap ordered by a caller-supplied comparator.
#:
#: The heap is not stable: equal elements leave in an unspecified order.
pub class Heap[T]: Iterable[T] {
    var items: list[T]
    let less: fn(T, T) -> bool

    #: Create an empty heap ordered by `less`.
    #:
    #: `less(a, b)` must be true when `a` leaves the heap before `b`. The
    #: default compares with `<`, giving a min-heap, so `T` must support `<`
    #: unless a comparator is supplied. O(1).
    #:
    #: ```
    #: let heap = Heap[int]()
    #: heap.push(3)
    #: heap.push(1)
    #: heap.peek()                    # 1
    #: let tallest = Heap[int](|a, b| a > b)
    #: tallest.push(3)
    #: tallest.push(9)
    #: tallest.peek()                 # 9
    #: ```
    pub fn init(self, less: fn(T, T) -> bool = |a, b| a < b) {
        self.items = []
        self.less = less
    }

    #: Build a heap containing a copy of `items` in O(n).
    #:
    #: Sifting down from the last internal node is linear; pushing one at a
    #: time would be O(n log n).
    #:
    #: ```
    #: let heap = Heap[int].from_list([5, 3, 8, 1])
    #: heap.pop()                     # 1
    #: ```
    pub static fn from_list(items: list[T], less: fn(T, T) -> bool = |a, b| a < b) -> Heap[T] {
        let heap = Heap[T](less)
        heap.items = [item for item in items]
        var index = heap.items.len() // 2 - 1
        while index >= 0 {
            heap._sift_down(index)
            index -= 1
        }
        return heap
    }

    #: Number of elements in the heap. O(1).
    pub fn len(self) -> int {
        return self.items.len()
    }

    #: Report whether the heap holds no elements. O(1).
    pub fn is_empty(self) -> bool {
        return self.items.len() == 0
    }

    #: Insert `value`. Amortised O(log n).
    pub fn push(self, value: T) -> void {
        self.items.push(value)
        self._sift_up(self.items.len() - 1)
    }

    #: Remove and return the first element in comparator order. O(log n).
    #:
    #: Raises `IndexError` when the heap is empty.
    pub fn pop(self) -> T {
        if self.items.len() == 0 { throw IndexError("pop from an empty Heap") }
        let top = self.items[0]
        let last = self.items.pop()
        if self.items.len() > 0 {
            self.items[0] = last
            self._sift_down(0)
        }
        return top
    }

    #: The first element in comparator order without removing it, or null. O(1).
    pub fn peek(self) -> T? {
        if self.items.len() == 0 { return null }
        return self.items[0]
    }

    #: Push `value` and pop the first element in one sift. O(log n).
    #:
    #: Cheaper than `push` followed by `pop`, and it never grows the backing
    #: list. When `value` would come out first anyway it is returned untouched
    #: and the heap is unchanged, which is what makes the bounded-size heap in
    #: `nlargest` work.
    #:
    #: ```
    #: let heap = Heap[int].from_list([2, 4, 6])
    #: heap.push_pop(1)               # 1
    #: heap.push_pop(5)               # 2
    #: heap.peek()                    # 4
    #: ```
    pub fn push_pop(self, value: T) -> T {
        if self.items.len() == 0 { return value }
        let top = self.items[0]
        if self.less(value, top) { return value }
        self.items[0] = value
        self._sift_down(0)
        return top
    }

    #: Pop the first element and push `value` in one sift. O(log n).
    #:
    #: Unlike `push_pop` the heap always ends up holding `value`. Raises
    #: `IndexError` when the heap is empty.
    pub fn replace(self, value: T) -> T {
        if self.items.len() == 0 { throw IndexError("replace on an empty Heap") }
        let top = self.items[0]
        self.items[0] = value
        self._sift_down(0)
        return top
    }

    #: Insert every element of `values`. O(m log(n + m)).
    pub fn extend(self, values: list[T]) -> void {
        for value in values { self.push(value) }
    }

    #: Empty the heap into a list in comparator order. O(n log n).
    #:
    #: The heap is empty afterwards; use `sorted` to keep it.
    pub fn drain(self) -> list[T] {
        var out: list[T] = []
        while self.items.len() > 0 { out.push(self.pop()) }
        return out
    }

    #: Return the elements in comparator order, leaving the heap intact.
    #: O(n log n).
    #:
    #: ```
    #: Heap[int].from_list([3, 1, 2]).sorted()      # [1, 2, 3]
    #: ```
    pub fn sorted(self) -> list[T] {
        return self.copy().drain()
    }

    #: Return an independent heap with the same elements and comparator. O(n).
    pub fn copy(self) -> Heap[T] {
        let clone = Heap[T](self.less)
        clone.items = [item for item in self.items]
        return clone
    }

    #: Discard every element. O(1).
    pub fn clear(self) -> void {
        self.items = []
    }

    #: Iterate in comparator order. O(n log n), and allocates a copy.
    pub fn iter(self) -> Iterator[T] {
        return self.sorted().iter()
    }

    #: Number of elements in the heap. O(1).
    fn __len__(self) -> int {
        return self.items.len()
    }

    #: Two heaps are equal when they yield equal sequences under the receiver's
    #: comparator. O(n log n).
    fn __eq__(self, other: Heap[T]) -> bool {
        if self.items.len() != other.items.len() { return false }
        return self.sorted() == other.sorted()
    }

    #: Render the elements in comparator order. O(n log n).
    fn __str__(self) -> str {
        return f"Heap({self.sorted()})"
    }

    fn _sift_up(self, from_index: int) -> void {
        var index = from_index
        while index > 0 {
            let parent = (index - 1) // 2
            if not self.less(self.items[index], self.items[parent]) { break }
            self._swap(index, parent)
            index = parent
        }
    }

    fn _sift_down(self, from_index: int) -> void {
        let count = self.items.len()
        var index = from_index
        loop {
            let left = index * 2 + 1
            if left >= count { break }
            var best = left
            let right = left + 1
            if right < count and self.less(self.items[right], self.items[left]) {
                best = right
            }
            if not self.less(self.items[best], self.items[index]) { break }
            self._swap(index, best)
            index = best
        }
    }

    fn _swap(self, a: int, b: int) -> void {
        let temp = self.items[a]
        self.items[a] = self.items[b]
        self.items[b] = temp
    }
}

#: A binary heap of unique keys carrying priorities, supporting decrease-key.
#:
#: A key-to-position map is kept alongside the heap array, so reaching an
#: element by key costs O(1) and updating its priority O(log n), rather than
#: the O(n) scan a plain heap would need.
pub class IndexedHeap[K, P]: Iterable[K] {
    var keys: list[K]
    var priorities: list[P]
    var index_of: dict[K, int]
    let less: fn(P, P) -> bool

    #: Create an empty indexed heap ordered by `less` over priorities.
    #:
    #: Each key appears at most once. The default comparator is `<`, so the
    #: smallest priority comes out first. O(1).
    #:
    #: ```
    #: let frontier = IndexedHeap[str, int]()
    #: frontier.push("a", 5)
    #: frontier.push_or_decrease("a", 2)     # true
    #: frontier.pop()                        # ("a", 2)
    #: ```
    pub fn init(self, less: fn(P, P) -> bool = |a, b| a < b) {
        self.keys = []
        self.priorities = []
        self.index_of = {}
        self.less = less
    }

    #: Number of keys in the heap. O(1).
    pub fn len(self) -> int {
        return self.keys.len()
    }

    #: Report whether the heap holds no keys. O(1).
    pub fn is_empty(self) -> bool {
        return self.keys.len() == 0
    }

    #: Report whether `key` is in the heap. O(1).
    pub fn contains(self, key: K) -> bool {
        return key in self.index_of
    }

    #: The current priority of `key`, or null when it is absent. O(1).
    pub fn priority_of(self, key: K) -> P? {
        if key not in self.index_of { return null }
        return self.priorities[self.index_of[key]]
    }

    #: Insert `key` with `priority`. Amortised O(log n).
    #:
    #: Raises `ValueError` when `key` is already present; use `set_priority` or
    #: `push_or_decrease` to update an existing key.
    pub fn push(self, key: K, priority: P) -> void {
        if key in self.index_of { throw ValueError(f"key already in the heap: {key}") }
        self.keys.push(key)
        self.priorities.push(priority)
        self.index_of[key] = self.keys.len() - 1
        self._sift_up(self.keys.len() - 1)
    }

    #: Lower the priority of `key` to `priority`. O(log n).
    #:
    #: Raises `KeyError` when `key` is absent and `ValueError` when `priority`
    #: does not strictly precede the current one, since sifting up would then
    #: leave the heap invariant broken.
    pub fn decrease_key(self, key: K, priority: P) -> void {
        if key not in self.index_of { throw KeyError(f"key not in the heap: {key}") }
        let index = self.index_of[key]
        if not self.less(priority, self.priorities[index]) {
            throw ValueError(
                f"priority {priority} does not precede the current {self.priorities[index]}"
            )
        }
        self.priorities[index] = priority
        self._sift_up(index)
    }

    #: Insert `key`, or lower its priority if that is an improvement. O(log n).
    #:
    #: Returns true when the heap changed. This is the relaxation step of
    #: Dijkstra's algorithm written once.
    pub fn push_or_decrease(self, key: K, priority: P) -> bool {
        if key not in self.index_of {
            self.push(key, priority)
            return true
        }
        let index = self.index_of[key]
        if not self.less(priority, self.priorities[index]) { return false }
        self.priorities[index] = priority
        self._sift_up(index)
        return true
    }

    #: Set the priority of `key`, moving it in either direction. O(log n).
    #:
    #: Inserts `key` when it is absent.
    pub fn set_priority(self, key: K, priority: P) -> void {
        if key not in self.index_of {
            self.push(key, priority)
            return
        }
        let index = self.index_of[key]
        self.priorities[index] = priority
        if self._sift_up(index) == index { self._sift_down(index) }
    }

    #: Remove and return the key with the first priority. O(log n).
    #:
    #: Raises `IndexError` when the heap is empty.
    pub fn pop(self) -> tuple[K, P] {
        if self.keys.len() == 0 { throw IndexError("pop from an empty IndexedHeap") }
        let key = self.keys[0]
        let priority = self.priorities[0]
        self._detach(0)
        if self.keys.len() > 0 { self._sift_down(0) }
        return (key, priority)
    }

    #: The key and priority at the front of the heap, or null when empty. O(1).
    pub fn peek(self) -> tuple[K, P]? {
        if self.keys.len() == 0 { return null }
        return (self.keys[0], self.priorities[0])
    }

    #: Remove `key` and return its priority. O(log n).
    #:
    #: Raises `KeyError` when `key` is absent.
    pub fn remove(self, key: K) -> P {
        if key not in self.index_of { throw KeyError(f"key not in the heap: {key}") }
        let index = self.index_of[key]
        let priority = self.priorities[index]
        self._detach(index)
        if index < self.keys.len() {
            if self._sift_up(index) == index { self._sift_down(index) }
        }
        return priority
    }

    #: Return an independent heap with the same entries and comparator. O(n).
    pub fn copy(self) -> IndexedHeap[K, P] {
        let clone = IndexedHeap[K, P](self.less)
        clone.keys = [key for key in self.keys]
        clone.priorities = [priority for priority in self.priorities]
        clone.index_of = {key: index for (key, index) in self.index_of.items()}
        return clone
    }

    #: Return the entries in priority order, leaving the heap intact.
    #: O(n log n).
    pub fn sorted(self) -> list[tuple[K, P]] {
        let clone = self.copy()
        var out: list[tuple[K, P]] = []
        while clone.len() > 0 { out.push(clone.pop()) }
        return out
    }

    #: Discard every entry. O(1).
    pub fn clear(self) -> void {
        self.keys = []
        self.priorities = []
        self.index_of = {}
    }

    #: Iterate the keys in priority order. O(n log n), and allocates a copy.
    pub fn iter(self) -> Iterator[K] {
        var order: list[K] = []
        for (key, _priority) in self.sorted() { order.push(key) }
        return order.iter()
    }

    #: Number of keys in the heap. O(1).
    fn __len__(self) -> int {
        return self.keys.len()
    }

    #: Report whether `key` is in the heap. O(1).
    fn __contains__(self, key: K) -> bool {
        return key in self.index_of
    }

    #: Two indexed heaps are equal when they hold the same key-to-priority
    #: mapping, regardless of array layout. O(n).
    fn __eq__(self, other: IndexedHeap[K, P]) -> bool {
        if self.keys.len() != other.keys.len() { return false }
        for (index, key) in self.keys.enumerate() {
            if key not in other.index_of { return false }
            if other.priorities[other.index_of[key]] != self.priorities[index] { return false }
        }
        return true
    }

    #: Render the entries in priority order. O(n log n).
    fn __str__(self) -> str {
        return f"IndexedHeap({self.sorted()})"
    }

    fn _detach(self, index: int) -> void {
        let last = self.keys.len() - 1
        let key = self.keys[index]
        self._swap(index, last)
        self.keys.pop()
        self.priorities.pop()
        self.index_of.remove(key)
    }

    fn _sift_up(self, from_index: int) -> int {
        var index = from_index
        while index > 0 {
            let parent = (index - 1) // 2
            if not self.less(self.priorities[index], self.priorities[parent]) { break }
            self._swap(index, parent)
            index = parent
        }
        return index
    }

    fn _sift_down(self, from_index: int) -> void {
        let count = self.keys.len()
        var index = from_index
        loop {
            let left = index * 2 + 1
            if left >= count { break }
            var best = left
            let right = left + 1
            if right < count and self.less(self.priorities[right], self.priorities[left]) {
                best = right
            }
            if not self.less(self.priorities[best], self.priorities[index]) { break }
            self._swap(index, best)
            index = best
        }
    }

    fn _swap(self, a: int, b: int) -> void {
        if a == b { return }
        let key_a = self.keys[a]
        let key_b = self.keys[b]
        self.keys[a] = key_b
        self.keys[b] = key_a
        let priority_a = self.priorities[a]
        self.priorities[a] = self.priorities[b]
        self.priorities[b] = priority_a
        self.index_of[key_a] = b
        self.index_of[key_b] = a
    }
}

#: Merge already-sorted lists into one sorted list. O(N log k) for k sources.
#:
#: Equal elements keep the order of their source lists, so the merge is stable.
#: Each source must already be sorted by `less`; the result is unspecified
#: otherwise.
#:
#: ```
#: merge([[1, 4, 7], [2, 3], [5]])      # [1, 2, 3, 4, 5, 7]
#: ```
pub fn merge[T](sources: list[list[T]], less: fn(T, T) -> bool = |a, b| a < b) -> list[T] {
    let heap = Heap[tuple[T, int, int]](
        |a, b| less(a[0], b[0]) or not less(b[0], a[0]) and a[1] < b[1]
    )
    for (source_index, source) in sources.enumerate() {
        if source.len() > 0 { heap.push((source[0], source_index, 0)) }
    }
    var out: list[T] = []
    while heap.len() > 0 {
        let (value, source_index, offset) = heap.pop()
        out.push(value)
        let source = sources[source_index]
        if offset + 1 < source.len() {
            heap.push((source[offset + 1], source_index, offset + 1))
        }
    }
    return out
}

#: Return the `count` largest elements of `items`, largest first. O(n log k).
#:
#: Only `count` elements are ever held in memory, which is the point: sorting
#: the whole input to take a few elements is O(n log n).
#:
#: ```
#: nlargest(2, [5, 1, 9, 3])            # [9, 5]
#: nlargest(0, [5, 1])                  # []
#: ```
pub fn nlargest[T](count: int, items: list[T], less: fn(T, T) -> bool = |a, b| a < b) -> list[T] {
    if count <= 0 { return [] }
    let heap = Heap[T](less)
    for item in items {
        if heap.len() < count { heap.push(item) } else { heap.push_pop(item) }
    }
    return heap.drain().reversed()
}

#: Return the `count` smallest elements of `items`, smallest first. O(n log k).
#:
#: ```
#: nsmallest(2, [5, 1, 9, 3])           # [1, 3]
#: ```
pub fn nsmallest[T](count: int, items: list[T], less: fn(T, T) -> bool = |a, b| a < b) -> list[T] {
    if count <= 0 { return [] }
    let heap = Heap[T](|a, b| less(b, a))
    for item in items {
        if heap.len() < count { heap.push(item) } else { heap.push_pop(item) }
    }
    return heap.drain().reversed()
}
