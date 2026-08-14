#: Ring-buffer sequences: the growable `Deque` and the fixed-capacity
#: `CircularBuffer`.
#:
#: Both store their elements in a flat list addressed modulo a capacity, so
#: both ends and every index are O(1). They differ in what happens when the
#: buffer fills: a `Deque` doubles, a `CircularBuffer` evicts its oldest
#: element. Both are re-exported from `std.collections`.
from std.core import Iterable, Iterator

const MIN_CAPACITY = 8

#: A double-ended queue with O(1) push, pop and indexing at either end.
#:
#: Elements live in a ring buffer over a flat list, which doubles when it
#: fills, so pushes are amortised O(1) and no element is ever shifted.
pub class Deque[T]: Iterable[T] {
    var slots: list[T]
    var start: int
    var count: int
    var capacity: int

    #: Create an empty deque. O(1).
    #:
    #: ```
    #: let queue = Deque[int]()
    #: queue.push_back(2)
    #: queue.push_front(1)
    #: queue.to_list()                     # [1, 2]
    #: ```
    pub fn init(self) {
        self.slots = []
        self.start = 0
        self.count = 0
        self.capacity = 0
    }

    #: Build a deque from `items`, first element at the front. O(n).
    pub static fn from_list(items: list[T]) -> Deque[T] {
        let result = Deque[T]()
        for item in items { result.push_back(item) }
        return result
    }

    #: Number of elements. O(1).
    pub fn len(self) -> int {
        return self.count
    }

    #: Report whether the deque holds no elements. O(1).
    pub fn is_empty(self) -> bool {
        return self.count == 0
    }

    #: Append `value` to the back. Amortised O(1).
    pub fn push_back(self, value: T) -> void {
        self._reserve(value)
        self.slots[(self.start + self.count) % self.capacity] = value
        self.count += 1
    }

    #: Insert `value` at the front. Amortised O(1).
    pub fn push_front(self, value: T) -> void {
        self._reserve(value)
        self.start = (self.start - 1 + self.capacity) % self.capacity
        self.slots[self.start] = value
        self.count += 1
    }

    #: Remove and return the front element. O(1).
    #:
    #: Raises `IndexError` when the deque is empty.
    pub fn pop_front(self) -> T {
        if self.count == 0 { throw IndexError("pop_front from an empty Deque") }
        let vacated = self.start
        let value = self.slots[vacated]
        self.start = (self.start + 1) % self.capacity
        self.count -= 1
        self._release(vacated)
        return value
    }

    #: Remove and return the back element. O(1).
    #:
    #: Raises `IndexError` when the deque is empty.
    pub fn pop_back(self) -> T {
        if self.count == 0 { throw IndexError("pop_back from an empty Deque") }
        let vacated = (self.start + self.count - 1) % self.capacity
        let value = self.slots[vacated]
        self.count -= 1
        self._release(vacated)
        return value
    }

    #: The front element, or null when the deque is empty. O(1).
    pub fn front(self) -> T? {
        if self.count == 0 { return null }
        return self.slots[self.start]
    }

    #: The back element, or null when the deque is empty. O(1).
    pub fn back(self) -> T? {
        if self.count == 0 { return null }
        return self.slots[(self.start + self.count - 1) % self.capacity]
    }

    #: The element at `index`, counted from the front. O(1).
    #:
    #: Negative indices count back from the end. Raises `IndexError` when
    #: `index` is out of range.
    pub fn get(self, index: int) -> T {
        return self.slots[self._physical(index)]
    }

    #: Replace the element at `index`, counted from the front. O(1).
    pub fn set(self, index: int, value: T) -> void {
        self.slots[self._physical(index)] = value
    }

    #: Append every element of `values` to the back. Amortised O(m).
    pub fn extend(self, values: list[T]) -> void {
        for value in values { self.push_back(value) }
    }

    #: Insert every element of `values` at the front, keeping their order.
    #: Amortised O(m).
    pub fn extend_front(self, values: list[T]) -> void {
        var index = values.len() - 1
        while index >= 0 {
            self.push_front(values[index])
            index -= 1
        }
    }

    #: Rotate `steps` places towards the back, wrapping around.
    #:
    #: A negative `steps` rotates towards the front. Runs in O(min(k, n - k))
    #: where k is `steps` reduced modulo the length, because the shorter
    #: direction is taken.
    #:
    #: ```
    #: let queue = Deque[int].from_list([1, 2, 3, 4])
    #: queue.rotate(1)
    #: queue.to_list()                     # [4, 1, 2, 3]
    #: queue.rotate(-1)
    #: queue.to_list()                     # [1, 2, 3, 4]
    #: ```
    pub fn rotate(self, steps: int) -> void {
        if self.count < 2 { return }
        var shift = steps % self.count
        if shift < 0 { shift += self.count }
        if shift * 2 > self.count {
            for _step in 0..self.count - shift { self.push_back(self.pop_front()) }
        } else {
            for _step in 0..shift { self.push_front(self.pop_back()) }
        }
    }

    #: Copy the elements into a list, front first. O(n).
    pub fn to_list(self) -> list[T] {
        var out: list[T] = []
        for offset in 0..self.count {
            out.push(self.slots[(self.start + offset) % self.capacity])
        }
        return out
    }

    #: Discard every element and release the backing buffer. O(1).
    pub fn clear(self) -> void {
        self.slots = []
        self.start = 0
        self.count = 0
        self.capacity = 0
    }

    #: Iterate from the front to the back. O(n), and allocates a snapshot.
    pub fn iter(self) -> Iterator[T] {
        return self.to_list().iter()
    }

    #: Number of elements. O(1).
    fn __len__(self) -> int {
        return self.count
    }

    #: The element at `index`, counted from the front. O(1).
    fn __getitem__(self, index: int) -> T {
        return self.get(index)
    }

    #: Replace the element at `index`, counted from the front. O(1).
    fn __setitem__(self, index: int, value: T) -> void {
        self.set(index, value)
    }

    #: Report whether some element equals `value`. O(n).
    fn __contains__(self, value: T) -> bool {
        for offset in 0..self.count {
            if self.slots[(self.start + offset) % self.capacity] == value { return true }
        }
        return false
    }

    #: Two deques are equal when they hold equal elements in the same order.
    #: O(n).
    fn __eq__(self, other: Deque[T]) -> bool {
        if self.count != other.count { return false }
        return self.to_list() == other.to_list()
    }

    #: Render the elements front first. O(n).
    fn __str__(self) -> str {
        return f"Deque({self.to_list()})"
    }

    fn _physical(self, index: int) -> int {
        var offset = index
        if offset < 0 { offset += self.count }
        if offset < 0 or offset >= self.count {
            throw IndexError(f"index {index} out of range for a Deque of {self.count}")
        }
        return (self.start + offset) % self.capacity
    }

    fn _reserve(self, value: T) -> void {
        if self.count < self.capacity { return }
        let grown = if self.capacity == 0 { MIN_CAPACITY } else { self.capacity * 2 }
        var next: list[T] = []
        for offset in 0..self.count {
            next.push(self.slots[(self.start + offset) % self.capacity])
        }
        while next.len() < grown { next.push(value) }
        self.slots = next
        self.start = 0
        self.capacity = grown
    }

    fn _release(self, index: int) -> void {
        if self.count == 0 {
            self.clear()
        } else {
            self.slots[index] = self.slots[self.start]
        }
    }
}

#: A fixed-capacity ring buffer: a queue that drops its oldest element instead
#: of growing.
#:
#: This is the structure behind a sliding window over a stream — the last n
#: samples, the last n log lines — where bounded memory matters more than
#: keeping everything.
pub class CircularBuffer[T]: Iterable[T] {
    var slots: list[T]
    var start: int
    var count: int
    var limit: int

    #: Create a buffer holding at most `capacity` elements. O(1).
    #:
    #: Raises `ValueError` when `capacity` is not positive.
    #:
    #: ```
    #: let recent = CircularBuffer[int](3)
    #: recent.push(1)
    #: recent.push(2)
    #: recent.push(3)
    #: recent.push(4)                      # 1
    #: recent.to_list()                    # [2, 3, 4]
    #: ```
    pub fn init(self, capacity: int) {
        if capacity < 1 { throw ValueError(f"capacity must be positive, got {capacity}") }
        self.slots = []
        self.start = 0
        self.count = 0
        self.limit = capacity
    }

    #: Build a buffer of `capacity` from `items`, dropping all but the last
    #: `capacity` of them. O(n).
    pub static fn from_list(items: list[T], capacity: int) -> CircularBuffer[T] {
        let buffer = CircularBuffer[T](capacity)
        for item in items { buffer.push(item) }
        return buffer
    }

    #: Number of elements currently held. O(1).
    pub fn len(self) -> int {
        return self.count
    }

    #: The greatest number of elements the buffer can hold. O(1).
    pub fn capacity(self) -> int {
        return self.limit
    }

    #: Report whether the buffer holds no elements. O(1).
    pub fn is_empty(self) -> bool {
        return self.count == 0
    }

    #: Report whether the buffer is at capacity. O(1).
    pub fn is_full(self) -> bool {
        return self.count == self.limit
    }

    #: Append `value`, returning the element it evicted, or null. O(1).
    #:
    #: When the buffer is full the oldest element is dropped to make room, which
    #: is what makes this a sliding window over a stream.
    pub fn push(self, value: T) -> T? {
        if self.count < self.limit {
            self._store((self.start + self.count) % self.limit, value)
            self.count += 1
            return null
        }
        let evicted = self.slots[self.start]
        self.slots[self.start] = value
        self.start = (self.start + 1) % self.limit
        return evicted
    }

    #: Append `value` unless the buffer is full; report whether it fit. O(1).
    pub fn try_push(self, value: T) -> bool {
        if self.count == self.limit { return false }
        self.push(value)
        return true
    }

    #: Remove and return the oldest element. O(1).
    #:
    #: Raises `IndexError` when the buffer is empty.
    pub fn pop(self) -> T {
        if self.count == 0 { throw IndexError("pop from an empty CircularBuffer") }
        let vacated = self.start
        let value = self.slots[vacated]
        self.start = (self.start + 1) % self.limit
        self.count -= 1
        self._release(vacated)
        return value
    }

    #: The oldest element, or null when the buffer is empty. O(1).
    pub fn first(self) -> T? {
        if self.count == 0 { return null }
        return self.slots[self.start]
    }

    #: The newest element, or null when the buffer is empty. O(1).
    pub fn last(self) -> T? {
        if self.count == 0 { return null }
        return self.slots[(self.start + self.count - 1) % self.limit]
    }

    #: The element `index` places from the oldest. O(1).
    #:
    #: Negative indices count back from the newest. Raises `IndexError` when
    #: `index` is out of range.
    pub fn get(self, index: int) -> T {
        return self.slots[self._physical(index)]
    }

    #: Replace the element at `index`, counted from the oldest. O(1).
    pub fn set(self, index: int, value: T) -> void {
        self.slots[self._physical(index)] = value
    }

    #: Copy the elements into a list, oldest first. O(n).
    pub fn to_list(self) -> list[T] {
        var out: list[T] = []
        for offset in 0..self.count {
            out.push(self.slots[(self.start + offset) % self.limit])
        }
        return out
    }

    #: Discard every element, keeping the capacity. O(1).
    pub fn clear(self) -> void {
        self.slots = []
        self.start = 0
        self.count = 0
    }

    #: Iterate from the oldest element to the newest. O(n), allocates a
    #: snapshot.
    pub fn iter(self) -> Iterator[T] {
        return self.to_list().iter()
    }

    #: Number of elements currently held. O(1).
    fn __len__(self) -> int {
        return self.count
    }

    #: The element `index` places from the oldest. O(1).
    fn __getitem__(self, index: int) -> T {
        return self.get(index)
    }

    #: Replace the element at `index`, counted from the oldest. O(1).
    fn __setitem__(self, index: int, value: T) -> void {
        self.set(index, value)
    }

    #: Report whether some element equals `value`. O(n).
    fn __contains__(self, value: T) -> bool {
        for offset in 0..self.count {
            if self.slots[(self.start + offset) % self.limit] == value { return true }
        }
        return false
    }

    #: Two buffers are equal when they hold equal elements in the same order,
    #: whatever their capacities. O(n).
    fn __eq__(self, other: CircularBuffer[T]) -> bool {
        if self.count != other.count { return false }
        return self.to_list() == other.to_list()
    }

    #: Render the elements oldest first. O(n).
    fn __str__(self) -> str {
        return f"CircularBuffer({self.to_list()})"
    }

    fn _physical(self, index: int) -> int {
        var offset = index
        if offset < 0 { offset += self.count }
        if offset < 0 or offset >= self.count {
            throw IndexError(f"index {index} out of range for a CircularBuffer of {self.count}")
        }
        return (self.start + offset) % self.limit
    }

    fn _store(self, index: int, value: T) -> void {
        if index == self.slots.len() {
            self.slots.push(value)
        } else {
            self.slots[index] = value
        }
    }

    fn _release(self, index: int) -> void {
        if self.count == 0 {
            self.slots = []
            self.start = 0
        } else {
            self.slots[index] = self.slots[self.start]
        }
    }
}
