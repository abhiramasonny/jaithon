#: Node-based sequences: `SinglyLinkedList` and `DoublyLinkedList`.
#:
#: The node types are public and the node-level splice operations are exposed,
#: because that is the whole reason to reach for a linked list: an LRU cache or
#: an insertion-ordered dictionary holds node handles in a hash map and needs
#: O(1) unlink and relink. A node handle belongs to exactly one list; passing a
#: node to a list that does not own it corrupts both.
from std.core import Iterable, Iterator

#: One element of a `SinglyLinkedList`: a value and the link after it.
pub class SinglyNode[T] {
    #: The value stored in this node. Assigning to it replaces the element.
    pub var value: T
    #: The following node, or null at the end of the list.
    #:
    #: Read it to walk forwards. Assigning to it by hand desynchronises the
    #: owning list's element count; use the list's splice methods instead.
    pub var next: SinglyNode[T]?

    #: Create a detached node holding `value`. O(1).
    pub fn init(self, value: T) {
        self.value = value
        self.next = null
    }

    #: Render the node's value. O(1).
    fn __str__(self) -> str {
        return f"SinglyNode({self.value})"
    }
}

#: A forward-linked list with O(1) insertion at both ends and O(1) removal at
#: the front.
pub class SinglyLinkedList[T]: Iterable[T] {
    var head: SinglyNode[T]?
    var tail: SinglyNode[T]?
    var count: int

    #: Create an empty list. O(1).
    pub fn init(self) {
        self.head = null
        self.tail = null
        self.count = 0
    }

    #: Build a list from `items`, first element at the front. O(n).
    #:
    #: ```
    #: SinglyLinkedList[int].from_list([1, 2, 3]).to_list()      # [1, 2, 3]
    #: ```
    pub static fn from_list(items: list[T]) -> SinglyLinkedList[T] {
        let result = SinglyLinkedList[T]()
        for item in items { result.push_back(item) }
        return result
    }

    #: Number of elements in the list. O(1).
    pub fn len(self) -> int {
        return self.count
    }

    #: Report whether the list holds no elements. O(1).
    pub fn is_empty(self) -> bool {
        return self.count == 0
    }

    #: The first value, or null when the list is empty. O(1).
    pub fn first(self) -> T? {
        return self.head?.value
    }

    #: The last value, or null when the list is empty. O(1).
    pub fn last(self) -> T? {
        return self.tail?.value
    }

    #: The node holding the first value, or null when the list is empty. O(1).
    pub fn first_node(self) -> SinglyNode[T]? {
        return self.head
    }

    #: The node holding the last value, or null when the list is empty. O(1).
    pub fn last_node(self) -> SinglyNode[T]? {
        return self.tail
    }

    #: Insert `value` at the front and return its node. O(1).
    pub fn push_front(self, value: T) -> SinglyNode[T] {
        let node = SinglyNode[T](value)
        node.next = self.head
        self.head = node
        if self.tail == null { self.tail = node }
        self.count += 1
        return node
    }

    #: Append `value` and return its node. O(1), because the tail is tracked.
    pub fn push_back(self, value: T) -> SinglyNode[T] {
        let node = SinglyNode[T](value)
        let tail = self.tail
        if tail == null { self.head = node } else { tail.next = node }
        self.tail = node
        self.count += 1
        return node
    }

    #: Remove and return the first value. O(1).
    #:
    #: Raises `IndexError` when the list is empty. There is no O(1) `pop_back`
    #: on a singly linked list; use `DoublyLinkedList` when you need one.
    pub fn pop_front(self) -> T {
        let node = self.head
        if node == null { throw IndexError("pop_front from an empty SinglyLinkedList") }
        self.head = node.next
        if self.head == null { self.tail = null }
        node.next = null
        self.count -= 1
        return node.value
    }

    #: Insert `value` directly after `node` and return the new node. O(1).
    #:
    #: `node` must belong to this list.
    pub fn insert_after(self, node: SinglyNode[T], value: T) -> SinglyNode[T] {
        let fresh = SinglyNode[T](value)
        fresh.next = node.next
        node.next = fresh
        if self.tail is node { self.tail = fresh }
        self.count += 1
        return fresh
    }

    #: Remove the node following `node` and return its value, or null when
    #: `node` is last. O(1).
    #:
    #: This is the only O(1) removal a singly linked list supports, since
    #: unlinking needs the predecessor.
    pub fn remove_after(self, node: SinglyNode[T]) -> T? {
        let doomed = node.next
        if doomed == null { return null }
        node.next = doomed.next
        if self.tail is doomed { self.tail = node }
        doomed.next = null
        self.count -= 1
        return doomed.value
    }

    #: Remove `node` from the list. O(n), because the predecessor is searched.
    #:
    #: Raises `ValueError` when `node` is not in this list.
    pub fn remove_node(self, node: SinglyNode[T]) -> T {
        if self.head is node { return self.pop_front() }
        var current = self.head
        while current != null {
            let previous = current
            if previous.next is node {
                previous.next = node.next
                if self.tail is node { self.tail = previous }
                node.next = null
                self.count -= 1
                return node.value
            }
            current = previous.next
        }
        throw ValueError("node is not a member of this list")
    }

    #: The first node whose value equals `value`, or null. O(n).
    pub fn find(self, value: T) -> SinglyNode[T]? {
        var current = self.head
        while current != null {
            let node = current
            if node.value == value { return node }
            current = node.next
        }
        return null
    }

    #: Remove the first occurrence of `value` and report whether it was found.
    #: O(n).
    pub fn remove(self, value: T) -> bool {
        let node = self.find(value)
        if node == null { return false }
        self.remove_node(node)
        return true
    }

    #: Reverse the list in place. O(n) time, O(1) extra space.
    #:
    #: ```
    #: let items = SinglyLinkedList[int].from_list([1, 2, 3])
    #: items.reverse()
    #: items.to_list()                     # [3, 2, 1]
    #: ```
    pub fn reverse(self) -> void {
        var previous: SinglyNode[T]? = null
        var current = self.head
        self.tail = self.head
        while current != null {
            let node = current
            current = node.next
            node.next = previous
            previous = node
        }
        self.head = previous
    }

    #: Copy the values into a list, front first. O(n).
    pub fn to_list(self) -> list[T] {
        var out: list[T] = []
        var current = self.head
        while current != null {
            let node = current
            out.push(node.value)
            current = node.next
        }
        return out
    }

    #: Collect the nodes, front first, so callers can splice while iterating.
    #: O(n).
    pub fn nodes(self) -> list[SinglyNode[T]] {
        var out: list[SinglyNode[T]] = []
        var current = self.head
        while current != null {
            let node = current
            out.push(node)
            current = node.next
        }
        return out
    }

    #: Discard every element. O(1); the detached nodes are collected.
    pub fn clear(self) -> void {
        self.head = null
        self.tail = null
        self.count = 0
    }

    #: Iterate the values from front to back. O(n), and allocates a snapshot.
    pub fn iter(self) -> Iterator[T] {
        return self.to_list().iter()
    }

    #: Number of elements in the list. O(1).
    fn __len__(self) -> int {
        return self.count
    }

    #: Report whether some element equals `value`. O(n).
    fn __contains__(self, value: T) -> bool {
        return self.find(value) != null
    }

    #: Two lists are equal when they hold equal values in the same order. O(n).
    fn __eq__(self, other: SinglyLinkedList[T]) -> bool {
        if self.count != other.count { return false }
        return self.to_list() == other.to_list()
    }

    #: Render the values front first. O(n).
    fn __str__(self) -> str {
        return f"SinglyLinkedList({self.to_list()})"
    }
}

#: One element of a `DoublyLinkedList`: a value and the links either side.
pub class DoublyNode[T] {
    #: The value stored in this node. Assigning to it replaces the element.
    pub var value: T
    #: The preceding node, or null at the front of the list.
    pub var prev: DoublyNode[T]?
    #: The following node, or null at the end of the list.
    #:
    #: Read `prev` and `next` to walk in either direction. Assigning to them by
    #: hand desynchronises the owning list; use the list's splice methods.
    pub var next: DoublyNode[T]?
    #: True while this node is a member of a list.
    #:
    #: The owning list maintains it; treat it as read-only. `remove_node` uses
    #: it to reject a node that has already been removed.
    pub var attached: bool

    #: Create a detached node holding `value`. O(1).
    pub fn init(self, value: T) {
        self.value = value
        self.prev = null
        self.next = null
        self.attached = false
    }

    #: Render the node's value. O(1).
    fn __str__(self) -> str {
        return f"DoublyNode({self.value})"
    }
}

#: A list linked in both directions, so any node can be spliced out, moved or
#: inserted beside in O(1) once you hold it.
pub class DoublyLinkedList[T]: Iterable[T] {
    var head: DoublyNode[T]?
    var tail: DoublyNode[T]?
    var count: int

    #: Create an empty list. O(1).
    pub fn init(self) {
        self.head = null
        self.tail = null
        self.count = 0
    }

    #: Build a list from `items`, first element at the front. O(n).
    pub static fn from_list(items: list[T]) -> DoublyLinkedList[T] {
        let result = DoublyLinkedList[T]()
        for item in items { result.push_back(item) }
        return result
    }

    #: Number of elements in the list. O(1).
    pub fn len(self) -> int {
        return self.count
    }

    #: Report whether the list holds no elements. O(1).
    pub fn is_empty(self) -> bool {
        return self.count == 0
    }

    #: The first value, or null when the list is empty. O(1).
    pub fn first(self) -> T? {
        return self.head?.value
    }

    #: The last value, or null when the list is empty. O(1).
    pub fn last(self) -> T? {
        return self.tail?.value
    }

    #: The node holding the first value, or null when the list is empty. O(1).
    pub fn first_node(self) -> DoublyNode[T]? {
        return self.head
    }

    #: The node holding the last value, or null when the list is empty. O(1).
    pub fn last_node(self) -> DoublyNode[T]? {
        return self.tail
    }

    #: Insert `value` at the front and return its node. O(1).
    pub fn push_front(self, value: T) -> DoublyNode[T] {
        let node = DoublyNode[T](value)
        node.attached = true
        self._link_front(node)
        return node
    }

    #: Append `value` and return its node. O(1).
    pub fn push_back(self, value: T) -> DoublyNode[T] {
        let node = DoublyNode[T](value)
        node.attached = true
        self._link_back(node)
        return node
    }

    #: Remove and return the first value. O(1).
    #:
    #: Raises `IndexError` when the list is empty.
    pub fn pop_front(self) -> T {
        let node = self.head
        if node == null { throw IndexError("pop_front from an empty DoublyLinkedList") }
        return self.remove_node(node)
    }

    #: Remove and return the last value. O(1).
    #:
    #: Raises `IndexError` when the list is empty.
    pub fn pop_back(self) -> T {
        let node = self.tail
        if node == null { throw IndexError("pop_back from an empty DoublyLinkedList") }
        return self.remove_node(node)
    }

    #: Insert `value` directly before `node` and return the new node. O(1).
    #:
    #: `node` must be attached to this list; raises `ValueError` otherwise.
    pub fn insert_before(self, node: DoublyNode[T], value: T) -> DoublyNode[T] {
        self._require_attached(node)
        let fresh = DoublyNode[T](value)
        fresh.attached = true
        let previous = node.prev
        fresh.prev = previous
        fresh.next = node
        node.prev = fresh
        if previous == null { self.head = fresh } else { previous.next = fresh }
        self.count += 1
        return fresh
    }

    #: Insert `value` directly after `node` and return the new node. O(1).
    #:
    #: `node` must be attached to this list; raises `ValueError` otherwise.
    #:
    #: ```
    #: let items = DoublyLinkedList[int]()
    #: let first = items.push_back(1)
    #: items.push_back(3)
    #: items.insert_after(first, 2)
    #: items.to_list()                     # [1, 2, 3]
    #: ```
    pub fn insert_after(self, node: DoublyNode[T], value: T) -> DoublyNode[T] {
        self._require_attached(node)
        let fresh = DoublyNode[T](value)
        fresh.attached = true
        let following = node.next
        fresh.prev = node
        fresh.next = following
        node.next = fresh
        if following == null { self.tail = fresh } else { following.prev = fresh }
        self.count += 1
        return fresh
    }

    #: Unlink `node` and return its value. O(1).
    #:
    #: Raises `ValueError` when the node is already detached. The node is left
    #: usable as a free-standing value holder.
    pub fn remove_node(self, node: DoublyNode[T]) -> T {
        self._require_attached(node)
        self._unlink(node)
        node.attached = false
        return node.value
    }

    #: Move `node` to the front of the list. O(1).
    #:
    #: This plus a key-to-node map is the whole of an LRU cache's bookkeeping.
    pub fn move_to_front(self, node: DoublyNode[T]) -> void {
        self._require_attached(node)
        if self.head is node { return }
        self._unlink(node)
        self._link_front(node)
    }

    #: Move `node` to the back of the list. O(1).
    pub fn move_to_back(self, node: DoublyNode[T]) -> void {
        self._require_attached(node)
        if self.tail is node { return }
        self._unlink(node)
        self._link_back(node)
    }

    #: The first node whose value equals `value`, or null. O(n).
    pub fn find(self, value: T) -> DoublyNode[T]? {
        var current = self.head
        while current != null {
            let node = current
            if node.value == value { return node }
            current = node.next
        }
        return null
    }

    #: Remove the first occurrence of `value` and report whether it was found.
    #: O(n).
    pub fn remove(self, value: T) -> bool {
        let node = self.find(value)
        if node == null { return false }
        self.remove_node(node)
        return true
    }

    #: Reverse the list in place. O(n) time, O(1) extra space.
    pub fn reverse(self) -> void {
        var current = self.head
        while current != null {
            let node = current
            current = node.next
            let previous = node.prev
            node.prev = node.next
            node.next = previous
        }
        let old_head = self.head
        self.head = self.tail
        self.tail = old_head
    }

    #: Copy the values into a list, front first. O(n).
    pub fn to_list(self) -> list[T] {
        var out: list[T] = []
        var current = self.head
        while current != null {
            let node = current
            out.push(node.value)
            current = node.next
        }
        return out
    }

    #: Copy the values into a list, back first. O(n).
    pub fn to_list_reversed(self) -> list[T] {
        var out: list[T] = []
        var current = self.tail
        while current != null {
            let node = current
            out.push(node.value)
            current = node.prev
        }
        return out
    }

    #: Collect the nodes, front first, so callers can splice while iterating.
    #: O(n).
    pub fn nodes(self) -> list[DoublyNode[T]] {
        var out: list[DoublyNode[T]] = []
        var current = self.head
        while current != null {
            let node = current
            out.push(node)
            current = node.next
        }
        return out
    }

    #: Discard every element. O(n).
    #:
    #: Every node is detached rather than merely dropped, so that a handle kept
    #: by a caller cannot later be spliced back into the emptied list, and so
    #: that the prev/next cycles are broken for the collector.
    pub fn clear(self) -> void {
        var current = self.head
        while current != null {
            let node = current
            current = node.next
            node.attached = false
            node.prev = null
            node.next = null
        }
        self.head = null
        self.tail = null
        self.count = 0
    }

    #: Iterate the values from front to back. O(n), and allocates a snapshot.
    pub fn iter(self) -> Iterator[T] {
        return self.to_list().iter()
    }

    #: Iterate the values from back to front. O(n), and allocates a snapshot.
    pub fn iter_reversed(self) -> Iterator[T] {
        return self.to_list_reversed().iter()
    }

    #: Number of elements in the list. O(1).
    fn __len__(self) -> int {
        return self.count
    }

    #: Report whether some element equals `value`. O(n).
    fn __contains__(self, value: T) -> bool {
        return self.find(value) != null
    }

    #: Two lists are equal when they hold equal values in the same order. O(n).
    fn __eq__(self, other: DoublyLinkedList[T]) -> bool {
        if self.count != other.count { return false }
        return self.to_list() == other.to_list()
    }

    #: Render the values front first. O(n).
    fn __str__(self) -> str {
        return f"DoublyLinkedList({self.to_list()})"
    }

    fn _require_attached(self, node: DoublyNode[T]) -> void {
        if not node.attached {
            throw ValueError(f"node {node} is not attached to a list")
        }
    }

    # Splice `node` out. The node keeps `attached` unchanged so that the move
    # operations can relink it without a window in which it looks detached.
    fn _unlink(self, node: DoublyNode[T]) -> void {
        let previous = node.prev
        let following = node.next
        if previous == null { self.head = following } else { previous.next = following }
        if following == null { self.tail = previous } else { following.prev = previous }
        node.prev = null
        node.next = null
        self.count -= 1
    }

    fn _link_front(self, node: DoublyNode[T]) -> void {
        let old_head = self.head
        node.prev = null
        node.next = old_head
        if old_head == null { self.tail = node } else { old_head.prev = node }
        self.head = node
        self.count += 1
    }

    fn _link_back(self, node: DoublyNode[T]) -> void {
        let old_tail = self.tail
        node.prev = old_tail
        node.next = null
        if old_tail == null { self.head = node } else { old_tail.next = node }
        self.tail = node
        self.count += 1
    }
}
