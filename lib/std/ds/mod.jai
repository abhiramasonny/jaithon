#: Data structures: stacks and queues, search trees, graphs, matrices and
#: string indexes.
#:
#: ```
#: from std.ds import Stack, Graph
#: let stack = Stack[int]()
#: stack.push(1)
#: stack.push(2)
#: stack.pop()      # 2
#: ```
from std.core import Iterable, Iterator
from std.collections import Deque

from .tree import natural_order, BinarySearchTree, AVLTree, SegmentTree, FenwickTree
from .graph import Edge, UnionFind, ForestStatistics, DisjointSetForest, Graph
from .matrix import Matrix
from .text import Trie, SuffixArray, RollingHash

export { natural_order, BinarySearchTree, AVLTree, SegmentTree, FenwickTree }
export { Edge, UnionFind, ForestStatistics, DisjointSetForest, Graph }
export { Matrix, Trie, SuffixArray, RollingHash }

#: A last-in-first-out stack with amortised O(1) push and pop.
#:
#: ```
#: let stack = Stack.from_list([1, 2, 3])
#: stack.peek()     # 3
#: stack.pop()      # 3
#: stack.len()      # 2
#: ```
pub class Stack[T]: Iterable[T] {
    var items: list[T]

    #: Create an empty stack.
    pub fn init(self) {
        self.items = []
    }

    #: Create a stack from an existing list, bottom element first.
    pub static fn from_list(items: list[T]) -> Stack[T] {
        let stack = Stack[T]()
        for item in items { stack.push(item) }
        return stack
    }

    #: Number of elements on the stack.
    pub fn len(self) -> int {
        return self.items.len()
    }

    #: Report whether the stack holds no elements. O(1).
    pub fn is_empty(self) -> bool {
        return self.items.len() == 0
    }

    #: Push `value` onto the top of the stack.
    pub fn push(self, value: T) -> void {
        self.items.push(value)
    }

    #: Remove and return the top element.
    #:
    #: Raises `IndexError` when the stack is empty.
    pub fn pop(self) -> T {
        if self.items.len() == 0 {
            throw IndexError("pop from an empty stack")
        }
        return self.items.pop()
    }

    #: The top element without removing it, or null when empty.
    pub fn peek(self) -> T? {
        if self.items.len() == 0 { return null }
        return self.items[self.items.len() - 1]
    }

    #: The elements bottom first, as a fresh list.
    pub fn to_list(self) -> list[T] {
        return [item for item in self.items]
    }

    #: Discard every element and release the backing list. O(1).
    pub fn clear(self) -> void {
        self.items = []
    }

    #: Iterate from the top of the stack downwards.
    pub fn iter(self) -> Iterator[T] {
        return self.items.reversed().iter()
    }

    fn __len__(self) -> int { return self.items.len() }

    fn __str__(self) -> str {
        return f"Stack({self.items})"
    }
}

#: A first-in-first-out queue with O(1) enqueue and dequeue.
#:
#: Backed by a `Deque`, so dequeuing never shifts the remaining elements the way
#: removing the head of a list would.
#:
#: ```
#: let queue = Queue.from_list(["a", "b"])
#: queue.enqueue("c")
#: queue.dequeue()   # "a"
#: queue.peek()      # "b"
#: queue.len()       # 2
#: ```
pub class Queue[T]: Iterable[T] {
    let items: Deque[T]

    #: Create an empty queue.
    pub fn init(self) {
        self.items = Deque[T]()
    }

    #: Create a queue from an existing list, front element first.
    pub static fn from_list(items: list[T]) -> Queue[T] {
        let queue = Queue[T]()
        for item in items { queue.enqueue(item) }
        return queue
    }

    #: Number of elements waiting.
    pub fn len(self) -> int {
        return self.items.len()
    }

    #: Report whether the queue holds no elements. O(1).
    pub fn is_empty(self) -> bool {
        return self.items.len() == 0
    }

    #: Add `value` to the back of the queue.
    pub fn enqueue(self, value: T) -> void {
        self.items.push_back(value)
    }

    #: Remove and return the front element.
    #:
    #: Raises `IndexError` when the queue is empty.
    pub fn dequeue(self) -> T {
        if self.items.len() == 0 {
            throw IndexError("dequeue from an empty queue")
        }
        return self.items.pop_front()
    }

    #: The front element without removing it, or null when empty.
    pub fn peek(self) -> T? {
        if self.items.len() == 0 { return null }
        return self.items.front()
    }

    #: The elements front first, as a fresh list.
    pub fn to_list(self) -> list[T] {
        return self.items.to_list()
    }

    #: Discard every element and release the deque's backing buffer. O(1).
    pub fn clear(self) -> void {
        self.items.clear()
    }

    #: Iterate from the front of the queue to the back.
    pub fn iter(self) -> Iterator[T] {
        return self.items.iter()
    }

    fn __len__(self) -> int { return self.items.len() }

    fn __str__(self) -> str {
        return f"Queue({self.to_list()})"
    }
}
