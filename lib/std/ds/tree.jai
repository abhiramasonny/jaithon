#: Ordered tree structures: an unbalanced search tree, an AVL tree, a segment
#: tree over any associative operation, and a Fenwick tree.
#:
#: Every `BinarySearchTree` walk in this module is iterative. Inserting sorted
#: keys into an unbalanced tree degenerates it into a chain of length n, so a
#: recursive traversal would raise `RecursionError` on inputs an explicit stack
#: handles without trouble. `AVLTree` is recursive where it is convenient
#: because its depth is bounded by 1.44 * log2(n) — under 100 frames for any
#: tree that fits in memory.

#: Compare `a` and `b` with `<` and `>`, returning -1, 0, or 1.
#:
#: This is the default ordering for `BinarySearchTree` and `AVLTree`. Pass a
#: different `fn(K, K) -> int` to either constructor to order keys another way.
#:
#: ```
#: natural_order(1, 2)       # -1
#: natural_order("b", "a")   # 1
#: ```
pub fn natural_order[T](a: T, b: T) -> int {
    if a < b { return -1 }
    if a > b { return 1 }
    return 0
}

class _BstNode[K, V] {
    pub var key: K
    pub var value: V
    pub var left: _BstNode[K, V]?
    pub var right: _BstNode[K, V]?

    pub fn init(self, key: K, value: V) {
        self.key = key
        self.value = value
        self.left = null
        self.right = null
    }
}

#: An unbalanced binary search tree mapping keys to values.
#:
#: Operations are O(h) where h is the current height: O(log n) for keys that
#: arrive in random order, O(n) for keys that arrive sorted. Use `AVLTree` when
#: the insertion order is adversarial or unknown.
#:
#: ```
#: let tree = BinarySearchTree[int, str]()
#: tree.insert(5, "five")
#: tree.insert(2, "two")
#: tree.insert(8, "eight")
#: tree.in_order()        # [2, 5, 8]
#: tree.find(2)           # "two"
#: tree.successor(5)      # 8
#: ```
pub class BinarySearchTree[K, V] {
    var root: _BstNode[K, V]?
    var count: int
    let compare: fn(K, K) -> int

    #: Create an empty tree ordered by `compare`.
    pub fn init(self, compare: fn(K, K) -> int = natural_order) {
        self.root = null
        self.count = 0
        self.compare = compare
    }

    #: Number of keys in the tree.
    pub fn len(self) -> int {
        return self.count
    }

    #: Report whether the tree holds no keys. O(1).
    #:
    #: The count is maintained by `insert` and `remove`, so this stays O(1) even
    #: when sorted input has degenerated the tree into a chain and every other
    #: query costs O(n).
    pub fn is_empty(self) -> bool {
        return self.count == 0
    }

    #: Remove every key.
    pub fn clear(self) -> void {
        self.root = null
        self.count = 0
    }

    #: Associate `value` with `key`, returning true when `key` was new.
    #:
    #: An existing key keeps its position and takes the new value.
    pub fn insert(self, key: K, value: V) -> bool {
        if self.root == null {
            self.root = _BstNode[K, V](key, value)
            self.count += 1
            return true
        }
        var current = self.root
        loop {
            let order = self._cmp(key, current.key)
            if order == 0 {
                current.value = value
                return false
            }
            if order < 0 {
                if current.left == null {
                    current.left = _BstNode[K, V](key, value)
                    self.count += 1
                    return true
                }
                current = current.left
            } else {
                if current.right == null {
                    current.right = _BstNode[K, V](key, value)
                    self.count += 1
                    return true
                }
                current = current.right
            }
        }
    }

    #: The value stored under `key`, or null when the key is absent.
    #:
    #: A stored value of null is indistinguishable from an absent key here; use
    #: `contains` when the difference matters.
    pub fn find(self, key: K) -> V? {
        var current = self.root
        while current != null {
            let order = self._cmp(key, current.key)
            if order == 0 { return current.value }
            current = order < 0 ? current.left : current.right
        }
        return null
    }

    #: Report whether `key` is present.
    pub fn contains(self, key: K) -> bool {
        var current = self.root
        while current != null {
            let order = self._cmp(key, current.key)
            if order == 0 { return true }
            current = order < 0 ? current.left : current.right
        }
        return false
    }

    #: The smallest key, or null when the tree is empty.
    pub fn min(self) -> K? {
        if self.root == null { return null }
        var current = self.root
        while current.left != null { current = current.left }
        return current.key
    }

    #: The largest key, or null when the tree is empty.
    pub fn max(self) -> K? {
        if self.root == null { return null }
        var current = self.root
        while current.right != null { current = current.right }
        return current.key
    }

    #: The smallest key strictly greater than `key`, or null when there is none.
    #:
    #: `key` itself need not be in the tree.
    pub fn successor(self, key: K) -> K? {
        var candidate: K? = null
        var current = self.root
        while current != null {
            if self._cmp(current.key, key) > 0 {
                candidate = current.key
                current = current.left
            } else {
                current = current.right
            }
        }
        return candidate
    }

    #: The largest key strictly less than `key`, or null when there is none.
    pub fn predecessor(self, key: K) -> K? {
        var candidate: K? = null
        var current = self.root
        while current != null {
            if self._cmp(current.key, key) < 0 {
                candidate = current.key
                current = current.right
            } else {
                current = current.left
            }
        }
        return candidate
    }

    #: Remove `key`, returning true when it was present.
    pub fn remove(self, key: K) -> bool {
        var parent: _BstNode[K, V]? = null
        var current = self.root
        while current != null {
            let order = self._cmp(key, current.key)
            if order == 0 { break }
            parent = current
            current = order < 0 ? current.left : current.right
        }
        if current == null { return false }
        if current.left != null and current.right != null {
            # Two children: copy the in-order successor's payload into this
            # node and delete the successor instead. The successor has no left
            # child, so it falls into the one-child case below.
            # The branch condition proved `current.right` is present, but a
            # field is not a narrowing target (spec §2.1 narrows bindings), so
            # the invariant is restated on a local.
            var successor_parent = current
            var successor = current.right ?? (throw RuntimeError("bst: right child vanished"))
            var lower = successor.left
            while lower != null {
                successor_parent = successor
                successor = lower
                lower = successor.left
            }
            current.key = successor.key
            current.value = successor.value
            parent = successor_parent
            current = successor
        }
        let child = current.left ?? current.right
        if parent == null {
            self.root = child
        } elif parent.left is current {
            parent.left = child
        } else {
            parent.right = child
        }
        self.count -= 1
        return true
    }

    #: Every (key, value) pair in ascending key order.
    pub fn items(self) -> list[tuple[K, V]] {
        let pairs: list[tuple[K, V]] = []
        let stack: list[_BstNode[K, V]] = []
        var current = self.root
        while current != null or stack.len() > 0 {
            while current != null {
                stack.push(current)
                current = current.left
            }
            let node = stack.pop()
            pairs.push((node.key, node.value))
            current = node.right
        }
        return pairs
    }

    #: Every key in ascending order.
    pub fn in_order(self) -> list[K] {
        let keys: list[K] = []
        for (key, _value) in self.items() { keys.push(key) }
        return keys
    }

    #: Every key, each node before its subtrees.
    pub fn pre_order(self) -> list[K] {
        let keys: list[K] = []
        let root = self.root
        if root == null { return keys }
        let stack: list[_BstNode[K, V]] = [root]
        while stack.len() > 0 {
            let node = stack.pop()
            keys.push(node.key)
            if node.right != null { stack.push(node.right) }
            if node.left != null { stack.push(node.left) }
        }
        return keys
    }

    #: Every key, each node after its subtrees.
    pub fn post_order(self) -> list[K] {
        let keys: list[K] = []
        for node in self._post_order_nodes() { keys.push(node.key) }
        return keys
    }

    #: Number of edges on the longest root-to-leaf path; -1 for an empty tree.
    #:
    #: ```
    #: let tree = BinarySearchTree[int, int]()
    #: tree.insert(1, 1)
    #: tree.insert(2, 2)
    #: tree.height()      # 1
    #: ```
    pub fn height(self) -> int {
        if self.root == null { return -1 }
        return self._heights()[id(self.root)]
    }

    #: Report whether every node's subtree heights differ by at most one.
    pub fn is_balanced(self) -> bool {
        let heights = self._heights()
        for node in self._post_order_nodes() {
            let left = node.left == null ? -1 : heights[id(node.left)]
            let right = node.right == null ? -1 : heights[id(node.right)]
            if abs(left - right) > 1 { return false }
        }
        return true
    }

    fn _cmp(self, a: K, b: K) -> int {
        let compare = self.compare
        return compare(a, b)
    }

    fn _post_order_nodes(self) -> list[_BstNode[K, V]] {
        let visited: list[_BstNode[K, V]] = []
        let root = self.root
        if root == null { return visited }
        let stack: list[_BstNode[K, V]] = [root]
        while stack.len() > 0 {
            let node = stack.pop()
            visited.push(node)
            # Left is pushed first so the right subtree is visited first; the
            # reversal at the end turns root-right-left into left-right-root.
            if node.left != null { stack.push(node.left) }
            if node.right != null { stack.push(node.right) }
        }
        return visited.reversed()
    }

    fn _heights(self) -> dict[int, int] {
        let heights: dict[int, int] = {}
        for node in self._post_order_nodes() {
            let left = node.left == null ? -1 : heights[id(node.left)]
            let right = node.right == null ? -1 : heights[id(node.right)]
            heights[id(node)] = 1 + max(left, right)
        }
        return heights
    }

    fn __len__(self) -> int { return self.count }

    fn __contains__(self, key: K) -> bool { return self.contains(key) }

    fn __str__(self) -> str {
        return f"BinarySearchTree({self.in_order()})"
    }
}

class _AvlNode[K, V] {
    pub var key: K
    pub var value: V
    pub var height: int
    pub var left: _AvlNode[K, V]?
    pub var right: _AvlNode[K, V]?

    pub fn init(self, key: K, value: V) {
        self.key = key
        self.value = value
        self.height = 1
        self.left = null
        self.right = null
    }
}

#: A height-balanced binary search tree with O(log n) insert, remove and lookup.
#:
#: The tree keeps every node's subtree heights within one of each other by
#: rotating on the way back up from a modification, so worst-case depth stays
#: logarithmic no matter what order keys arrive in.
#:
#: ```
#: let tree = AVLTree[int, int]()
#: for key in 0..16 { tree.insert(key, key * key) }
#: tree.len()           # 16
#: tree.height()        # 4
#: tree.find(12)        # 144
#: tree.is_balanced()   # true
#: ```
pub class AVLTree[K, V] {
    var root: _AvlNode[K, V]?
    var count: int
    let compare: fn(K, K) -> int

    #: Create an empty tree ordered by `compare`.
    pub fn init(self, compare: fn(K, K) -> int = natural_order) {
        self.root = null
        self.count = 0
        self.compare = compare
    }

    #: Number of keys in the tree.
    pub fn len(self) -> int {
        return self.count
    }

    #: Report whether the tree holds no keys. O(1).
    #:
    #: An empty tree has a `height()` of -1 and a one-key tree has 0, so test
    #: emptiness with this rather than with `height() == 0`.
    pub fn is_empty(self) -> bool {
        return self.count == 0
    }

    #: Remove every key.
    pub fn clear(self) -> void {
        self.root = null
        self.count = 0
    }

    #: Associate `value` with `key`, returning true when `key` was new.
    pub fn insert(self, key: K, value: V) -> bool {
        let before = self.count
        self.root = self._insert(self.root, key, value)
        return self.count > before
    }

    #: Remove `key`, returning true when it was present.
    pub fn remove(self, key: K) -> bool {
        let before = self.count
        self.root = self._remove(self.root, key)
        return self.count < before
    }

    #: The value stored under `key`, or null when the key is absent.
    pub fn find(self, key: K) -> V? {
        let node = self._find_node(key)
        if node == null { return null }
        return node.value
    }

    #: Report whether `key` is present.
    pub fn contains(self, key: K) -> bool {
        return self._find_node(key) != null
    }

    #: The smallest key, or null when the tree is empty.
    pub fn min(self) -> K? {
        if self.root == null { return null }
        var current = self.root
        while current.left != null { current = current.left }
        return current.key
    }

    #: The largest key, or null when the tree is empty.
    pub fn max(self) -> K? {
        if self.root == null { return null }
        var current = self.root
        while current.right != null { current = current.right }
        return current.key
    }

    #: The smallest key strictly greater than `key`, or null when there is none.
    pub fn successor(self, key: K) -> K? {
        var candidate: K? = null
        var current = self.root
        while current != null {
            if self._cmp(current.key, key) > 0 {
                candidate = current.key
                current = current.left
            } else {
                current = current.right
            }
        }
        return candidate
    }

    #: Every (key, value) pair in ascending key order.
    pub fn items(self) -> list[tuple[K, V]] {
        let pairs: list[tuple[K, V]] = []
        let stack: list[_AvlNode[K, V]] = []
        var current = self.root
        while current != null or stack.len() > 0 {
            while current != null {
                stack.push(current)
                current = current.left
            }
            let node = stack.pop()
            pairs.push((node.key, node.value))
            current = node.right
        }
        return pairs
    }

    #: Every key in ascending order.
    pub fn in_order(self) -> list[K] {
        let keys: list[K] = []
        for (key, _value) in self.items() { keys.push(key) }
        return keys
    }

    #: Number of edges on the longest root-to-leaf path; -1 for an empty tree.
    pub fn height(self) -> int {
        return self._height_of(self.root) - 1
    }

    #: Verify the AVL invariant: stored heights are accurate and no node is
    #: more than one level out of balance.
    pub fn is_balanced(self) -> bool {
        return self._check(self.root) >= 0
    }

    fn _cmp(self, a: K, b: K) -> int {
        let compare = self.compare
        return compare(a, b)
    }

    fn _find_node(self, key: K) -> _AvlNode[K, V]? {
        var current = self.root
        while current != null {
            let order = self._cmp(key, current.key)
            if order == 0 { return current }
            current = order < 0 ? current.left : current.right
        }
        return null
    }

    fn _height_of(self, node: _AvlNode[K, V]?) -> int {
        if node == null { return 0 }
        return node.height
    }

    fn _update_height(self, node: _AvlNode[K, V]) -> void {
        node.height = 1 + max(self._height_of(node.left), self._height_of(node.right))
    }

    fn _balance_factor(self, node: _AvlNode[K, V]) -> int {
        return self._height_of(node.left) - self._height_of(node.right)
    }

    #: A rotation is only ever reached from a node whose balance factor proved
    #: the child exists. A field is not a narrowing target (spec §2.1 narrows
    #: bindings), so each rotation restates that invariant on a local and fails
    #: loudly rather than silently rotating around nothing.
    fn _rotate_right(self, node: _AvlNode[K, V]) -> _AvlNode[K, V] {
        let pivot = node.left ?? (throw RuntimeError("avl: left-heavy node has no left child"))
        node.left = pivot.right
        pivot.right = node
        self._update_height(node)
        self._update_height(pivot)
        return pivot
    }

    fn _rotate_left(self, node: _AvlNode[K, V]) -> _AvlNode[K, V] {
        let pivot = node.right ?? (throw RuntimeError("avl: right-heavy node has no right child"))
        node.right = pivot.left
        pivot.left = node
        self._update_height(node)
        self._update_height(pivot)
        return pivot
    }

    fn _rebalance(self, node: _AvlNode[K, V]) -> _AvlNode[K, V] {
        self._update_height(node)
        let balance = self._balance_factor(node)
        if balance > 1 {
            # Left-heavy. A right-leaning left child needs the left-right
            # double rotation, which is a left rotation of the child first.
            let left = node.left ?? (throw RuntimeError("avl: left-heavy node has no left child"))
            if self._balance_factor(left) < 0 {
                node.left = self._rotate_left(left)
            }
            return self._rotate_right(node)
        }
        if balance < -1 {
            let right = node.right ?? (throw RuntimeError("avl: right-heavy node has no right child"))
            if self._balance_factor(right) > 0 {
                node.right = self._rotate_right(right)
            }
            return self._rotate_left(node)
        }
        return node
    }

    fn _insert(self, node: _AvlNode[K, V]?, key: K, value: V) -> _AvlNode[K, V] {
        if node == null {
            self.count += 1
            return _AvlNode[K, V](key, value)
        }
        let order = self._cmp(key, node.key)
        if order == 0 {
            node.value = value
            return node
        }
        if order < 0 {
            node.left = self._insert(node.left, key, value)
        } else {
            node.right = self._insert(node.right, key, value)
        }
        return self._rebalance(node)
    }

    fn _remove(self, node: _AvlNode[K, V]?, key: K) -> _AvlNode[K, V]? {
        if node == null { return null }
        let order = self._cmp(key, node.key)
        if order < 0 {
            node.left = self._remove(node.left, key)
        } elif order > 0 {
            node.right = self._remove(node.right, key)
        } else {
            if node.left == null {
                self.count -= 1
                return node.right
            }
            if node.right == null {
                self.count -= 1
                return node.left
            }
            # Two children: copy the in-order successor's payload here and
            # delete the successor from the right subtree, where it has no left
            # child and so hits one of the two cases above. The count is
            # adjusted there, exactly once.
            var successor = node.right
            while successor.left != null { successor = successor.left }
            node.key = successor.key
            node.value = successor.value
            node.right = self._remove(node.right, successor.key)
        }
        return self._rebalance(node)
    }

    fn _check(self, node: _AvlNode[K, V]?) -> int {
        if node == null { return 0 }
        let left = self._check(node.left)
        let right = self._check(node.right)
        if left < 0 or right < 0 { return -1 }
        if abs(left - right) > 1 { return -1 }
        if node.height != 1 + max(left, right) { return -1 }
        return node.height
    }

    fn __len__(self) -> int { return self.count }

    fn __contains__(self, key: K) -> bool { return self.contains(key) }

    fn __str__(self) -> str {
        return f"AVLTree({self.in_order()})"
    }
}

#: A segment tree answering range queries over any associative operation.
#:
#: `combine` must be associative and `identity` must satisfy
#: `combine(identity, x) == x == combine(x, identity)`. Commutativity is not
#: required: queries fold the left and right frontiers separately and join them
#: in index order, so `combine` may be string concatenation or matrix product.
#:
#: Queries and updates are O(log n); construction is O(n).
#:
#: ```
#: let tree = SegmentTree[int]([1, 2, 3, 4], |a, b| a + b, 0)
#: tree.query(1, 3)     # 5
#: tree.update(1, 10)
#: tree.query(0, 4)     # 18
#: tree.get(1)          # 10
#: ```
pub class SegmentTree[T] {
    let size: int
    let nodes: list[T]
    let combine: fn(T, T) -> T
    let identity: T

    #: Build a tree over `values`.
    pub fn init(self, values: list[T], combine: fn(T, T) -> T, identity: T) {
        let count = values.len()
        self.size = count
        self.combine = combine
        self.identity = identity
        let nodes: list[T] = []
        for _slot in 0..2 * count { nodes.push(identity) }
        for (index, value) in values.enumerate() { nodes[count + index] = value }
        var parent = count - 1
        while parent >= 1 {
            nodes[parent] = combine(nodes[2 * parent], nodes[2 * parent + 1])
            parent -= 1
        }
        self.nodes = nodes
    }

    #: Number of leaves.
    pub fn len(self) -> int {
        return self.size
    }

    #: The value at `index`.
    pub fn get(self, index: int) -> T {
        self._check_index(index)
        return self.nodes[self.size + index]
    }

    #: Replace the value at `index` and refresh the path to the root.
    pub fn update(self, index: int, value: T) -> void {
        self._check_index(index)
        let combine = self.combine
        var position = self.size + index
        self.nodes[position] = value
        position //= 2
        while position >= 1 {
            self.nodes[position] = combine(self.nodes[2 * position], self.nodes[2 * position + 1])
            position //= 2
        }
    }

    #: Combine the values in the half-open range `[lo, hi)`.
    #:
    #: An empty range yields `identity`.
    pub fn query(self, lo: int, hi: int) -> T {
        if lo < 0 or hi > self.size or lo > hi {
            throw IndexError(f"range [{lo}, {hi}) is not within [0, {self.size})")
        }
        let combine = self.combine
        var left_fold = self.identity
        var right_fold = self.identity
        var left = lo + self.size
        var right = hi + self.size
        while left < right {
            if left % 2 == 1 {
                left_fold = combine(left_fold, self.nodes[left])
                left += 1
            }
            if right % 2 == 1 {
                right -= 1
                right_fold = combine(self.nodes[right], right_fold)
            }
            left //= 2
            right //= 2
        }
        return combine(left_fold, right_fold)
    }

    #: Combine every value in the tree.
    pub fn total(self) -> T {
        return self.query(0, self.size)
    }

    fn _check_index(self, index: int) -> void {
        if index < 0 or index >= self.size {
            throw IndexError(f"index {index} is not within [0, {self.size})")
        }
    }

    fn __len__(self) -> int { return self.size }
}

#: A Fenwick tree (binary indexed tree) over integers with O(log n) point
#: update and prefix sum.
#:
#: Smaller and faster than a segment tree, at the cost of only supporting
#: invertible operations — here, addition.
#:
#: ```
#: let sums = FenwickTree.from_list([3, 1, 4, 1, 5])
#: sums.prefix_sum(3)      # 8
#: sums.range_sum(1, 4)    # 6
#: sums.add(0, 10)
#: sums.range_sum(0, 5)    # 24
#: ```
pub class FenwickTree {
    let size: int
    let nodes: list[int]

    #: Create a tree of `size` zeros.
    pub fn init(self, size: int) {
        if size < 0 {
            throw ValueError(f"size must be non-negative, got {size}")
        }
        self.size = size
        let nodes: list[int] = []
        # Index 0 is unused: the `index & -index` stride only works 1-based.
        for _slot in 0..=size { nodes.push(0) }
        self.nodes = nodes
    }

    #: Build a tree holding `values`, in O(n) rather than n insertions.
    pub static fn from_list(values: list[int]) -> FenwickTree {
        let tree = FenwickTree(values.len())
        for (index, value) in values.enumerate() { tree.nodes[index + 1] = value }
        for index in 1..=values.len() {
            let parent = index + (index & -index)
            if parent <= values.len() { tree.nodes[parent] += tree.nodes[index] }
        }
        return tree
    }

    #: Number of elements.
    pub fn len(self) -> int {
        return self.size
    }

    #: Add `delta` to the element at `index`.
    pub fn add(self, index: int, delta: int) -> void {
        if index < 0 or index >= self.size {
            throw IndexError(f"index {index} is not within [0, {self.size})")
        }
        var position = index + 1
        while position <= self.size {
            self.nodes[position] += delta
            position += position & -position
        }
    }

    #: Sum of the first `count` elements.
    #:
    #: `prefix_sum(0)` is 0 and `prefix_sum(len())` is the total.
    pub fn prefix_sum(self, count: int) -> int {
        if count < 0 or count > self.size {
            throw IndexError(f"count {count} is not within [0, {self.size}]")
        }
        var total = 0
        var position = count
        while position > 0 {
            total += self.nodes[position]
            position -= position & -position
        }
        return total
    }

    #: Sum of the half-open range `[lo, hi)`.
    pub fn range_sum(self, lo: int, hi: int) -> int {
        if lo > hi {
            throw IndexError(f"range [{lo}, {hi}) is empty backwards")
        }
        return self.prefix_sum(hi) - self.prefix_sum(lo)
    }

    #: The element at `index`.
    pub fn get(self, index: int) -> int {
        return self.range_sum(index, index + 1)
    }

    #: Replace the element at `index` with `value`.
    pub fn set(self, index: int, value: int) -> void {
        self.add(index, value - self.get(index))
    }

    #: The smallest index whose prefix sum through it reaches `target`.
    #:
    #: Returns `len()` when no prefix reaches `target`. Every element must be
    #: non-negative, since the search relies on prefix sums being monotonic.
    #: This turns a Fenwick tree of counts into an order-statistic structure:
    #: with one count per value, `lower_bound(k)` is the k-th smallest.
    #:
    #: ```
    #: let counts = FenwickTree.from_list([1, 0, 2, 1])
    #: counts.lower_bound(1)   # 0
    #: counts.lower_bound(2)   # 2
    #: counts.lower_bound(4)   # 3
    #: counts.lower_bound(9)   # 4
    #: ```
    pub fn lower_bound(self, target: int) -> int {
        var position = 0
        var remaining = target
        var step = 1
        while step * 2 <= self.size { step *= 2 }
        while step > 0 {
            let probe = position + step
            if probe <= self.size and self.nodes[probe] < remaining {
                position = probe
                remaining -= self.nodes[probe]
            }
            step //= 2
        }
        return position
    }

    #: Every element, in index order.
    pub fn to_list(self) -> list[int] {
        let values: list[int] = []
        var running = 0
        for index in 1..=self.size {
            let total = self.prefix_sum(index)
            values.push(total - running)
            running = total
        }
        return values
    }

    fn __len__(self) -> int { return self.size }

    fn __str__(self) -> str {
        return f"FenwickTree({self.to_list()})"
    }
}
