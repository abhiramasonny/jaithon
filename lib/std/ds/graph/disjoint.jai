#: Disjoint sets: the integer-slot `UnionFind` and the value-keyed forest
#: built on it.

#: A union-find structure over the integers `0..size` with path compression and
#: union by rank.
#:
#: Both operations run in inverse-Ackermann amortised time, which is under 5 for
#: any input that fits in memory.
#:
#: ```
#: let sets = UnionFind(5)
#: sets.union(0, 1)      # true
#: sets.union(1, 2)      # true
#: sets.union(0, 2)      # false
#: sets.connected(0, 2)  # true
#: sets.count()          # 3
#: ```
pub class UnionFind {
    let parent: list[int]
    let rank: list[int]
    let sizes: list[int]
    var groups: int

    #: Create `size` singleton sets.
    pub fn init(self, size: int) {
        if size < 0 {
            throw ValueError(f"size must be non-negative, got {size}")
        }
        let parent: list[int] = []
        let rank: list[int] = []
        let sizes: list[int] = []
        for element in 0..size {
            parent.push(element)
            rank.push(0)
            sizes.push(1)
        }
        self.parent = parent
        self.rank = rank
        self.sizes = sizes
        self.groups = size
    }

    #: Number of elements.
    pub fn len(self) -> int {
        return self.parent.len()
    }

    #: Number of disjoint sets.
    pub fn count(self) -> int {
        return self.groups
    }

    #: Append a new singleton set and return its element index.
    pub fn add(self) -> int {
        let element = self.parent.len()
        self.parent.push(element)
        self.rank.push(0)
        self.sizes.push(1)
        self.groups += 1
        return element
    }

    #: The representative of the set containing `element`.
    pub fn find(self, element: int) -> int {
        self._check(element)
        var root = element
        while self.parent[root] != root { root = self.parent[root] }
        var walk = element
        while self.parent[walk] != root {
            let next = self.parent[walk]
            self.parent[walk] = root
            walk = next
        }
        return root
    }

    #: Merge the sets containing `a` and `b`, returning false when already merged.
    pub fn union(self, a: int, b: int) -> bool {
        let left = self.find(a)
        let right = self.find(b)
        if left == right { return false }
        var high = left
        var low = right
        if self.rank[high] < self.rank[low] {
            high = right
            low = left
        }
        self.parent[low] = high
        self.sizes[high] += self.sizes[low]
        if self.rank[high] == self.rank[low] { self.rank[high] += 1 }
        self.groups -= 1
        return true
    }

    #: Report whether `a` and `b` are in the same set.
    pub fn connected(self, a: int, b: int) -> bool {
        return self.find(a) == self.find(b)
    }

    #: Number of elements in the set containing `element`.
    pub fn component_size(self, element: int) -> int {
        return self.sizes[self.find(element)]
    }

    #: The members of every set, each group in ascending element order.
    pub fn components(self) -> list[list[int]] {
        let groups: dict[int, list[int]] = {}
        for element in 0..self.parent.len() {
            let root = self.find(element)
            if not groups.has(root) { groups[root] = [] }
            groups[root].push(element)
        }
        let result: list[list[int]] = []
        for element in 0..self.parent.len() {
            if groups.has(element) { result.push(groups[element]) }
        }
        return result
    }

    fn _check(self, element: int) -> void {
        if element < 0 or element >= self.parent.len() {
            throw IndexError(f"element {element} is not within [0, {self.parent.len()})")
        }
    }

    fn __len__(self) -> int { return self.parent.len() }
}

#: Summary statistics of a `DisjointSetForest`.
pub class ForestStatistics {
    #: Total number of elements across all sets.
    pub let element_count: int
    #: Number of disjoint sets.
    pub let component_count: int
    #: Size of the largest set, or 0 when there are none.
    pub let largest: int
    #: Size of the smallest set, or 0 when there are none.
    pub let smallest: int
    #: Mean set size, or 0.0 when there are no sets.
    pub let mean_size: float

    #: Summarise a list of set sizes.
    pub fn init(self, sizes: list[int]) {
        var total = 0
        var largest = 0
        var smallest = 0
        for (position, size) in sizes.enumerate() {
            total += size
            if size > largest { largest = size }
            if position == 0 or size < smallest { smallest = size }
        }
        self.element_count = total
        self.component_count = sizes.len()
        self.largest = largest
        self.smallest = smallest
        self.mean_size = sizes.len() == 0 ? 0.0 : float(total) / float(sizes.len())
    }

    fn __str__(self) -> str {
        return f"ForestStatistics({self.element_count} elements, {self.component_count} sets)"
    }
}

#: A union-find over arbitrary hashable values rather than integer slots.
#:
#: Elements are added on demand, so this is the form to reach for when merging
#: names, coordinates or graph nodes.
#:
#: ```
#: let forest = DisjointSetForest()
#: forest.union("ada", "grace")
#: forest.union("alan", "alonzo")
#: forest.connected("ada", "grace")   # true
#: forest.connected("ada", "alan")    # false
#: forest.statistics().largest        # 2
#: ```
pub class DisjointSetForest {
    let slots: dict[any, int]
    let elements: list[any]
    let sets: UnionFind

    #: Create an empty forest.
    pub fn init(self) {
        self.slots = {}
        self.elements = []
        self.sets = UnionFind(0)
    }

    #: Number of elements.
    pub fn len(self) -> int {
        return self.elements.len()
    }

    #: Number of disjoint sets.
    pub fn count(self) -> int {
        return self.sets.count()
    }

    #: Add `element` as its own singleton set, returning false when already present.
    pub fn add(self, element: any) -> bool {
        if self.slots.has(element) { return false }
        self.slots[element] = self.sets.add()
        self.elements.push(element)
        return true
    }

    #: The representative element of the set containing `element`.
    pub fn find(self, element: any) -> any {
        self._require(element)
        return self.elements[self.sets.find(self.slots[element])]
    }

    #: Merge the sets containing `a` and `b`, returning false when already merged.
    #:
    #: Both elements are added first if they are not already present.
    pub fn union(self, a: any, b: any) -> bool {
        self.add(a)
        self.add(b)
        return self.sets.union(self.slots[a], self.slots[b])
    }

    #: Report whether `a` and `b` are in the same set.
    pub fn connected(self, a: any, b: any) -> bool {
        self._require(a)
        self._require(b)
        return self.sets.connected(self.slots[a], self.slots[b])
    }

    #: Number of elements in the set containing `element`.
    pub fn component_size(self, element: any) -> int {
        self._require(element)
        return self.sets.component_size(self.slots[element])
    }

    #: The members of every set.
    pub fn components(self) -> list[list[any]] {
        let result: list[list[any]] = []
        for group in self.sets.components() {
            let members: list[any] = []
            for slot in group { members.push(self.elements[slot]) }
            result.push(members)
        }
        return result
    }

    #: Element and component counts, and the largest, smallest and mean set size.
    pub fn statistics(self) -> ForestStatistics {
        let sizes: list[int] = []
        for group in self.sets.components() { sizes.push(group.len()) }
        return ForestStatistics(sizes)
    }

    fn _require(self, element: any) -> void {
        if not self.slots.has(element) {
            throw KeyError(f"element {element} is not in the forest")
        }
    }

    fn __len__(self) -> int { return self.elements.len() }
}
