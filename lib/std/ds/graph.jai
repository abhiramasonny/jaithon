#: Graphs and disjoint sets.
#:
#: `Graph` covers directed/undirected and weighted/unweighted with one type:
#: unweighted means weight 1.0, undirected means each edge is stored twice.
#:
#: Unreachability is *absence*, never a sentinel: a node missing from a
#: distance map cannot be reached from the source.
#:
#: Every traversal is iterative — a graph is not a balanced tree, and a
#: million-node path graph would blow a recursive walk's stack.
from std.collections import IndexedHeap

const FLOW_EPSILON = 1e-9

#: A weighted, directed arc between two nodes.
#:
#: An undirected graph reports each of its edges once, oriented from whichever
#: endpoint entered the graph first.
pub class Edge {
    #: The node the arc leaves.
    pub let source: any
    #: The node the arc enters.
    pub let target: any
    #: The cost, capacity or length of the arc; 1.0 when unweighted.
    pub let weight: float

    #: Create an arc from `source` to `target`.
    pub fn init(self, source: any, target: any, weight: float) {
        self.source = source
        self.target = target
        self.weight = weight
    }

    fn __str__(self) -> str {
        return f"Edge({self.source} -> {self.target}, {self.weight})"
    }
}

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
        # Second pass instead of recursion: the chain can be long before the
        # first compression, and this keeps the stack flat.
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

#: A graph with adjacency-list storage, directed or undirected, weighted or not.
#:
#: Nodes are any hashable value. There is at most one edge between an ordered
#: pair of nodes; adding an edge that already exists replaces its weight.
#:
#: ```
#: let graph = Graph()
#: graph.add_edge("a", "b", 4.0)
#: graph.add_edge("b", "c", 1.0)
#: graph.add_edge("a", "c", 9.0)
#: graph.neighbors("a")             # ["b", "c"]
#: graph.degree("b")                # 2
#: graph.shortest_path("a", "c")    # (5.0, ["a", "b", "c"])
#: ```
pub class Graph {
    let directed: bool
    let order: list[any]
    let slots: dict[any, int]
    let adjacency: dict[any, list[Edge]]

    #: Create an empty graph, undirected unless `directed` is true.
    pub fn init(self, directed: bool = false) {
        self.directed = directed
        self.order = []
        self.slots = {}
        self.adjacency = {}
    }

    #: Build a graph from `(source, target)` pairs, each of weight 1.0.
    #:
    #: ```
    #: let graph = Graph.from_edges([("a", "b"), ("b", "c")], directed: true)
    #: graph.topological_sort()   # ["a", "b", "c"]
    #: ```
    pub static fn from_edges(pairs: list[tuple[any, any]], directed: bool = false) -> Graph {
        let graph = Graph(directed)
        for (source, target) in pairs { graph.add_edge(source, target) }
        return graph
    }

    #: Report whether edges are one-way.
    pub fn is_directed(self) -> bool {
        return self.directed
    }

    #: Number of nodes.
    pub fn node_count(self) -> int {
        return self.order.len()
    }

    #: Number of edges, counting an undirected edge once.
    pub fn edge_count(self) -> int {
        var total = 0
        for node in self.order { total += self.adjacency[node].len() }
        if self.directed { return total }
        var loops = 0
        for node in self.order {
            for edge in self.adjacency[node] {
                if edge.target == node { loops += 1 }
            }
        }
        return (total - loops) // 2 + loops
    }

    #: Every node, in the order it was first added.
    pub fn nodes(self) -> list[any] {
        return [node for node in self.order]
    }

    #: Every edge, an undirected edge appearing once.
    pub fn edges(self) -> list[Edge] {
        let result: list[Edge] = []
        for (position, node) in self.order.enumerate() {
            for edge in self.adjacency[node] {
                # Undirected edges are stored twice; keep the copy whose target
                # is not earlier in insertion order so each pair is emitted once.
                if self.directed or self.slots[edge.target] >= position {
                    result.push(edge)
                }
            }
        }
        return result
    }

    #: Add `node`, returning false when it was already present.
    pub fn add_node(self, node: any) -> bool {
        if self.slots.has(node) { return false }
        self.slots[node] = self.order.len()
        self.order.push(node)
        self.adjacency[node] = []
        return true
    }

    #: Report whether `node` is in the graph.
    pub fn has_node(self, node: any) -> bool {
        return self.slots.has(node)
    }

    #: Connect `source` to `target`, adding either node if it is missing.
    #:
    #: Re-adding an existing edge replaces its weight.
    pub fn add_edge(self, source: any, target: any, weight: float = 1.0) -> void {
        self.add_node(source)
        self.add_node(target)
        self._put(source, target, weight)
        if not self.directed and source != target {
            self._put(target, source, weight)
        }
    }

    #: Remove the edge from `source` to `target`, returning false when absent.
    pub fn remove_edge(self, source: any, target: any) -> bool {
        self._require_node(source)
        self._require_node(target)
        let removed = self._drop(source, target)
        if not self.directed and source != target { self._drop(target, source) }
        return removed
    }

    #: Report whether an edge runs from `source` to `target`.
    pub fn has_edge(self, source: any, target: any) -> bool {
        return self.edge_weight(source, target) != null
    }

    #: The weight of the edge from `source` to `target`, or null when absent.
    pub fn edge_weight(self, source: any, target: any) -> float? {
        self._require_node(source)
        self._require_node(target)
        for edge in self.adjacency[source] {
            if edge.target == target { return edge.weight }
        }
        return null
    }

    #: The nodes reachable from `node` by one edge, in insertion order.
    pub fn neighbors(self, node: any) -> list[any] {
        self._require_node(node)
        let result: list[any] = []
        for edge in self.adjacency[node] { result.push(edge.target) }
        return result
    }

    #: The edges leaving `node`, in insertion order.
    pub fn edges_from(self, node: any) -> list[Edge] {
        self._require_node(node)
        return [edge for edge in self.adjacency[node]]
    }

    #: Number of edges leaving `node`; for an undirected graph, its degree.
    pub fn degree(self, node: any) -> int {
        self._require_node(node)
        return self.adjacency[node].len()
    }

    #: Number of edges entering `node`.
    #:
    #: Adjacency lists only record outgoing edges, so this scans the whole graph
    #: in O(V + E). Build the reverse graph once instead of calling it per node.
    pub fn in_degree(self, node: any) -> int {
        self._require_node(node)
        if not self.directed { return self.adjacency[node].len() }
        var total = 0
        for other in self.order {
            for edge in self.adjacency[other] {
                if edge.target == node { total += 1 }
            }
        }
        return total
    }

    #: Nodes in breadth-first order from `start`, `start` first.
    #:
    #: ```
    #: let graph = Graph.from_edges([("a", "b"), ("a", "c"), ("b", "d")])
    #: graph.bfs("a")   # ["a", "b", "c", "d"]
    #: ```
    pub fn bfs(self, start: any) -> list[any] {
        self._require_node(start)
        let visited: set[any] = set()
        visited.add(start)
        let order: list[any] = [start]
        # A list plus a read cursor is a queue with O(1) amortised dequeue; the
        # discarded prefix is bounded by the number of nodes.
        var cursor = 0
        while cursor < order.len() {
            let node = order[cursor]
            cursor += 1
            for edge in self.adjacency[node] {
                if not visited.has(edge.target) {
                    visited.add(edge.target)
                    order.push(edge.target)
                }
            }
        }
        return order
    }

    #: Nodes in depth-first order from `start`, `start` first.
    #:
    #: ```
    #: let graph = Graph.from_edges([("a", "b"), ("a", "c"), ("b", "d")])
    #: graph.dfs("a")   # ["a", "b", "d", "c"]
    #: ```
    pub fn dfs(self, start: any) -> list[any] {
        self._require_node(start)
        let visited: set[any] = set()
        let order: list[any] = []
        let stack: list[any] = [start]
        while stack.len() > 0 {
            let node = stack.pop()
            if visited.has(node) { continue }
            visited.add(node)
            order.push(node)
            # Reversed so the first neighbour is popped first, matching the
            # order a recursive walk would visit them in.
            for edge in self.adjacency[node].reversed() {
                if not visited.has(edge.target) { stack.push(edge.target) }
            }
        }
        return order
    }

    #: Report whether `target` is reachable from `source`.
    pub fn has_path(self, source: any, target: any) -> bool {
        self._require_node(source)
        self._require_node(target)
        if source == target { return true }
        for node in self.bfs(source) {
            if node == target { return true }
        }
        return false
    }

    #: The connected components, each a list of nodes in insertion order.
    #:
    #: For a directed graph these are the *weakly* connected components: edge
    #: direction is ignored. Use `strongly_connected_components` otherwise.
    pub fn connected_components(self) -> list[list[any]] {
        let forest = DisjointSetForest()
        for node in self.order { forest.add(node) }
        for edge in self.edges() { forest.union(edge.source, edge.target) }
        let groups: dict[any, list[any]] = {}
        let result: list[list[any]] = []
        for node in self.order {
            let root = forest.find(node)
            if not groups.has(root) {
                groups[root] = []
                result.push(groups[root])
            }
            groups[root].push(node)
        }
        return result
    }

    #: The strongly connected components of a directed graph, by Tarjan.
    #:
    #: Each component is a maximal set of nodes that can all reach each other, in
    #: reverse topological order of the condensation. O(V + E).
    #:
    #: ```
    #: let graph = Graph(directed: true)
    #: graph.add_edge("a", "b")
    #: graph.add_edge("b", "a")
    #: graph.add_edge("b", "c")
    #: graph.strongly_connected_components()   # [["c"], ["b", "a"]]
    #: ```
    pub fn strongly_connected_components(self) -> list[list[any]] {
        if not self.directed {
            return self.connected_components()
        }
        let count = self.order.len()
        let adjacency = self._adjacency_slots()
        let index: list[int] = []
        let low: list[int] = []
        let on_stack: list[bool] = []
        for _slot in 0..count {
            index.push(-1)
            low.push(0)
            on_stack.push(false)
        }
        var next_index = 0
        let pending: list[int] = []
        let components: list[list[any]] = []
        for start in 0..count {
            if index[start] != -1 { continue }
            # Each frame is (node, position of the next neighbour to explore),
            # which is exactly the state a recursive Tarjan keeps on the stack.
            let work: list[tuple[int, int]] = [(start, 0)]
            while work.len() > 0 {
                let frame = work[work.len() - 1]
                let node = frame[0]
                var position = frame[1]
                if position == 0 {
                    index[node] = next_index
                    low[node] = next_index
                    next_index += 1
                    pending.push(node)
                    on_stack[node] = true
                }
                var descended = false
                while position < adjacency[node].len() {
                    let child = adjacency[node][position]
                    position += 1
                    if index[child] == -1 {
                        work[work.len() - 1] = (node, position)
                        work.push((child, 0))
                        descended = true
                        break
                    } elif on_stack[child] {
                        low[node] = min(low[node], index[child])
                    }
                }
                if descended { continue }
                work.pop()
                if work.len() > 0 {
                    let parent = work[work.len() - 1][0]
                    low[parent] = min(low[parent], low[node])
                }
                if low[node] == index[node] {
                    let component: list[any] = []
                    loop {
                        let member = pending.pop()
                        on_stack[member] = false
                        component.push(self.order[member])
                        if member == node { break }
                    }
                    components.push(component)
                }
            }
        }
        return components
    }

    #: The nodes of a directed acyclic graph in dependency order, by Kahn.
    #:
    #: Raises `ValueError` when the graph is undirected or contains a cycle.
    pub fn topological_sort(self) -> list[any] {
        if not self.directed {
            throw ValueError("topological_sort requires a directed graph")
        }
        let ordered = self._kahn()
        let count = self.order.len()
        if ordered.len() != count {
            throw ValueError(
                f"graph contains a cycle: {count - ordered.len()} of {count} nodes are unordered"
            )
        }
        return ordered
    }

    #: Report whether the graph contains a cycle.
    #:
    #: A self-loop is a cycle. In an undirected graph a single edge traversed
    #: back and forth is not.
    pub fn has_cycle(self) -> bool {
        if self.directed {
            return self._kahn().len() != self.order.len()
        }
        let forest = DisjointSetForest()
        for node in self.order { forest.add(node) }
        for edge in self.edges() {
            if edge.source == edge.target { return true }
            if not forest.union(edge.source, edge.target) { return true }
        }
        return false
    }

    #: Report whether the nodes can be two-coloured with no edge inside a colour.
    #:
    #: Bipartiteness is a property of the underlying undirected graph — colouring
    #: crosses edges both directions even when directed. A self-loop is non-bipartite.
    #:
    #: ```
    #: Graph.from_edges([("a", "b"), ("b", "c"), ("c", "d")]).is_bipartite()   # true
    #: Graph.from_edges([("a", "b"), ("b", "c"), ("c", "a")]).is_bipartite()   # false
    #: ```
    pub fn is_bipartite(self) -> bool {
        let around = self._undirected_neighbours()
        let colour: dict[any, int] = {}
        for start in self.order {
            if colour.has(start) { continue }
            colour[start] = 0
            let queue: list[any] = [start]
            var cursor = 0
            while cursor < queue.len() {
                let node = queue[cursor]
                cursor += 1
                for other in around[node] {
                    if not colour.has(other) {
                        colour[other] = 1 - colour[node]
                        queue.push(other)
                    } elif colour[other] == colour[node] {
                        return false
                    }
                }
            }
        }
        return true
    }

    #: The cost and node sequence of a cheapest path, or null when unreachable.
    #:
    #: Dijkstra's algorithm with an indexed heap, O((V + E) log V). Raises
    #: `ValueError` on a negative weight — use `bellman_ford` for those.
    pub fn shortest_path(self, source: any, target: any) -> tuple[float, list[any]]? {
        self._require_node(target)
        let search = self._dijkstra(source)
        let distance = search[0]
        let previous = search[1]
        if not distance.has(target) { return null }
        let path: list[any] = [target]
        var node = target
        while previous.has(node) {
            node = previous[node]
            path.push(node)
        }
        return (distance[target], path.reversed())
    }

    #: The cost of a cheapest path to every node reachable from `source`.
    #:
    #: Unreachable nodes are absent from the result.
    #:
    #: ```
    #: let graph = Graph()
    #: graph.add_edge("a", "b", 2.0)
    #: graph.add_edge("b", "c", 3.0)
    #: graph.shortest_distances("a")["c"]   # 5.0
    #: ```
    pub fn shortest_distances(self, source: any) -> dict[any, float] {
        return self._dijkstra(source)[0]
    }

    #: The cost of a cheapest path to every node, allowing negative weights.
    #:
    #: Bellman-Ford, O(V * E). Raises `ValueError` when a negative-weight cycle
    #: is reachable from `source`, because no cheapest path exists then.
    pub fn bellman_ford(self, source: any) -> dict[any, float] {
        self._require_node(source)
        if not self.directed {
            for edge in self.edges() {
                if edge.weight < 0.0 {
                    let pair = f"{edge.source}-{edge.target}"
                    throw ValueError(f"undirected edge {pair} has negative weight {edge.weight}")
                }
            }
        }
        let distance: dict[any, float] = {source: 0.0}
        let arcs = self._arcs()
        for _round in 0..self.order.len() {
            var relaxed = false
            for edge in arcs {
                if not distance.has(edge.source) { continue }
                let candidate = distance[edge.source] + edge.weight
                if not distance.has(edge.target) or candidate < distance[edge.target] {
                    distance[edge.target] = candidate
                    relaxed = true
                }
            }
            if not relaxed { return distance }
        }
        # A relaxation on the V-th pass means some path uses V edges, which can
        # only happen if it repeats a node around a negative cycle.
        for edge in arcs {
            if not distance.has(edge.source) { continue }
            if distance[edge.source] + edge.weight < distance[edge.target] {
                throw ValueError(f"negative-weight cycle reachable from {source} via {edge.source}")
            }
        }
        return distance
    }

    #: Report whether the graph contains a negative-weight cycle anywhere.
    #:
    #: Runs Bellman-Ford from a virtual source joined to every node at cost 0,
    #: which finds cycles in components no single node can reach, in O(V * E).
    pub fn has_negative_cycle(self) -> bool {
        if not self.directed {
            for edge in self.edges() {
                if edge.weight < 0.0 { return true }
            }
            return false
        }
        if self.order.len() == 0 { return false }
        let distance: dict[any, float] = {}
        for node in self.order { distance[node] = 0.0 }
        let arcs = self._arcs()
        for _round in 0..self.order.len() {
            var relaxed = false
            for edge in arcs {
                let candidate = distance[edge.source] + edge.weight
                if candidate < distance[edge.target] {
                    distance[edge.target] = candidate
                    relaxed = true
                }
            }
            # V - 1 rounds settle every acyclic path, so a relaxation left over
            # on the V-th round can only come from a cycle that keeps paying.
            if not relaxed { return false }
        }
        return true
    }

    #: All-pairs cheapest costs, as `distance[source][target]`.
    #:
    #: Floyd-Warshall, O(V^3). Unreachable pairs are absent. Raises `ValueError`
    #: when a negative cycle makes the answer meaningless.
    pub fn floyd_warshall(self) -> dict[any, dict[any, float]] {
        let distance: dict[any, dict[any, float]] = {}
        for node in self.order { distance[node] = {node: 0.0} }
        for edge in self._arcs() {
            let row = distance[edge.source]
            if not row.has(edge.target) or edge.weight < row[edge.target] {
                row[edge.target] = edge.weight
            }
        }
        for middle in self.order {
            let through = distance[middle]
            for start in self.order {
                # Skipped, not special-cased: with start == middle, row and
                # pivot are the same dict, and mutating one while iterating it is undefined.
                if start == middle { continue }
                let row = distance[start]
                if not row.has(middle) { continue }
                let prefix = row[middle]
                for (finish, suffix) in through.items() {
                    let candidate = prefix + suffix
                    if not row.has(finish) or candidate < row[finish] {
                        row[finish] = candidate
                    }
                }
            }
        }
        for node in self.order {
            if distance[node][node] < 0.0 {
                throw ValueError(f"negative-weight cycle through {node}")
            }
        }
        return distance
    }

    #: A minimum spanning forest of an undirected graph, by Kruskal.
    #:
    #: Returns one edge per tree edge, cheapest first, in O(E log E). A
    #: disconnected graph yields a forest: one tree per component.
    #:
    #: ```
    #: let graph = Graph()
    #: graph.add_edge("a", "b", 1.0)
    #: graph.add_edge("b", "c", 2.0)
    #: graph.add_edge("a", "c", 3.0)
    #: let tree = graph.minimum_spanning_tree()
    #: tree.len()          # 2
    #: tree[0].weight      # 1.0
    #: ```
    pub fn minimum_spanning_tree(self) -> list[Edge] {
        if self.directed {
            throw ValueError("minimum_spanning_tree requires an undirected graph")
        }
        let candidates = self.edges()
        # Sorts positions, not (weight, position) tuples — `<` isn't defined
        # between tuples. Stable sort keeps equal weights in insertion order, so the tree is reproducible.
        let positions: list[int] = []
        for position in 0..candidates.len() { positions.push(position) }
        let sets = UnionFind(self.order.len())
        let chosen: list[Edge] = []
        for position in sorted(positions, |index| candidates[index].weight) {
            let edge = candidates[position]
            if sets.union(self.slots[edge.source], self.slots[edge.target]) {
                chosen.push(edge)
            }
        }
        return chosen
    }

    #: A minimum spanning tree of the component containing `start`, by Prim.
    #:
    #: O(E log V) with an indexed heap. Prefer this over `minimum_spanning_tree`
    #: on a dense graph, or when only one component is of interest.
    pub fn minimum_spanning_tree_from(self, start: any) -> list[Edge] {
        if self.directed {
            throw ValueError("minimum_spanning_tree_from requires an undirected graph")
        }
        self._require_node(start)
        let inside: set[any] = set()
        inside.add(start)
        let chosen: list[Edge] = []
        let frontier = IndexedHeap[any, any]()
        let cheapest: dict[any, Edge] = {}
        self._offer(start, frontier, cheapest, inside)
        while not frontier.is_empty() {
            let (node, _priority) = frontier.pop()
            # `decrease` may leave a stale duplicate entry in the heap;
            # skipping settled nodes keeps this correct either way.
            if inside.has(node) { continue }
            inside.add(node)
            chosen.push(cheapest[node])
            self._offer(node, frontier, cheapest, inside)
        }
        return chosen
    }

    #: The maximum flow from `source` to `sink`, by Dinic.
    #:
    #: Edge weights are capacities and must be non-negative; an undirected edge is
    #: one pipe usable either direction. O(V^2 * E), O(E * sqrt(E)) on unit capacities.
    #:
    #: ```
    #: let graph = Graph(directed: true)
    #: graph.add_edge("s", "a", 3.0)
    #: graph.add_edge("s", "b", 2.0)
    #: graph.add_edge("a", "t", 2.0)
    #: graph.add_edge("b", "t", 3.0)
    #: graph.max_flow("s", "t")   # 4.0
    #: ```
    pub fn max_flow(self, source: any, sink: any) -> float {
        self._require_node(source)
        self._require_node(sink)
        if source == sink {
            throw ValueError(f"source and sink must differ, both are {source}")
        }
        let count = self.order.len()
        let target_of: list[int] = []
        let capacity: list[float] = []
        let incident: list[list[int]] = []
        for _slot in 0..count { incident.push([]) }
        for edge in self.edges() {
            if edge.weight < 0.0 {
                let arc = f"{edge.source} -> {edge.target}"
                throw ValueError(f"capacity must be non-negative, {arc} has {edge.weight}")
            }
            let from_slot = self.slots[edge.source]
            let to_slot = self.slots[edge.target]
            incident[from_slot].push(target_of.len())
            target_of.push(to_slot)
            capacity.push(edge.weight)
            incident[to_slot].push(target_of.len())
            target_of.push(from_slot)
            capacity.push(self.directed ? 0.0 : edge.weight)
        }
        let start = self.slots[source]
        let finish = self.slots[sink]
        var flow = 0.0
        loop {
            let level = self._level_graph(incident, target_of, capacity, start)
            if level[finish] < 0 { return flow }
            let progress: list[int] = []
            for _slot in 0..count { progress.push(0) }
            let path: list[int] = []
            var node = start
            loop {
                if node == finish {
                    var bottleneck = capacity[path[0]]
                    for arc in path { bottleneck = min(bottleneck, capacity[arc]) }
                    for arc in path {
                        capacity[arc] -= bottleneck
                        capacity[arc ^ 1] += bottleneck
                    }
                    flow += bottleneck
                    # Retreat to just before the first arc this augmentation
                    # saturated; everything earlier still has residual room.
                    var cut = 0
                    while cut < path.len() and capacity[path[cut]] > FLOW_EPSILON { cut += 1 }
                    node = cut == 0 ? start : target_of[path[cut - 1]]
                    while path.len() > cut { path.pop() }
                    continue
                }
                var advanced = false
                while progress[node] < incident[node].len() {
                    let arc = incident[node][progress[node]]
                    if capacity[arc] > FLOW_EPSILON and level[target_of[arc]] == level[node] + 1 {
                        path.push(arc)
                        node = target_of[arc]
                        advanced = true
                        break
                    }
                    progress[node] += 1
                }
                if advanced { continue }
                if node == start { break }
                # Dead end: no admissible arc leaves this node in this phase.
                level[node] = -1
                let arc = path.pop()
                node = target_of[arc ^ 1]
                progress[node] += 1
            }
        }
    }

    fn _put(self, source: any, target: any, weight: float) -> void {
        let edges = self.adjacency[source]
        for (position, edge) in edges.enumerate() {
            if edge.target == target {
                edges[position] = Edge(source, target, weight)
                return
            }
        }
        edges.push(Edge(source, target, weight))
    }

    fn _drop(self, source: any, target: any) -> bool {
        let kept: list[Edge] = []
        var removed = false
        for edge in self.adjacency[source] {
            if edge.target == target {
                removed = true
            } else {
                kept.push(edge)
            }
        }
        self.adjacency[source] = kept
        return removed
    }

    fn _require_node(self, node: any) -> void {
        if not self.slots.has(node) {
            throw KeyError(f"node {node} is not in the graph")
        }
    }

    #: Every arc, both directions of an undirected edge.
    fn _arcs(self) -> list[Edge] {
        let result: list[Edge] = []
        for node in self.order {
            for edge in self.adjacency[node] { result.push(edge) }
        }
        return result
    }

    #: Kahn's algorithm; the result is shorter than the graph when a cycle
    #: leaves some nodes with an in-degree that never reaches zero.
    fn _kahn(self) -> list[any] {
        let count = self.order.len()
        let adjacency = self._adjacency_slots()
        let incoming: list[int] = []
        for _slot in 0..count { incoming.push(0) }
        for slot in 0..count {
            for child in adjacency[slot] { incoming[child] += 1 }
        }
        let queue: list[int] = []
        for slot in 0..count {
            if incoming[slot] == 0 { queue.push(slot) }
        }
        let ordered: list[any] = []
        var cursor = 0
        while cursor < queue.len() {
            let node = queue[cursor]
            cursor += 1
            ordered.push(self.order[node])
            for child in adjacency[node] {
                incoming[child] -= 1
                if incoming[child] == 0 { queue.push(child) }
            }
        }
        return ordered
    }

    #: Neighbours ignoring direction, so a walk can cross an arc backwards.
    fn _undirected_neighbours(self) -> dict[any, list[any]] {
        let around: dict[any, list[any]] = {}
        for node in self.order { around[node] = [] }
        for node in self.order {
            for edge in self.adjacency[node] {
                around[node].push(edge.target)
                if self.directed { around[edge.target].push(node) }
            }
        }
        return around
    }

    fn _adjacency_slots(self) -> list[list[int]] {
        let result: list[list[int]] = []
        for node in self.order {
            let row: list[int] = []
            for edge in self.adjacency[node] { row.push(self.slots[edge.target]) }
            result.push(row)
        }
        return result
    }

    fn _dijkstra(self, source: any) -> tuple[dict[any, float], dict[any, any]] {
        self._require_node(source)
        let distance: dict[any, float] = {source: 0.0}
        let previous: dict[any, any] = {}
        let settled: set[any] = set()
        let frontier = IndexedHeap[any, any]()
        frontier.push(source, 0.0)
        while not frontier.is_empty() {
            let (node, _priority) = frontier.pop()
            settled.add(node)
            for edge in self.adjacency[node] {
                if edge.weight < 0.0 {
                    let arc = f"{node} -> {edge.target}"
                    throw ValueError(f"Dijkstra needs non-negative weights, {arc} is {edge.weight}")
                }
                if settled.has(edge.target) { continue }
                let candidate = distance[node] + edge.weight
                if not distance.has(edge.target) {
                    distance[edge.target] = candidate
                    previous[edge.target] = node
                    frontier.push(edge.target, candidate)
                } elif candidate < distance[edge.target] {
                    distance[edge.target] = candidate
                    previous[edge.target] = node
                    frontier.push_or_decrease(edge.target, candidate)
                }
            }
        }
        return (distance, previous)
    }

    fn _offer(
        self,
        node: any,
        frontier: IndexedHeap[any, any],
        cheapest: dict[any, Edge],
        inside: set[any]
    ) -> void {
        for edge in self.adjacency[node] {
            if inside.has(edge.target) { continue }
            if not cheapest.has(edge.target) {
                cheapest[edge.target] = edge
                frontier.push(edge.target, edge.weight)
            } elif edge.weight < cheapest[edge.target].weight {
                cheapest[edge.target] = edge
                frontier.push_or_decrease(edge.target, edge.weight)
            }
        }
    }

    #: Breadth-first distances in the residual graph; -1 where unreachable.
    fn _level_graph(
        self,
        incident: list[list[int]],
        target_of: list[int],
        capacity: list[float],
        start: int
    ) -> list[int] {
        let level: list[int] = []
        for _slot in 0..incident.len() { level.push(-1) }
        level[start] = 0
        let queue: list[int] = [start]
        var cursor = 0
        while cursor < queue.len() {
            let node = queue[cursor]
            cursor += 1
            for arc in incident[node] {
                let next = target_of[arc]
                if capacity[arc] > FLOW_EPSILON and level[next] < 0 {
                    level[next] = level[node] + 1
                    queue.push(next)
                }
            }
        }
        return level
    }

    fn __len__(self) -> int { return self.order.len() }

    fn __contains__(self, node: any) -> bool { return self.slots.has(node) }

    fn __str__(self) -> str {
        let kind = self.directed ? "digraph" : "graph"
        return f"Graph({kind}, {self.order.len()} nodes, {self.edge_count()} edges)"
    }
}
