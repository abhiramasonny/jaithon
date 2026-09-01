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
#:
#: Facade: imports and re-exports every sibling module in `ds/graph/`.
from std.ds.graph.adjacency import Edge, Graph
from std.ds.graph.disjoint import DisjointSetForest, ForestStatistics, UnionFind

export { Edge, Graph }
export { DisjointSetForest, ForestStatistics, UnionFind }
