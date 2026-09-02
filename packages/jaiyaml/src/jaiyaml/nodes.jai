#: The representation graph: what the composer builds and the constructor
#: reads, and what the representer builds and the serializer walks.
#:
#: One `Node` class with a `NodeKind`. A scalar keeps its text in `value`, a
#: sequence its children in `items`, a mapping its key-value pairs in
#: `pairs`. The three static constructors mirror PyYAML's `ScalarNode`,
#: `SequenceNode` and `MappingNode`.
from .error import Mark

#: Which shape a node has.
pub enum NodeKind { Scalar, Sequence, Mapping }

#: One node of the representation graph.
pub class Node {
    #: Which shape this is.
    pub let kind: NodeKind

    #: The resolved tag, such as `tag:yaml.org,2002:int`.
    pub var tag: str

    #: A scalar's text; empty for collections.
    pub var value: str

    #: A sequence's children; empty for the other kinds.
    pub var items: list[Node]

    #: A mapping's pairs in document order; empty for the other kinds.
    pub var pairs: list[tuple[Node, Node]]

    #: Where the node starts, when it came from text.
    pub var start_mark: Mark?

    #: Where the node ends, when it came from text.
    pub var end_mark: Mark?

    #: A scalar's style: `'`, `"`, `|`, `>`, or null for plain.
    pub var style: str?

    #: Whether a collection was, or should be, written in flow style.
    pub var flow_style: bool?

    #: A bare node of `kind` with `tag`; use the static constructors.
    pub fn init(self, kind: NodeKind, tag: str) {
        self.kind = kind
        self.tag = tag
        self.value = ""
        self.items = []
        self.pairs = []
        self.start_mark = null
        self.end_mark = null
        self.style = null
        self.flow_style = null
    }

    #: A scalar node.
    pub static fn scalar(
        tag: str,
        value: str,
        start_mark: Mark? = null,
        end_mark: Mark? = null,
        style: str? = null
    ) -> Node {
        let node = Node(NodeKind.Scalar, tag)
        node.value = value
        node.start_mark = start_mark
        node.end_mark = end_mark
        node.style = style
        return node
    }

    #: A sequence node.
    pub static fn sequence(
        tag: str,
        items: list[Node],
        start_mark: Mark? = null,
        end_mark: Mark? = null,
        flow_style: bool? = null
    ) -> Node {
        let node = Node(NodeKind.Sequence, tag)
        node.items = items
        node.start_mark = start_mark
        node.end_mark = end_mark
        node.flow_style = flow_style
        return node
    }

    #: A mapping node.
    pub static fn mapping(
        tag: str,
        pairs: list[tuple[Node, Node]],
        start_mark: Mark? = null,
        end_mark: Mark? = null,
        flow_style: bool? = null
    ) -> Node {
        let node = Node(NodeKind.Mapping, tag)
        node.pairs = pairs
        node.start_mark = start_mark
        node.end_mark = end_mark
        node.flow_style = flow_style
        return node
    }

    #: PyYAML's `id`: `scalar`, `sequence` or `mapping`.
    pub fn id(self) -> str {
        return match self.kind {
            NodeKind.Scalar => "scalar",
            NodeKind.Sequence => "sequence",
            NodeKind.Mapping => "mapping",
        }
    }

    fn __str__(self) -> str {
        if self.kind == NodeKind.Scalar { return f"Node({self.tag}, {repr(self.value)})" }
        if self.kind == NodeKind.Sequence { return f"Node({self.tag}, {self.items.len()} items)" }
        return f"Node({self.tag}, {self.pairs.len()} pairs)"
    }
}
