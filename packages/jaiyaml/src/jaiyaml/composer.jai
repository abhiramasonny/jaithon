#: Events to nodes: anchors are recorded, aliases are replaced by the node
#: they name, and every untagged node is resolved.
#:
#: An alias resolves to the same `Node` object as its anchor, not a copy, so
#: a document that refers to one collection twice yields one node referred
#: to twice, and the constructor can build one value for it. The anchor
#: table is cleared between documents, as in PyYAML.
from .error import ComposerError, python_repr
from .events import Event, EventKind
from .nodes import Node, NodeKind
from .parser import Parser
from .reader import STRING_NAME
from .resolver import resolve

#: Builds the node graph of each document in a text.
pub class Composer {
    #: The parser the events come from.
    pub let parser: Parser

    var anchors: dict[str, Node]

    #: Start composing `text`.
    pub fn init(self, text: str, name: str = STRING_NAME) {
        self.parser = Parser(text, name)
        self.anchors = {}
    }

    #: Whether another document follows; consumes the STREAM-START event.
    pub fn check_node(self) -> bool {
        if self.parser.check_event(EventKind.StreamStart) { self.parser.get_event() }
        return not self.parser.check_event(EventKind.StreamEnd)
    }

    #: The root node of the next document, or null at the end of the stream.
    pub fn get_node(self) -> Node? {
        if not self.parser.check_event(EventKind.StreamEnd) { return self.compose_document() }
        return null
    }

    #: The root node of the only document, or null for an empty stream.
    #:
    #: Raises `ComposerError` when a second document follows.
    pub fn get_single_node(self) -> Node? {
        self.parser.get_event()
        var document: Node? = null
        if not self.parser.check_event(EventKind.StreamEnd) { document = self.compose_document() }
        if not self.parser.check_event(EventKind.StreamEnd) {
            let event = self.parser.get_event()
            let root = document ?? Node.scalar("", "")
            throw ComposerError(
                "expected a single document in the stream",
                root.start_mark,
                "but found another document",
                event.start_mark
            )
        }
        self.parser.get_event()
        return document
    }

    fn compose_document(self) -> Node {
        self.parser.get_event()
        let node = self.compose_node()
        self.parser.get_event()
        self.anchors = {}
        return node
    }

    fn compose_node(self) -> Node {
        if self.parser.check_event(EventKind.Alias) {
            let event = self.parser.get_event()
            let anchor = event.anchor ?? ""
            if anchor not in self.anchors {
                throw ComposerError(
                    null,
                    null,
                    f"found undefined alias {python_repr(anchor)}",
                    event.start_mark
                )
            }
            return self.anchors[anchor]
        }
        let event = self.parser.peek_event() ?? Event(EventKind.StreamEnd)
        let anchor = event.anchor
        if anchor is not null {
            let name = anchor ?? ""
            if name in self.anchors {
                throw ComposerError(
                    f"found duplicate anchor {python_repr(name)}; first occurrence",
                    self.anchors[name].start_mark,
                    "second occurrence",
                    event.start_mark
                )
            }
        }
        if self.parser.check_event(EventKind.Scalar) { return self.compose_scalar_node(anchor) }
        if self.parser.check_event(EventKind.SequenceStart) {
            return self.compose_sequence_node(anchor)
        }
        return self.compose_mapping_node(anchor)
    }

    fn compose_scalar_node(self, anchor: str?) -> Node {
        let event = self.parser.get_event()
        var tag = event.tag
        if tag is null or (tag ?? "") == "!" {
            tag = resolve(NodeKind.Scalar, event.value, event.implicit)
        }
        let node = Node.scalar(
            tag ?? "",
            event.value,
            event.start_mark,
            event.end_mark,
            event.style
        )
        if anchor is not null { self.anchors[anchor ?? ""] = node }
        return node
    }

    fn compose_sequence_node(self, anchor: str?) -> Node {
        let start_event = self.parser.get_event()
        var tag = start_event.tag
        if tag is null or (tag ?? "") == "!" {
            tag = resolve(NodeKind.Sequence, null, start_event.implicit)
        }
        let node = Node.sequence(
            tag ?? "",
            [],
            start_event.start_mark,
            null,
            start_event.flow_style
        )
        if anchor is not null { self.anchors[anchor ?? ""] = node }
        while not self.parser.check_event(EventKind.SequenceEnd) {
            node.items.push(self.compose_node())
        }
        let end_event = self.parser.get_event()
        node.end_mark = end_event.end_mark
        return node
    }

    fn compose_mapping_node(self, anchor: str?) -> Node {
        let start_event = self.parser.get_event()
        var tag = start_event.tag
        if tag is null or (tag ?? "") == "!" {
            tag = resolve(NodeKind.Mapping, null, start_event.implicit)
        }
        let node = Node.mapping(tag ?? "", [], start_event.start_mark, null, start_event.flow_style)
        if anchor is not null { self.anchors[anchor ?? ""] = node }
        while not self.parser.check_event(EventKind.MappingEnd) {
            let item_key = self.compose_node()
            let item_value = self.compose_node()
            node.pairs.push((item_key, item_value))
        }
        let end_event = self.parser.get_event()
        node.end_mark = end_event.end_mark
        return node
    }
}
