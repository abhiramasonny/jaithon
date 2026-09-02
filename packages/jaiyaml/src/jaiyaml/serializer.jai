#: Nodes to events, with anchors for the nodes that appear more than once.
#:
#: PyYAML's serializer: a first walk counts how often each node object is
#: reached and gives every repeated one an anchor named `id001`, `id002`
#: and so on, in the order the repeats are found; a second walk emits
#: events, an alias for every repeat. The implicit flags on each scalar are
#: decided here by asking the resolver whether the node's tag is what the
#: text would resolve to anyway.
from .emitter import Emitter
from .error import SerializerError
from .events import Event, EventKind
from .nodes import Node, NodeKind
from .resolver import resolve

#: Turns node graphs into a stream of events for an `Emitter`.
pub class Serializer {
    #: The emitter the events go to.
    pub let emitter: Emitter

    let use_explicit_start: bool
    let use_explicit_end: bool
    let use_version: tuple[int, int]?
    let use_tags: dict[str, str]?
    var serialized_nodes: dict[int, bool]
    var anchors: dict[int, str?]
    var last_anchor_id: int
    var closed: bool?

    #: Serialize onto `emitter` with the given document decorations.
    pub fn init(
        self,
        emitter: Emitter,
        explicit_start: bool = false,
        explicit_end: bool = false,
        version: tuple[int, int]? = null,
        tags: dict[str, str]? = null
    ) {
        self.emitter = emitter
        self.use_explicit_start = explicit_start
        self.use_explicit_end = explicit_end
        self.use_version = version
        self.use_tags = tags
        self.serialized_nodes = {}
        self.anchors = {}
        self.last_anchor_id = 0
        self.closed = null
    }

    #: Emit STREAM-START; must come before the first `serialize`.
    pub fn open(self) -> void {
        if self.closed is null {
            self.emitter.emit(Event(EventKind.StreamStart))
            self.closed = false
        } elif self.closed ?? false {
            throw SerializerError("serializer is closed")
        } else {
            throw SerializerError("serializer is already opened")
        }
    }

    #: Emit STREAM-END; nothing may follow.
    pub fn close(self) -> void {
        if self.closed is null {
            throw SerializerError("serializer is not opened")
        } elif not (self.closed ?? false) {
            self.emitter.emit(Event(EventKind.StreamEnd))
            self.closed = true
        }
    }

    #: Emit one document with `node` at its root.
    pub fn serialize(self, node: Node) -> void {
        if self.closed is null {
            throw SerializerError("serializer is not opened")
        } elif self.closed ?? false {
            throw SerializerError("serializer is closed")
        }
        let start = Event(EventKind.DocumentStart)
        start.explicit = self.use_explicit_start
        start.version = self.use_version
        start.tags = self.use_tags
        self.emitter.emit(start)
        self.anchor_node(node)
        self.serialize_node(node)
        let end = Event(EventKind.DocumentEnd)
        end.explicit = self.use_explicit_end
        self.emitter.emit(end)
        self.serialized_nodes = {}
        self.anchors = {}
        self.last_anchor_id = 0
    }

    fn anchor_node(self, node: Node) -> void {
        let key = id(node)
        if key in self.anchors {
            if self.anchors[key] is null { self.anchors[key] = self.generate_anchor() }
            return
        }
        self.anchors[key] = null
        if node.kind == NodeKind.Sequence {
            for item in node.items { self.anchor_node(item) }
        } elif node.kind == NodeKind.Mapping {
            for (item_key, item_value) in node.pairs {
                self.anchor_node(item_key)
                self.anchor_node(item_value)
            }
        }
    }

    fn generate_anchor(self) -> str {
        self.last_anchor_id += 1
        var number = str(self.last_anchor_id)
        while number.len() < 3 { number = "0" + number }
        return "id" + number
    }

    fn serialize_node(self, node: Node) -> void {
        let key = id(node)
        let alias = self.anchors[key]
        if key in self.serialized_nodes {
            let event = Event(EventKind.Alias)
            event.anchor = alias
            self.emitter.emit(event)
            return
        }
        self.serialized_nodes[key] = true
        if node.kind == NodeKind.Scalar {
            let detected_tag = resolve(NodeKind.Scalar, node.value, true)
            let default_tag = resolve(NodeKind.Scalar, node.value, false)
            let event = Event(EventKind.Scalar)
            event.anchor = alias
            event.tag = node.tag
            event.implicit = node.tag == detected_tag
            event.implicit_quoted = node.tag == default_tag
            event.value = node.value
            event.style = node.style
            self.emitter.emit(event)
        } elif node.kind == NodeKind.Sequence {
            let event = Event(EventKind.SequenceStart)
            event.anchor = alias
            event.tag = node.tag
            event.implicit = node.tag == resolve(NodeKind.Sequence, null, true)
            event.flow_style = node.flow_style
            self.emitter.emit(event)
            for item in node.items { self.serialize_node(item) }
            self.emitter.emit(Event(EventKind.SequenceEnd))
        } else {
            let event = Event(EventKind.MappingStart)
            event.anchor = alias
            event.tag = node.tag
            event.implicit = node.tag == resolve(NodeKind.Mapping, null, true)
            event.flow_style = node.flow_style
            self.emitter.emit(event)
            for (item_key, item_value) in node.pairs {
                self.serialize_node(item_key)
                self.serialize_node(item_value)
            }
            self.emitter.emit(Event(EventKind.MappingEnd))
        }
    }
}
