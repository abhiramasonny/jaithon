#: The events the parser produces and the emitter consumes.
#:
#: One `Event` class with an `EventKind`, for the same reason `Token` is one
#: class. The scalar's two implicit flags -- PyYAML's `implicit` tuple -- are
#: `implicit` and `implicit_quoted`: the first says a plain rendering may go
#: untagged, the second says a quoted rendering may. Collections only use
#: `implicit`.
from .error import Mark

#: Which event this is.
pub enum EventKind {
    StreamStart,
    StreamEnd,
    DocumentStart,
    DocumentEnd,
    Alias,
    Scalar,
    SequenceStart,
    SequenceEnd,
    MappingStart,
    MappingEnd,
}

#: One parsing or emitting event.
pub class Event {
    #: Which kind this is.
    pub let kind: EventKind

    #: Where the event starts; null for events the serializer makes.
    pub var start_mark: Mark?

    #: Where the event ends; null for events the serializer makes.
    pub var end_mark: Mark?

    #: The anchor on a node, or the alias's target.
    pub var anchor: str?

    #: The node's tag, resolved through the tag handles; null when none.
    pub var tag: str?

    #: Whether an untagged plain scalar, or an untagged collection, is fine.
    pub var implicit: bool

    #: Whether an untagged quoted scalar is fine.
    pub var implicit_quoted: bool

    #: The scalar's text.
    pub var value: str

    #: The scalar's style: `'`, `"`, `|`, `>`, or null for plain.
    pub var style: str?

    #: Whether a collection is written in flow style; null leaves the emitter
    #: to choose.
    pub var flow_style: bool?

    #: Whether a document had, or should have, its `---` or `...` written.
    pub var explicit: bool

    #: A document's `%YAML` version.
    pub var version: tuple[int, int]?

    #: A document's `%TAG` handles.
    pub var tags: dict[str, str]?

    #: An event of `kind` between the two marks.
    pub fn init(self, kind: EventKind, start_mark: Mark? = null, end_mark: Mark? = null) {
        self.kind = kind
        self.start_mark = start_mark
        self.end_mark = end_mark
        self.anchor = null
        self.tag = null
        self.implicit = false
        self.implicit_quoted = false
        self.value = ""
        self.style = null
        self.flow_style = null
        self.explicit = false
        self.version = null
        self.tags = null
    }

    #: Whether this opens a sequence or a mapping.
    pub fn is_collection_start(self) -> bool {
        return self.kind == EventKind.SequenceStart or self.kind == EventKind.MappingStart
    }

    #: Whether this closes a sequence or a mapping.
    pub fn is_collection_end(self) -> bool {
        return self.kind == EventKind.SequenceEnd or self.kind == EventKind.MappingEnd
    }

    #: Whether this names a node: an alias, a scalar or a collection start.
    pub fn is_node(self) -> bool {
        return self.kind == EventKind.Alias or
            self.kind == EventKind.Scalar or
            self.is_collection_start()
    }

    fn __str__(self) -> str {
        if self.kind == EventKind.Scalar { return f"Event({self.kind}, {repr(self.value)})" }
        return f"Event({self.kind})"
    }
}
