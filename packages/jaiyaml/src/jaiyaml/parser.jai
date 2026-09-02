#: Tokens to events: PyYAML's LL(1) parser, state for state.
#:
#: The grammar is the one PyYAML documents at the top of its parser:
#:
#: ```
#: stream            ::= STREAM-START implicit_document? explicit_document* STREAM-END
#: implicit_document ::= block_node DOCUMENT-END*
#: explicit_document ::= DIRECTIVE* DOCUMENT-START block_node? DOCUMENT-END*
#: block_node        ::= ALIAS | properties block_content? | block_content
#: flow_node         ::= ALIAS | properties flow_content? | flow_content
#: properties        ::= TAG ANCHOR? | ANCHOR TAG?
#: block_content     ::= block_collection | flow_collection | SCALAR
#: flow_content      ::= flow_collection | SCALAR
#: ```
#:
#: Rather than recursing, the parser keeps its continuation as a bound
#: method in `state` and a stack of them in `states`, exactly as PyYAML does,
#: so a document nested a thousand deep costs no call stack.
from .error import Mark, ParserError
from .events import Event, EventKind
from .reader import STRING_NAME
from .scanner import Scanner
from .tokens import Token, TokenKind

let DEFAULT_TAGS: dict[str, str] = {"!": "!", "!!": "tag:yaml.org,2002:"}

#: Produces the event stream for one text.
pub class Parser {
    #: The scanner the tokens come from.
    pub let scanner: Scanner

    var current_event: Event?
    var yaml_version: tuple[int, int]?
    var tag_handles: dict[str, str]
    var states: list[fn() -> Event]
    var marks: list[Mark]
    var state: (fn() -> Event)?

    #: Start parsing `text`.
    pub fn init(self, text: str, name: str = STRING_NAME) {
        self.scanner = Scanner(text, name)
        self.current_event = null
        self.yaml_version = null
        self.tag_handles = {}
        self.states = []
        self.marks = []
        self.state = self.parse_stream_start
    }

    #: Whether the next event is one of `kinds`, or exists at all when none
    #: are named.
    pub fn check_event(self, ...kinds: EventKind) -> bool {
        if self.current_event is null { self.current_event = self.produce() }
        let event = self.current_event
        if event is null { return false }
        if kinds.len() == 0 { return true }
        let head = (event ?? Event(EventKind.StreamEnd)).kind
        for kind in kinds {
            if head == kind { return true }
        }
        return false
    }

    #: The next event without consuming it, or null at the end.
    pub fn peek_event(self) -> Event? {
        if self.current_event is null { self.current_event = self.produce() }
        return self.current_event
    }

    #: Consume and return the next event.
    pub fn get_event(self) -> Event {
        if self.current_event is null { self.current_event = self.produce() }
        let event = self.current_event
        self.current_event = null
        if event is null { throw ParserError(null, null, "no more events", null) }
        return event ?? Event(EventKind.StreamEnd)
    }

    fn produce(self) -> Event? {
        let step = self.state
        if step is null { return null }
        let run = step ?? self.parse_stream_start
        return run()
    }

    fn next_token(self) -> Token { return self.scanner.get_token() }

    fn look(self) -> Token {
        let token = self.scanner.peek_token()
        if token is null {
            throw ParserError(null, null, "unexpected end of the token stream", null)
        }
        return token ?? self.scanner.get_token()
    }

    fn parse_stream_start(self) -> Event {
        let token = self.next_token()
        let event = Event(EventKind.StreamStart, token.start_mark, token.end_mark)
        self.state = self.parse_implicit_document_start
        return event
    }

    fn parse_implicit_document_start(self) -> Event {
        if not self.scanner.check_token(TokenKind.Directive, TokenKind.DocumentStart, TokenKind.StreamEnd) {
            self.tag_handles = {"!": "!", "!!": "tag:yaml.org,2002:"}
            let token = self.look()
            let event = Event(EventKind.DocumentStart, token.start_mark, token.start_mark)
            event.explicit = false
            self.states.push(self.parse_document_end)
            self.state = self.parse_block_node
            return event
        }
        return self.parse_document_start()
    }

    fn parse_document_start(self) -> Event {
        while self.scanner.check_token(TokenKind.DocumentEnd) { self.next_token() }
        if not self.scanner.check_token(TokenKind.StreamEnd) {
            var token = self.look()
            let start_mark = token.start_mark
            let (version, tags) = self.process_directives()
            if not self.scanner.check_token(TokenKind.DocumentStart) {
                let found = self.look()
                throw ParserError(
                    null,
                    null,
                    f"expected '<document start>', but found {python_quote(found.id())}",
                    found.start_mark
                )
            }
            token = self.next_token()
            let event = Event(EventKind.DocumentStart, start_mark, token.end_mark)
            event.explicit = true
            event.version = version
            event.tags = tags
            self.states.push(self.parse_document_end)
            self.state = self.parse_document_content
            return event
        }
        let token = self.next_token()
        let event = Event(EventKind.StreamEnd, token.start_mark, token.end_mark)
        self.state = null
        return event
    }

    fn parse_document_end(self) -> Event {
        var token = self.look()
        let start_mark = token.start_mark
        var end_mark = token.start_mark
        var explicit = false
        if self.scanner.check_token(TokenKind.DocumentEnd) {
            token = self.next_token()
            end_mark = token.end_mark
            explicit = true
        }
        let event = Event(EventKind.DocumentEnd, start_mark, end_mark)
        event.explicit = explicit
        self.state = self.parse_document_start
        return event
    }

    fn parse_document_content(self) -> Event {
        if self.scanner.check_token(
            TokenKind.Directive,
            TokenKind.DocumentStart,
            TokenKind.DocumentEnd,
            TokenKind.StreamEnd
        ) {
            let event = self.process_empty_scalar(self.look().start_mark)
            self.state = self.states.pop()
            return event
        }
        return self.parse_block_node()
    }

    fn process_directives(self) -> tuple[tuple[int, int]?, dict[str, str]?] {
        self.yaml_version = null
        self.tag_handles = {}
        while self.scanner.check_token(TokenKind.Directive) {
            let token = self.next_token()
            if token.value == "YAML" {
                if self.yaml_version is not null {
                    throw ParserError(
                        null,
                        null,
                        "found duplicate YAML directive",
                        token.start_mark
                    )
                }
                if token.major != 1 {
                    throw ParserError(
                        null,
                        null,
                        "found incompatible YAML document (version 1.* is required)",
                        token.start_mark
                    )
                }
                self.yaml_version = (token.major, token.minor)
            } elif token.value == "TAG" {
                let handle = token.handle ?? ""
                let prefix = token.suffix ?? ""
                if handle in self.tag_handles {
                    throw ParserError(
                        null,
                        null,
                        f"duplicate tag handle {python_quote(handle)}",
                        token.start_mark
                    )
                }
                self.tag_handles[handle] = prefix
            }
        }
        var tags: dict[str, str]? = null
        if self.tag_handles.len() > 0 {
            let copy: dict[str, str] = {}
            for (handle, prefix) in self.tag_handles.items() { copy[handle] = prefix }
            tags = copy
        }
        for (handle, prefix) in DEFAULT_TAGS.items() {
            if handle not in self.tag_handles { self.tag_handles[handle] = prefix }
        }
        return (self.yaml_version, tags)
    }

    fn parse_block_node(self) -> Event { return self.parse_node(true, false) }

    fn parse_flow_node(self) -> Event { return self.parse_node(false, false) }

    fn parse_block_node_or_indentless_sequence(self) -> Event { return self.parse_node(true, true) }

    fn parse_node(self, block: bool, indentless_sequence: bool) -> Event {
        if self.scanner.check_token(TokenKind.Alias) {
            let token = self.next_token()
            let event = Event(EventKind.Alias, token.start_mark, token.end_mark)
            event.anchor = token.value
            self.state = self.states.pop()
            return event
        }
        var anchor: str? = null
        var tag: str? = null
        var tag_handle: str? = null
        var tag_suffix: str? = null
        var has_tag = false
        var start_mark: Mark? = null
        var end_mark: Mark? = null
        var tag_mark: Mark? = null
        if self.scanner.check_token(TokenKind.Anchor) {
            var token = self.next_token()
            start_mark = token.start_mark
            end_mark = token.end_mark
            anchor = token.value
            if self.scanner.check_token(TokenKind.Tag) {
                token = self.next_token()
                tag_mark = token.start_mark
                end_mark = token.end_mark
                has_tag = true
                tag_handle = token.handle
                tag_suffix = token.suffix
            }
        } elif self.scanner.check_token(TokenKind.Tag) {
            var token = self.next_token()
            start_mark = token.start_mark
            tag_mark = token.start_mark
            end_mark = token.end_mark
            has_tag = true
            tag_handle = token.handle
            tag_suffix = token.suffix
            if self.scanner.check_token(TokenKind.Anchor) {
                token = self.next_token()
                end_mark = token.end_mark
                anchor = token.value
            }
        }
        if has_tag {
            let suffix = tag_suffix ?? ""
            if tag_handle is not null {
                let handle = tag_handle ?? ""
                if handle not in self.tag_handles {
                    throw ParserError(
                        "while parsing a node",
                        start_mark,
                        f"found undefined tag handle {python_quote(handle)}",
                        tag_mark
                    )
                }
                tag = self.tag_handles[handle] + suffix
            } else {
                tag = suffix
            }
        }
        if start_mark is null {
            let token = self.look()
            start_mark = token.start_mark
            end_mark = token.start_mark
        }
        let implicit = tag is null or (tag ?? "") == "!"
        if indentless_sequence and self.scanner.check_token(TokenKind.BlockEntry) {
            end_mark = self.look().end_mark
            let event = Event(EventKind.SequenceStart, start_mark, end_mark)
            event.anchor = anchor
            event.tag = tag
            event.implicit = implicit
            self.state = self.parse_indentless_sequence_entry
            return event
        }
        if self.scanner.check_token(TokenKind.Scalar) {
            let token = self.next_token()
            end_mark = token.end_mark
            let event = Event(EventKind.Scalar, start_mark, end_mark)
            event.anchor = anchor
            event.tag = tag
            if token.plain and tag is null or (tag ?? "") == "!" {
                event.implicit = true
                event.implicit_quoted = false
            } elif tag is null {
                event.implicit = false
                event.implicit_quoted = true
            } else {
                event.implicit = false
                event.implicit_quoted = false
            }
            event.value = token.value
            event.style = token.style
            self.state = self.states.pop()
            return event
        }
        if self.scanner.check_token(TokenKind.FlowSequenceStart) {
            end_mark = self.look().end_mark
            let event = Event(EventKind.SequenceStart, start_mark, end_mark)
            event.anchor = anchor
            event.tag = tag
            event.implicit = implicit
            event.flow_style = true
            self.state = self.parse_flow_sequence_first_entry
            return event
        }
        if self.scanner.check_token(TokenKind.FlowMappingStart) {
            end_mark = self.look().end_mark
            let event = Event(EventKind.MappingStart, start_mark, end_mark)
            event.anchor = anchor
            event.tag = tag
            event.implicit = implicit
            event.flow_style = true
            self.state = self.parse_flow_mapping_first_key
            return event
        }
        if block and self.scanner.check_token(TokenKind.BlockSequenceStart) {
            end_mark = self.look().start_mark
            let event = Event(EventKind.SequenceStart, start_mark, end_mark)
            event.anchor = anchor
            event.tag = tag
            event.implicit = implicit
            event.flow_style = false
            self.state = self.parse_block_sequence_first_entry
            return event
        }
        if block and self.scanner.check_token(TokenKind.BlockMappingStart) {
            end_mark = self.look().start_mark
            let event = Event(EventKind.MappingStart, start_mark, end_mark)
            event.anchor = anchor
            event.tag = tag
            event.implicit = implicit
            event.flow_style = false
            self.state = self.parse_block_mapping_first_key
            return event
        }
        if anchor is not null or tag is not null {
            let event = Event(EventKind.Scalar, start_mark, end_mark)
            event.anchor = anchor
            event.tag = tag
            event.implicit = implicit
            event.implicit_quoted = false
            event.value = ""
            self.state = self.states.pop()
            return event
        }
        let node = block ? "block" : "flow"
        let token = self.look()
        throw ParserError(
            f"while parsing a {node} node",
            start_mark,
            f"expected the node content, but found {python_quote(token.id())}",
            token.start_mark
        )
    }

    fn parse_block_sequence_first_entry(self) -> Event {
        let token = self.next_token()
        self.marks.push(token.start_mark)
        return self.parse_block_sequence_entry()
    }

    fn parse_block_sequence_entry(self) -> Event {
        if self.scanner.check_token(TokenKind.BlockEntry) {
            let token = self.next_token()
            if not self.scanner.check_token(TokenKind.BlockEntry, TokenKind.BlockEnd) {
                self.states.push(self.parse_block_sequence_entry)
                return self.parse_block_node()
            }
            self.state = self.parse_block_sequence_entry
            return self.process_empty_scalar(token.end_mark)
        }
        if not self.scanner.check_token(TokenKind.BlockEnd) {
            let token = self.look()
            throw ParserError(
                "while parsing a block collection",
                self.marks[self.marks.len() - 1],
                f"expected <block end>, but found {python_quote(token.id())}",
                token.start_mark
            )
        }
        let token = self.next_token()
        let event = Event(EventKind.SequenceEnd, token.start_mark, token.end_mark)
        self.state = self.states.pop()
        self.marks.pop()
        return event
    }

    fn parse_indentless_sequence_entry(self) -> Event {
        if self.scanner.check_token(TokenKind.BlockEntry) {
            let token = self.next_token()
            if not self.scanner.check_token(TokenKind.BlockEntry, TokenKind.Key, TokenKind.Value, TokenKind.BlockEnd) {
                self.states.push(self.parse_indentless_sequence_entry)
                return self.parse_block_node()
            }
            self.state = self.parse_indentless_sequence_entry
            return self.process_empty_scalar(token.end_mark)
        }
        let token = self.look()
        let event = Event(EventKind.SequenceEnd, token.start_mark, token.start_mark)
        self.state = self.states.pop()
        return event
    }

    fn parse_block_mapping_first_key(self) -> Event {
        let token = self.next_token()
        self.marks.push(token.start_mark)
        return self.parse_block_mapping_key()
    }

    fn parse_block_mapping_key(self) -> Event {
        if self.scanner.check_token(TokenKind.Key) {
            let token = self.next_token()
            if not self.scanner.check_token(TokenKind.Key, TokenKind.Value, TokenKind.BlockEnd) {
                self.states.push(self.parse_block_mapping_value)
                return self.parse_block_node_or_indentless_sequence()
            }
            self.state = self.parse_block_mapping_value
            return self.process_empty_scalar(token.end_mark)
        }
        if not self.scanner.check_token(TokenKind.BlockEnd) {
            let token = self.look()
            throw ParserError(
                "while parsing a block mapping",
                self.marks[self.marks.len() - 1],
                f"expected <block end>, but found {python_quote(token.id())}",
                token.start_mark
            )
        }
        let token = self.next_token()
        let event = Event(EventKind.MappingEnd, token.start_mark, token.end_mark)
        self.state = self.states.pop()
        self.marks.pop()
        return event
    }

    fn parse_block_mapping_value(self) -> Event {
        if self.scanner.check_token(TokenKind.Value) {
            let token = self.next_token()
            if not self.scanner.check_token(TokenKind.Key, TokenKind.Value, TokenKind.BlockEnd) {
                self.states.push(self.parse_block_mapping_key)
                return self.parse_block_node_or_indentless_sequence()
            }
            self.state = self.parse_block_mapping_key
            return self.process_empty_scalar(token.end_mark)
        }
        self.state = self.parse_block_mapping_key
        let token = self.look()
        return self.process_empty_scalar(token.start_mark)
    }

    fn parse_flow_sequence_first_entry(self) -> Event {
        let token = self.next_token()
        self.marks.push(token.start_mark)
        return self.parse_flow_sequence_entry(true)
    }

    fn parse_flow_sequence_entry(self, first: bool = false) -> Event {
        if not self.scanner.check_token(TokenKind.FlowSequenceEnd) {
            if not first {
                if self.scanner.check_token(TokenKind.FlowEntry) {
                    self.next_token()
                } else {
                    let token = self.look()
                    throw ParserError(
                        "while parsing a flow sequence",
                        self.marks[self.marks.len() - 1],
                        f"expected ',' or ']', but got {python_quote(token.id())}",
                        token.start_mark
                    )
                }
            }
            if self.scanner.check_token(TokenKind.Key) {
                let token = self.look()
                let event = Event(EventKind.MappingStart, token.start_mark, token.end_mark)
                event.implicit = true
                event.flow_style = true
                self.state = self.parse_flow_sequence_entry_mapping_key
                return event
            } elif not self.scanner.check_token(TokenKind.FlowSequenceEnd) {
                self.states.push(self.parse_flow_sequence_entry_again)
                return self.parse_flow_node()
            }
        }
        let token = self.next_token()
        let event = Event(EventKind.SequenceEnd, token.start_mark, token.end_mark)
        self.state = self.states.pop()
        self.marks.pop()
        return event
    }

    fn parse_flow_sequence_entry_again(self) -> Event {
        return self.parse_flow_sequence_entry(false)
    }

    fn parse_flow_sequence_entry_mapping_key(self) -> Event {
        let token = self.next_token()
        if not self.scanner.check_token(TokenKind.Value, TokenKind.FlowEntry, TokenKind.FlowSequenceEnd) {
            self.states.push(self.parse_flow_sequence_entry_mapping_value)
            return self.parse_flow_node()
        }
        self.state = self.parse_flow_sequence_entry_mapping_value
        return self.process_empty_scalar(token.end_mark)
    }

    fn parse_flow_sequence_entry_mapping_value(self) -> Event {
        if self.scanner.check_token(TokenKind.Value) {
            let token = self.next_token()
            if not self.scanner.check_token(TokenKind.FlowEntry, TokenKind.FlowSequenceEnd) {
                self.states.push(self.parse_flow_sequence_entry_mapping_end)
                return self.parse_flow_node()
            }
            self.state = self.parse_flow_sequence_entry_mapping_end
            return self.process_empty_scalar(token.end_mark)
        }
        self.state = self.parse_flow_sequence_entry_mapping_end
        let token = self.look()
        return self.process_empty_scalar(token.start_mark)
    }

    fn parse_flow_sequence_entry_mapping_end(self) -> Event {
        self.state = self.parse_flow_sequence_entry_again
        let token = self.look()
        return Event(EventKind.MappingEnd, token.start_mark, token.start_mark)
    }

    fn parse_flow_mapping_first_key(self) -> Event {
        let token = self.next_token()
        self.marks.push(token.start_mark)
        return self.parse_flow_mapping_key(true)
    }

    fn parse_flow_mapping_key(self, first: bool = false) -> Event {
        if not self.scanner.check_token(TokenKind.FlowMappingEnd) {
            if not first {
                if self.scanner.check_token(TokenKind.FlowEntry) {
                    self.next_token()
                } else {
                    let token = self.look()
                    throw ParserError(
                        "while parsing a flow mapping",
                        self.marks[self.marks.len() - 1],
                        "expected ',' or '}', but got " + python_quote(token.id()),
                        token.start_mark
                    )
                }
            }
            if self.scanner.check_token(TokenKind.Key) {
                let token = self.next_token()
                if not self.scanner.check_token(TokenKind.Value, TokenKind.FlowEntry, TokenKind.FlowMappingEnd) {
                    self.states.push(self.parse_flow_mapping_value)
                    return self.parse_flow_node()
                }
                self.state = self.parse_flow_mapping_value
                return self.process_empty_scalar(token.end_mark)
            } elif not self.scanner.check_token(TokenKind.FlowMappingEnd) {
                self.states.push(self.parse_flow_mapping_empty_value)
                return self.parse_flow_node()
            }
        }
        let token = self.next_token()
        let event = Event(EventKind.MappingEnd, token.start_mark, token.end_mark)
        self.state = self.states.pop()
        self.marks.pop()
        return event
    }

    fn parse_flow_mapping_key_again(self) -> Event { return self.parse_flow_mapping_key(false) }

    fn parse_flow_mapping_value(self) -> Event {
        if self.scanner.check_token(TokenKind.Value) {
            let token = self.next_token()
            if not self.scanner.check_token(TokenKind.FlowEntry, TokenKind.FlowMappingEnd) {
                self.states.push(self.parse_flow_mapping_key_again)
                return self.parse_flow_node()
            }
            self.state = self.parse_flow_mapping_key_again
            return self.process_empty_scalar(token.end_mark)
        }
        self.state = self.parse_flow_mapping_key_again
        let token = self.look()
        return self.process_empty_scalar(token.start_mark)
    }

    fn parse_flow_mapping_empty_value(self) -> Event {
        self.state = self.parse_flow_mapping_key_again
        return self.process_empty_scalar(self.look().start_mark)
    }

    fn process_empty_scalar(self, mark: Mark) -> Event {
        let event = Event(EventKind.Scalar, mark, mark)
        event.implicit = true
        event.implicit_quoted = false
        event.value = ""
        return event
    }
}

#: Python's `%r` of a short ASCII name: single quotes around it.
fn python_quote(text: str) -> str { return "'" + text + "'" }
