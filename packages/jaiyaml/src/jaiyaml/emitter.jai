#: Events to text: PyYAML's emitter, state for state and writer for writer.
#:
#: The emitter decides how every scalar is written by analysing its text --
#: which indicators it starts with, whether it has leading or trailing
#: space, breaks, characters outside the printable range -- and picking the
#: first style that can hold it: plain, single-quoted, double-quoted, or the
#: block style the event asked for. Plain and quoted scalars are folded at
#: `width`; block collections indent by `indent`; `allow_unicode` decides
#: whether non-ASCII is written as itself or as `\u` escapes.
#:
#: The output goes to a `StringBuilder`; `result()` returns it. Everything
#: PyYAML writes for a string source is written here, byte for byte, which
#: is what the differential test in `tests/py` holds the emitter to.
from std.str import StringBuilder, repeat

from .error import EmitterError, hex_upper, python_repr
from .events import Event, EventKind

let BREAKS = "\n\u{85}\u{2028}\u{2029}"
let BLANK_OR_BREAK = "\u{0} \t\r\n\u{85}\u{2028}\u{2029}"
let SPACE_OR_BREAK = " \n\u{85}\u{2028}\u{2029}"
let LEADING_INDICATORS = "#,[]{}&*!|>'\"%@`"
let URI_CHARS = "-;/?:@&=+$,_.~*'()[]"

let ESCAPE_REPLACEMENTS: dict[str, str] = {
    "\u{0}": "0",
    "\u{7}": "a",
    "\u{8}": "b",
    "\u{9}": "t",
    "\u{A}": "n",
    "\u{B}": "v",
    "\u{C}": "f",
    "\u{D}": "r",
    "\u{1B}": "e",
    "\"": "\"",
    "\\": "\\",
    "\u{85}": "N",
    "\u{A0}": "_",
    "\u{2028}": "L",
    "\u{2029}": "P",
}

fn is_word_char(ch: str) -> bool {
    return ch >= "0" and ch <= "9" or
        ch >= "A" and ch <= "Z" or
        ch >= "a" and ch <= "z" or
        ch == "-" or
        ch == "_"
}

fn is_break(ch: str?) -> bool {
    if ch is null { return false }
    return (ch ?? "") in BREAKS
}

#: What the emitter learned about one scalar's text.
class ScalarAnalysis {
    pub let scalar: str
    pub let empty: bool
    pub let multiline: bool
    pub let allow_flow_plain: bool
    pub let allow_block_plain: bool
    pub let allow_single_quoted: bool
    pub let allow_double_quoted: bool
    pub let allow_block: bool

    pub fn init(
        self,
        scalar: str,
        empty: bool,
        multiline: bool,
        allow_flow_plain: bool,
        allow_block_plain: bool,
        allow_single_quoted: bool,
        allow_double_quoted: bool,
        allow_block: bool
    ) {
        self.scalar = scalar
        self.empty = empty
        self.multiline = multiline
        self.allow_flow_plain = allow_flow_plain
        self.allow_block_plain = allow_block_plain
        self.allow_single_quoted = allow_single_quoted
        self.allow_double_quoted = allow_double_quoted
        self.allow_block = allow_block
    }
}

#: Writes a stream of events as YAML text.
pub class Emitter {
    let stream: StringBuilder
    var states: list[fn() -> void]
    var state: (fn() -> void)?
    var events: list[Event]
    var event: Event?
    var indents: list[int?]
    var indent: int?
    var flow_level: int
    var root_context: bool
    var sequence_context: bool
    var mapping_context: bool
    var simple_key_context: bool
    var line: int
    var column: int
    var whitespace: bool
    var indention: bool
    var open_ended: bool
    let canonical: bool
    let allow_unicode: bool
    let best_indent: int
    let best_width: int
    let best_line_break: str
    var tag_prefixes: dict[str, str]
    var prepared_anchor: str?
    var prepared_tag: str?
    var analysis: ScalarAnalysis?
    var style: str?

    #: An emitter with PyYAML's formatting options.
    #:
    #: `indent` outside 2 to 9 falls back to 2, `width` no larger than twice
    #: the indent falls back to 80, and `line_break` must be `\r`, `\n` or
    #: `\r\n`.
    pub fn init(
        self,
        canonical: bool = false,
        indent: int? = null,
        width: int? = null,
        allow_unicode: bool = false,
        line_break: str? = null
    ) {
        self.stream = StringBuilder()
        self.states = []
        self.state = self.expect_stream_start
        self.events = []
        self.event = null
        self.indents = []
        self.indent = null
        self.flow_level = 0
        self.root_context = false
        self.sequence_context = false
        self.mapping_context = false
        self.simple_key_context = false
        self.line = 0
        self.column = 0
        self.whitespace = true
        self.indention = true
        self.open_ended = false
        self.canonical = canonical
        self.allow_unicode = allow_unicode
        var chosen_indent = 2
        if indent is not null {
            let asked = indent ?? 2
            if asked > 1 and asked < 10 { chosen_indent = asked }
        }
        self.best_indent = chosen_indent
        var chosen_width = 80
        if width is not null {
            let asked = width ?? 80
            if asked > chosen_indent * 2 { chosen_width = asked }
        }
        self.best_width = chosen_width
        var chosen_break = "\n"
        if line_break is not null {
            let asked = line_break ?? "\n"
            if asked == "\r" or asked == "\n" or asked == "\r\n" { chosen_break = asked }
        }
        self.best_line_break = chosen_break
        self.tag_prefixes = {}
        self.prepared_anchor = null
        self.prepared_tag = null
        self.analysis = null
        self.style = null
    }

    #: Everything written so far.
    pub fn result(self) -> str { return self.stream.build() }

    #: Queue `event`, writing as many queued events as can be decided.
    pub fn emit(self, event: Event) -> void {
        self.events.push(event)
        while not self.need_more_events() {
            self.event = self.events.pop(0)
            let step = self.state
            if step is null { throw EmitterError("the emitter has finished") }
            let run = step ?? self.expect_nothing
            run()
            self.event = null
        }
    }

    fn current(self) -> Event {
        let event = self.event
        if event is null { throw EmitterError("no current event") }
        return event ?? Event(EventKind.StreamEnd)
    }

    fn need_more_events(self) -> bool {
        if self.events.len() == 0 { return true }
        let event = self.events[0]
        if event.kind == EventKind.DocumentStart { return self.need_events(1) }
        if event.kind == EventKind.SequenceStart { return self.need_events(2) }
        if event.kind == EventKind.MappingStart { return self.need_events(3) }
        return false
    }

    fn need_events(self, count: int) -> bool {
        var level = 0
        var index = 1
        while index < self.events.len() {
            let event = self.events[index]
            if event.kind == EventKind.DocumentStart or event.is_collection_start() {
                level += 1
            } elif event.kind == EventKind.DocumentEnd or event.is_collection_end() {
                level -= 1
            } elif event.kind == EventKind.StreamEnd {
                level = -1
            }
            if level < 0 { return false }
            index += 1
        }
        return self.events.len() < count + 1
    }

    fn increase_indent(self, flow: bool = false, indentless: bool = false) -> void {
        self.indents.push(self.indent)
        if self.indent is null {
            self.indent = flow ? self.best_indent : 0
        } elif not indentless {
            self.indent = (self.indent ?? 0) + self.best_indent
        }
    }

    fn expect_stream_start(self) -> void {
        let event = self.current()
        if event.kind == EventKind.StreamStart {
            self.state = self.expect_first_document_start
        } else {
            throw EmitterError(f"expected StreamStartEvent, but got {event}")
        }
    }

    fn expect_nothing(self) -> void {
        throw EmitterError(f"expected nothing, but got {self.current()}")
    }

    fn expect_first_document_start(self) -> void { self.expect_document_start(true) }

    fn expect_document_start(self, first: bool = false) -> void {
        let event = self.current()
        if event.kind == EventKind.DocumentStart {
            if (event.version is not null or event.tags is not null) and self.open_ended {
                self.write_indicator("...", true)
                self.write_indent()
            }
            if event.version is not null {
                let version_text = self.prepare_version(event.version ?? (1, 1))
                self.write_version_directive(version_text)
            }
            self.tag_prefixes = {"!": "!", "tag:yaml.org,2002:": "!!"}
            if event.tags is not null {
                let tags = event.tags ?? {}
                for handle in sorted(tags.keys()) {
                    let prefix = tags[handle]
                    self.tag_prefixes[prefix] = handle
                    let handle_text = self.prepare_tag_handle(handle)
                    let prefix_text = self.prepare_tag_prefix(prefix)
                    self.write_tag_directive(handle_text, prefix_text)
                }
            }
            let implicit = first and
                not event.explicit and
                not self.canonical and
                event.version is null and
                event.tags is null and
                not self.check_empty_document()
            if not implicit {
                self.write_indent()
                self.write_indicator("---", true)
                if self.canonical { self.write_indent() }
            }
            self.state = self.expect_document_root
        } elif event.kind == EventKind.StreamEnd {
            if self.open_ended {
                self.write_indicator("...", true)
                self.write_indent()
            }
            self.state = self.expect_nothing
        } else {
            throw EmitterError(f"expected DocumentStartEvent, but got {event}")
        }
    }

    fn expect_document_start_again(self) -> void { self.expect_document_start(false) }

    fn expect_document_end(self) -> void {
        let event = self.current()
        if event.kind == EventKind.DocumentEnd {
            self.write_indent()
            if event.explicit {
                self.write_indicator("...", true)
                self.write_indent()
            }
            self.state = self.expect_document_start_again
        } else {
            throw EmitterError(f"expected DocumentEndEvent, but got {event}")
        }
    }

    fn expect_document_root(self) -> void {
        self.states.push(self.expect_document_end)
        self.expect_node(true, false, false, false)
    }

    fn expect_node(self, root: bool, sequence: bool, mapping: bool, simple_key: bool) -> void {
        self.root_context = root
        self.sequence_context = sequence
        self.mapping_context = mapping
        self.simple_key_context = simple_key
        let event = self.current()
        if event.kind == EventKind.Alias {
            self.expect_alias()
        } elif event.kind == EventKind.Scalar or event.is_collection_start() {
            self.process_anchor("&")
            self.process_tag()
            if event.kind == EventKind.Scalar {
                self.expect_scalar()
            } elif event.kind == EventKind.SequenceStart {
                if self.flow_level > 0 or
                    self.canonical or
                    (event.flow_style ?? false) or
                    self.check_empty_sequence() {
                    self.expect_flow_sequence()
                } else {
                    self.expect_block_sequence()
                }
            } else {
                if self.flow_level > 0 or
                    self.canonical or
                    (event.flow_style ?? false) or
                    self.check_empty_mapping() {
                    self.expect_flow_mapping()
                } else {
                    self.expect_block_mapping()
                }
            }
        } else {
            throw EmitterError(f"expected NodeEvent, but got {event}")
        }
    }

    fn expect_alias(self) -> void {
        if self.current().anchor is null { throw EmitterError("anchor is not specified for alias") }
        self.process_anchor("*")
        self.state = self.states.pop()
    }

    fn expect_scalar(self) -> void {
        self.increase_indent(true)
        self.process_scalar()
        self.indent = self.indents.pop()
        self.state = self.states.pop()
    }

    fn expect_flow_sequence(self) -> void {
        self.write_indicator("[", true, true)
        self.flow_level += 1
        self.increase_indent(true)
        self.state = self.expect_first_flow_sequence_item
    }

    fn expect_first_flow_sequence_item(self) -> void {
        if self.current().kind == EventKind.SequenceEnd {
            self.indent = self.indents.pop()
            self.flow_level -= 1
            self.write_indicator("]", false)
            self.state = self.states.pop()
        } else {
            if self.canonical or self.column > self.best_width { self.write_indent() }
            self.states.push(self.expect_flow_sequence_item)
            self.expect_node(false, true, false, false)
        }
    }

    fn expect_flow_sequence_item(self) -> void {
        if self.current().kind == EventKind.SequenceEnd {
            self.indent = self.indents.pop()
            self.flow_level -= 1
            if self.canonical {
                self.write_indicator(",", false)
                self.write_indent()
            }
            self.write_indicator("]", false)
            self.state = self.states.pop()
        } else {
            self.write_indicator(",", false)
            if self.canonical or self.column > self.best_width { self.write_indent() }
            self.states.push(self.expect_flow_sequence_item)
            self.expect_node(false, true, false, false)
        }
    }

    fn expect_flow_mapping(self) -> void {
        self.write_indicator("{", true, true)
        self.flow_level += 1
        self.increase_indent(true)
        self.state = self.expect_first_flow_mapping_key
    }

    fn expect_first_flow_mapping_key(self) -> void {
        if self.current().kind == EventKind.MappingEnd {
            self.indent = self.indents.pop()
            self.flow_level -= 1
            self.write_indicator("}", false)
            self.state = self.states.pop()
        } else {
            if self.canonical or self.column > self.best_width { self.write_indent() }
            if not self.canonical and self.check_simple_key() {
                self.states.push(self.expect_flow_mapping_simple_value)
                self.expect_node(false, false, true, true)
            } else {
                self.write_indicator("?", true)
                self.states.push(self.expect_flow_mapping_value)
                self.expect_node(false, false, true, false)
            }
        }
    }

    fn expect_flow_mapping_key(self) -> void {
        if self.current().kind == EventKind.MappingEnd {
            self.indent = self.indents.pop()
            self.flow_level -= 1
            if self.canonical {
                self.write_indicator(",", false)
                self.write_indent()
            }
            self.write_indicator("}", false)
            self.state = self.states.pop()
        } else {
            self.write_indicator(",", false)
            if self.canonical or self.column > self.best_width { self.write_indent() }
            if not self.canonical and self.check_simple_key() {
                self.states.push(self.expect_flow_mapping_simple_value)
                self.expect_node(false, false, true, true)
            } else {
                self.write_indicator("?", true)
                self.states.push(self.expect_flow_mapping_value)
                self.expect_node(false, false, true, false)
            }
        }
    }

    fn expect_flow_mapping_simple_value(self) -> void {
        self.write_indicator(":", false)
        self.states.push(self.expect_flow_mapping_key)
        self.expect_node(false, false, true, false)
    }

    fn expect_flow_mapping_value(self) -> void {
        if self.canonical or self.column > self.best_width { self.write_indent() }
        self.write_indicator(":", true)
        self.states.push(self.expect_flow_mapping_key)
        self.expect_node(false, false, true, false)
    }

    fn expect_block_sequence(self) -> void {
        let indentless = self.mapping_context and not self.indention
        self.increase_indent(false, indentless)
        self.state = self.expect_first_block_sequence_item
    }

    fn expect_first_block_sequence_item(self) -> void { self.expect_block_sequence_item(true) }

    fn expect_block_sequence_item(self, first: bool = false) -> void {
        if not first and self.current().kind == EventKind.SequenceEnd {
            self.indent = self.indents.pop()
            self.state = self.states.pop()
        } else {
            self.write_indent()
            self.write_indicator("-", true, false, true)
            self.states.push(self.expect_block_sequence_item_again)
            self.expect_node(false, true, false, false)
        }
    }

    fn expect_block_sequence_item_again(self) -> void { self.expect_block_sequence_item(false) }

    fn expect_block_mapping(self) -> void {
        self.increase_indent(false)
        self.state = self.expect_first_block_mapping_key
    }

    fn expect_first_block_mapping_key(self) -> void { self.expect_block_mapping_key(true) }

    fn expect_block_mapping_key(self, first: bool = false) -> void {
        if not first and self.current().kind == EventKind.MappingEnd {
            self.indent = self.indents.pop()
            self.state = self.states.pop()
        } else {
            self.write_indent()
            if self.check_simple_key() {
                self.states.push(self.expect_block_mapping_simple_value)
                self.expect_node(false, false, true, true)
            } else {
                self.write_indicator("?", true, false, true)
                self.states.push(self.expect_block_mapping_value)
                self.expect_node(false, false, true, false)
            }
        }
    }

    fn expect_block_mapping_key_again(self) -> void { self.expect_block_mapping_key(false) }

    fn expect_block_mapping_simple_value(self) -> void {
        self.write_indicator(":", false)
        self.states.push(self.expect_block_mapping_key_again)
        self.expect_node(false, false, true, false)
    }

    fn expect_block_mapping_value(self) -> void {
        self.write_indent()
        self.write_indicator(":", true, false, true)
        self.states.push(self.expect_block_mapping_key_again)
        self.expect_node(false, false, true, false)
    }

    fn check_empty_sequence(self) -> bool {
        return self.current().kind == EventKind.SequenceStart and
            self.events.len() > 0 and
            self.events[0].kind == EventKind.SequenceEnd
    }

    fn check_empty_mapping(self) -> bool {
        return self.current().kind == EventKind.MappingStart and
            self.events.len() > 0 and
            self.events[0].kind == EventKind.MappingEnd
    }

    fn check_empty_document(self) -> bool {
        if self.current().kind != EventKind.DocumentStart or self.events.len() == 0 { return false }
        let event = self.events[0]
        return event.kind == EventKind.Scalar and
            event.anchor is null and
            event.tag is null and
            event.implicit and
            event.value == ""
    }

    fn check_simple_key(self) -> bool {
        let event = self.current()
        var length = 0
        if event.is_node() and event.anchor is not null {
            if self.prepared_anchor is null {
                self.prepared_anchor = self.prepare_anchor(event.anchor ?? "")
            }
            length += (self.prepared_anchor ?? "").len()
        }
        if (event.kind == EventKind.Scalar or event.is_collection_start()) and event.tag is not null {
            if self.prepared_tag is null { self.prepared_tag = self.prepare_tag(event.tag ?? "") }
            length += (self.prepared_tag ?? "").len()
        }
        var analysis: ScalarAnalysis? = null
        if event.kind == EventKind.Scalar {
            if self.analysis is null { self.analysis = self.analyze_scalar(event.value) }
            analysis = self.analysis
            length += self.analyzed().scalar.len()
        }
        if length >= 128 { return false }
        if event.kind == EventKind.Alias { return true }
        if event.kind == EventKind.Scalar {
            let found = self.analyzed()
            if not found.empty and not found.multiline { return true }
        }
        return self.check_empty_sequence() or self.check_empty_mapping()
    }

    fn analyzed(self) -> ScalarAnalysis {
        let analysis = self.analysis
        if analysis is null { throw EmitterError("scalar not analysed") }
        return analysis ?? self.analyze_scalar("")
    }

    fn process_anchor(self, indicator: str) -> void {
        let event = self.current()
        if event.anchor is null {
            self.prepared_anchor = null
            return
        }
        if self.prepared_anchor is null {
            self.prepared_anchor = self.prepare_anchor(event.anchor ?? "")
        }
        let prepared = self.prepared_anchor ?? ""
        if prepared.len() > 0 { self.write_indicator(indicator + prepared, true) }
        self.prepared_anchor = null
    }

    fn process_tag(self) -> void {
        let event = self.current()
        var tag = event.tag
        if event.kind == EventKind.Scalar {
            if self.style is null { self.style = self.choose_scalar_style() }
            let style = self.style ?? ""
            if (not self.canonical or tag is null) and
                (style == "" and event.implicit or style != "" and event.implicit_quoted) {
                self.prepared_tag = null
                return
            }
            if event.implicit and tag is null {
                tag = "!"
                self.prepared_tag = null
            }
        } else {
            if (not self.canonical or tag is null) and event.implicit {
                self.prepared_tag = null
                return
            }
        }
        if tag is null { throw EmitterError("tag is not specified") }
        if self.prepared_tag is null { self.prepared_tag = self.prepare_tag(tag ?? "") }
        let prepared = self.prepared_tag ?? ""
        if prepared.len() > 0 { self.write_indicator(prepared, true) }
        self.prepared_tag = null
    }

    fn choose_scalar_style(self) -> str {
        let event = self.current()
        if self.analysis is null { self.analysis = self.analyze_scalar(event.value) }
        let analysis = self.analyzed()
        let event_style = event.style ?? ""
        if event_style == "\"" or self.canonical { return "\"" }
        if event_style == "" and event.implicit {
            if not (self.simple_key_context and (analysis.empty or analysis.multiline)) and
                (self.flow_level > 0 and analysis.allow_flow_plain or self.flow_level == 0 and analysis.allow_block_plain) {
                return ""
            }
        }
        if event_style == "|" or event_style == ">" {
            if self.flow_level == 0 and not self.simple_key_context and analysis.allow_block {
                return event_style
            }
        }
        if event_style == "" or event_style == "'" {
            if analysis.allow_single_quoted and not (self.simple_key_context and analysis.multiline) {
                return "'"
            }
        }
        return "\""
    }

    fn process_scalar(self) -> void {
        let event = self.current()
        if self.analysis is null { self.analysis = self.analyze_scalar(event.value) }
        if self.style is null { self.style = self.choose_scalar_style() }
        let analysis = self.analyzed()
        let style = self.style ?? ""
        let split = not self.simple_key_context
        if style == "\"" {
            self.write_double_quoted(analysis.scalar, split)
        } elif style == "'" {
            self.write_single_quoted(analysis.scalar, split)
        } elif style == ">" {
            self.write_folded(analysis.scalar)
        } elif style == "|" {
            self.write_literal(analysis.scalar)
        } else {
            self.write_plain(analysis.scalar, split)
        }
        self.analysis = null
        self.style = null
    }

    fn prepare_version(self, version: tuple[int, int]) -> str {
        let (major, minor) = version
        if major != 1 { throw EmitterError(f"unsupported YAML version: {major}.{minor}") }
        return f"{major}.{minor}"
    }

    fn prepare_tag_handle(self, handle: str) -> str {
        if handle.len() == 0 { throw EmitterError("tag handle must not be empty") }
        if handle[0] != "!" or handle[handle.len() - 1] != "!" {
            throw EmitterError(f"tag handle must start and end with '!': {python_repr(handle)}")
        }
        for ch in handle[1:handle.len() - 1] {
            if not is_word_char(ch) {
                throw EmitterError(
                    f"invalid character {python_repr(ch)} in the tag handle: {python_repr(handle)}"
                )
            }
        }
        return handle
    }

    fn prepare_tag_prefix(self, prefix: str) -> str {
        if prefix.len() == 0 { throw EmitterError("tag prefix must not be empty") }
        let chunks = StringBuilder()
        var start = 0
        var end = 0
        if prefix[0] == "!" { end = 1 }
        while end < prefix.len() {
            let ch = prefix[end]
            if is_word_char(ch) or ch in "-;/?!:@&=+$,_.~*'()[]" {
                end += 1
            } else {
                if start < end { chunks.push(prefix[start:end]) }
                start = end + 1
                end = start
                for byte in ch.to_bytes() { chunks.push("%" + hex_upper(byte, 2)) }
            }
        }
        if start < end { chunks.push(prefix[start:end]) }
        return chunks.build()
    }

    fn prepare_tag(self, tag: str) -> str {
        if tag.len() == 0 { throw EmitterError("tag must not be empty") }
        if tag == "!" { return tag }
        var handle: str? = null
        var suffix = tag
        for prefix in sorted(self.tag_prefixes.keys()) {
            if tag.starts_with(prefix) and (prefix == "!" or prefix.len() < tag.len()) {
                handle = self.tag_prefixes[prefix]
                suffix = tag[prefix.len():]
            }
        }
        let chunks = StringBuilder()
        var start = 0
        var end = 0
        while end < suffix.len() {
            let ch = suffix[end]
            if is_word_char(ch) or ch in URI_CHARS or ch == "!" and (handle ?? "") != "!" {
                end += 1
            } else {
                if start < end { chunks.push(suffix[start:end]) }
                start = end + 1
                end = start
                for byte in ch.to_bytes() { chunks.push("%" + hex_upper(byte, 2)) }
            }
        }
        if start < end { chunks.push(suffix[start:end]) }
        let suffix_text = chunks.build()
        if handle is not null and (handle ?? "").len() > 0 { return (handle ?? "") + suffix_text }
        return "!<" + suffix_text + ">"
    }

    fn prepare_anchor(self, anchor: str) -> str {
        if anchor.len() == 0 { throw EmitterError("anchor must not be empty") }
        for ch in anchor {
            if not is_word_char(ch) {
                throw EmitterError(
                    f"invalid character {python_repr(ch)} in the anchor: {python_repr(anchor)}"
                )
            }
        }
        return anchor
    }

    fn analyze_scalar(self, scalar: str) -> ScalarAnalysis {
        if scalar.len() == 0 {
            return ScalarAnalysis(scalar, true, false, false, true, true, true, false)
        }
        var block_indicators = false
        var flow_indicators = false
        var line_breaks = false
        var special_characters = false
        var leading_space = false
        var leading_break = false
        var trailing_space = false
        var trailing_break = false
        var break_space = false
        var space_break = false
        if scalar.starts_with("---") or scalar.starts_with("...") {
            block_indicators = true
            flow_indicators = true
        }
        var preceded_by_whitespace = true
        var followed_by_whitespace = scalar.len() == 1 or scalar[1] in BLANK_OR_BREAK
        var previous_space = false
        var previous_break = false
        var index = 0
        let length = scalar.len()
        while index < length {
            let ch = scalar[index]
            if index == 0 {
                if ch in LEADING_INDICATORS {
                    flow_indicators = true
                    block_indicators = true
                }
                if ch == "?" or ch == ":" {
                    flow_indicators = true
                    if followed_by_whitespace { block_indicators = true }
                }
                if ch == "-" and followed_by_whitespace {
                    flow_indicators = true
                    block_indicators = true
                }
            } else {
                if ch in ",?[]{}" { flow_indicators = true }
                if ch == ":" {
                    flow_indicators = true
                    if followed_by_whitespace { block_indicators = true }
                }
                if ch == "#" and preceded_by_whitespace {
                    flow_indicators = true
                    block_indicators = true
                }
            }
            let code = ord(ch)
            if ch in BREAKS { line_breaks = true }
            if not (ch == "\n" or code >= 0x20 and code <= 0x7E) {
                let unicode = code == 0x85 or
                    code >= 0xA0 and code <= 0xD7FF or
                    code >= 0xE000 and code <= 0xFFFD or
                    code >= 0x10000 and code < 0x10FFFF
                if unicode and code != 0xFEFF {
                    if not self.allow_unicode { special_characters = true }
                } else {
                    special_characters = true
                }
            }
            if ch == " " {
                if index == 0 { leading_space = true }
                if index == length - 1 { trailing_space = true }
                if previous_break { break_space = true }
                previous_space = true
                previous_break = false
            } elif ch in BREAKS {
                if index == 0 { leading_break = true }
                if index == length - 1 { trailing_break = true }
                if previous_space { space_break = true }
                previous_space = false
                previous_break = true
            } else {
                previous_space = false
                previous_break = false
            }
            index += 1
            preceded_by_whitespace = ch in BLANK_OR_BREAK
            followed_by_whitespace = index + 1 >= length or scalar[index + 1] in BLANK_OR_BREAK
        }
        var allow_flow_plain = true
        var allow_block_plain = true
        var allow_single_quoted = true
        var allow_double_quoted = true
        var allow_block = true
        if leading_space or leading_break or trailing_space or trailing_break {
            allow_flow_plain = false
            allow_block_plain = false
        }
        if trailing_space { allow_block = false }
        if break_space {
            allow_flow_plain = false
            allow_block_plain = false
            allow_single_quoted = false
        }
        if space_break or special_characters {
            allow_flow_plain = false
            allow_block_plain = false
            allow_single_quoted = false
            allow_block = false
        }
        if line_breaks {
            allow_flow_plain = false
            allow_block_plain = false
        }
        if flow_indicators { allow_flow_plain = false }
        if block_indicators { allow_block_plain = false }
        return ScalarAnalysis(
            scalar,
            false,
            line_breaks,
            allow_flow_plain,
            allow_block_plain,
            allow_single_quoted,
            allow_double_quoted,
            allow_block
        )
    }

    fn write(self, data: str) -> void { self.stream.push(data) }

    fn write_indicator(
        self,
        indicator: str,
        need_whitespace: bool,
        whitespace: bool = false,
        indention: bool = false
    ) -> void {
        let data = self.whitespace or not need_whitespace ? indicator : " " + indicator
        self.whitespace = whitespace
        self.indention = self.indention and indention
        self.column += data.len()
        self.open_ended = false
        self.write(data)
    }

    fn write_indent(self) -> void {
        let indent = self.indent ?? 0
        if not self.indention or
            self.column > indent or
            self.column == indent and not self.whitespace {
            self.write_line_break()
        }
        if self.column < indent {
            self.whitespace = true
            self.write(repeat(" ", indent - self.column))
            self.column = indent
        }
    }

    fn write_line_break(self, data: str? = null) -> void {
        self.whitespace = true
        self.indention = true
        self.line += 1
        self.column = 0
        self.write(data ?? self.best_line_break)
    }

    fn write_version_directive(self, version_text: str) -> void {
        self.write(f"%YAML {version_text}")
        self.write_line_break()
    }

    fn write_tag_directive(self, handle_text: str, prefix_text: str) -> void {
        self.write(f"%TAG {handle_text} {prefix_text}")
        self.write_line_break()
    }

    fn write_single_quoted(self, text: str, split: bool = true) -> void {
        self.write_indicator("'", true)
        var spaces = false
        var breaks = false
        var start = 0
        var end = 0
        let length = text.len()
        while end <= length {
            var ch: str? = null
            if end < length { ch = text[end] }
            let current = ch ?? ""
            if spaces {
                if ch is null or current != " " {
                    if start + 1 == end and
                        self.column > self.best_width and
                        split and
                        start != 0 and
                        end != length {
                        self.write_indent()
                    } else {
                        let data = text[start:end]
                        self.column += data.len()
                        self.write(data)
                    }
                    start = end
                }
            } elif breaks {
                if ch is null or not is_break(ch) {
                    if text[start] == "\n" { self.write_line_break() }
                    for br in text[start:end] {
                        if br == "\n" {
                            self.write_line_break()
                        } else {
                            self.write_line_break(br)
                        }
                    }
                    self.write_indent()
                    start = end
                }
            } else {
                if ch is null or current in SPACE_OR_BREAK or current == "'" {
                    if start < end {
                        let data = text[start:end]
                        self.column += data.len()
                        self.write(data)
                        start = end
                    }
                }
            }
            if ch is not null and current == "'" {
                self.column += 2
                self.write("''")
                start = end + 1
            }
            if ch is not null {
                spaces = current == " "
                breaks = is_break(ch)
            }
            end += 1
        }
        self.write_indicator("'", false)
    }

    fn write_double_quoted(self, text: str, split: bool = true) -> void {
        self.write_indicator("\"", true)
        var start = 0
        var end = 0
        let length = text.len()
        while end <= length {
            var ch: str? = null
            if end < length { ch = text[end] }
            let current = ch ?? ""
            var escape = ch is null
            if not escape {
                let code = ord(current)
                if current in "\"\\\u{85}\u{2028}\u{2029}\u{FEFF}" {
                    escape = true
                } elif not (code >= 0x20 and code <= 0x7E or self.allow_unicode and (code >= 0xA0 and code <= 0xD7FF or code >= 0xE000 and code <= 0xFFFD)) {
                    escape = true
                }
            }
            if escape {
                if start < end {
                    let data = text[start:end]
                    self.column += data.len()
                    self.write(data)
                    start = end
                }
                if ch is not null {
                    let code = ord(current)
                    var data = ""
                    if current in ESCAPE_REPLACEMENTS {
                        data = "\\" + ESCAPE_REPLACEMENTS[current]
                    } elif code <= 0xFF {
                        data = "\\x" + hex_upper(code, 2)
                    } elif code <= 0xFFFF {
                        data = "\\u" + hex_upper(code, 4)
                    } else {
                        data = "\\U" + hex_upper(code, 8)
                    }
                    self.column += data.len()
                    self.write(data)
                    start = end + 1
                }
            }
            if end > 0 and
                end < length - 1 and
                (current == " " or start >= end) and
                self.column + (end - start) > self.best_width and
                split {
                let data = text[start:end] + "\\"
                if start < end { start = end }
                self.column += data.len()
                self.write(data)
                self.write_indent()
                self.whitespace = false
                self.indention = false
                if text[start] == " " {
                    self.column += 1
                    self.write("\\")
                }
            }
            end += 1
        }
        self.write_indicator("\"", false)
    }

    fn determine_block_hints(self, text: str) -> str {
        var hints = ""
        if text.len() > 0 {
            if text[0] in SPACE_OR_BREAK { hints = hints + str(self.best_indent) }
            let last = text[text.len() - 1]
            if last not in BREAKS {
                hints = hints + "-"
            } elif text.len() == 1 or text[text.len() - 2] in BREAKS {
                hints = hints + "+"
            }
        }
        return hints
    }

    fn write_folded(self, text: str) -> void {
        let hints = self.determine_block_hints(text)
        self.write_indicator(">" + hints, true)
        if hints.len() > 0 and hints[hints.len() - 1] == "+" { self.open_ended = true }
        self.write_line_break()
        var leading_space = true
        var spaces = false
        var breaks = true
        var start = 0
        var end = 0
        let length = text.len()
        while end <= length {
            var ch: str? = null
            if end < length { ch = text[end] }
            let current = ch ?? ""
            if breaks {
                if ch is null or not is_break(ch) {
                    if not leading_space and
                        ch is not null and
                        current != " " and
                        text[start] == "\n" {
                        self.write_line_break()
                    }
                    leading_space = current == " "
                    for br in text[start:end] {
                        if br == "\n" {
                            self.write_line_break()
                        } else {
                            self.write_line_break(br)
                        }
                    }
                    if ch is not null { self.write_indent() }
                    start = end
                }
            } elif spaces {
                if current != " " {
                    if start + 1 == end and self.column > self.best_width {
                        self.write_indent()
                    } else {
                        let data = text[start:end]
                        self.column += data.len()
                        self.write(data)
                    }
                    start = end
                }
            } else {
                if ch is null or current in SPACE_OR_BREAK {
                    let data = text[start:end]
                    self.column += data.len()
                    self.write(data)
                    if ch is null { self.write_line_break() }
                    start = end
                }
            }
            if ch is not null {
                breaks = is_break(ch)
                spaces = current == " "
            }
            end += 1
        }
    }

    fn write_literal(self, text: str) -> void {
        let hints = self.determine_block_hints(text)
        self.write_indicator("|" + hints, true)
        if hints.len() > 0 and hints[hints.len() - 1] == "+" { self.open_ended = true }
        self.write_line_break()
        var breaks = true
        var start = 0
        var end = 0
        let length = text.len()
        while end <= length {
            var ch: str? = null
            if end < length { ch = text[end] }
            if breaks {
                if ch is null or not is_break(ch) {
                    for br in text[start:end] {
                        if br == "\n" {
                            self.write_line_break()
                        } else {
                            self.write_line_break(br)
                        }
                    }
                    if ch is not null { self.write_indent() }
                    start = end
                }
            } else {
                if ch is null or is_break(ch) {
                    self.write(text[start:end])
                    if ch is null { self.write_line_break() }
                    start = end
                }
            }
            if ch is not null { breaks = is_break(ch) }
            end += 1
        }
    }

    fn write_plain(self, text: str, split: bool = true) -> void {
        if self.root_context { self.open_ended = true }
        if text.len() == 0 { return }
        if not self.whitespace {
            self.column += 1
            self.write(" ")
        }
        self.whitespace = false
        self.indention = false
        var spaces = false
        var breaks = false
        var start = 0
        var end = 0
        let length = text.len()
        while end <= length {
            var ch: str? = null
            if end < length { ch = text[end] }
            let current = ch ?? ""
            if spaces {
                if current != " " {
                    if start + 1 == end and self.column > self.best_width and split {
                        self.write_indent()
                        self.whitespace = false
                        self.indention = false
                    } else {
                        let data = text[start:end]
                        self.column += data.len()
                        self.write(data)
                    }
                    start = end
                }
            } elif breaks {
                if not is_break(ch) {
                    if text[start] == "\n" { self.write_line_break() }
                    for br in text[start:end] {
                        if br == "\n" {
                            self.write_line_break()
                        } else {
                            self.write_line_break(br)
                        }
                    }
                    self.write_indent()
                    self.whitespace = false
                    self.indention = false
                    start = end
                }
            } else {
                if ch is null or current in SPACE_OR_BREAK {
                    let data = text[start:end]
                    self.column += data.len()
                    self.write(data)
                    start = end
                }
            }
            if ch is not null {
                spaces = current == " "
                breaks = is_break(ch)
            }
            end += 1
        }
    }
}
