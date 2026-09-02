#: YAML reading and writing -- this language's PyYAML.
#:
#: `safe_load`, `safe_load_all`, `safe_dump` and `safe_dump_all` take and
#: return the same things PyYAML's do, under the same names and keyword
#: arguments, so a program written against `import yaml` translates line for
#: line with `import jaiyaml as yaml`. Every stage is exposed as well:
#: `scan`, `parse`, `compose`, `compose_all`, `serialize`, `serialize_all`
#: and `emit`, plus the `Scanner`, `Parser`, `Composer`, `Constructor`,
#: `Representer`, `Serializer` and `Emitter` classes behind them.
#:
#: ```
#: import jaiyaml as yaml
#:
#: let config = yaml.safe_load("name: demo\nports: [80, 443]\n")
#: config["ports"][1]                            # 443
#: yaml.safe_dump({"a": 1, "b": [true, null]})   # "a: 1\nb:\n- true\n- null\n"
#: ```
#:
#: Loading follows the YAML 1.2 core schema -- `null`, `true`/`false`,
#: decimal, `0o` and `0x` ints, floats with `.inf` and `.nan` -- and not the
#: YAML 1.1 types PyYAML adds (`yes`/`no`, octal `010`, sexagesimal,
#: timestamps). Every error is a `YAMLError`; the ones with a position are
#: `MarkedYAMLError`s and carry `line` and `column`, one-based, in the
#: message and as fields. The README lists each deliberate difference.
#:
#: Nothing here is seeded: the package compiles from source on first import.
from .composer import Composer
from .constructor import Constructor, base64_decode, base64_encode
from .emitter import Emitter
from .error import ComposerError, ConstructorError, EmitterError, Mark, MarkedYAMLError
from .error import ParserError, ReaderError, RepresenterError, ScannerError, SerializerError
from .error import YAMLError
from .events import Event, EventKind
from .nodes import Node, NodeKind
from .parser import Parser
from .reader import Reader, STRING_NAME
from .representer import Representer
from .resolver import BINARY_TAG, BOOL_TAG, DEFAULT_MAPPING_TAG, DEFAULT_SCALAR_TAG
from .resolver import DEFAULT_SEQUENCE_TAG, FLOAT_TAG, INT_TAG, MAP_TAG, MERGE_TAG, NULL_TAG
from .resolver import OMAP_TAG, PAIRS_TAG, SEQ_TAG, SET_TAG, STR_TAG, resolve, resolve_plain
from .scanner import Scanner
from .serializer import Serializer
from .tokens import Token, TokenKind

export { safe_load, safe_load_all, safe_dump, safe_dump_all }
export { scan, parse, compose, compose_all, serialize, serialize_all, emit }
export { Mark, YAMLError, MarkedYAMLError, ReaderError, ScannerError, ParserError }
export { ComposerError, ConstructorError, RepresenterError, SerializerError, EmitterError }
export { Token, TokenKind, Event, EventKind, Node, NodeKind }
export { Reader, Scanner, Parser, Composer, Constructor, Representer, Serializer, Emitter }
export { resolve, resolve_plain, base64_decode, base64_encode, STRING_NAME }
export { DEFAULT_SCALAR_TAG, DEFAULT_SEQUENCE_TAG, DEFAULT_MAPPING_TAG }
export { NULL_TAG, BOOL_TAG, INT_TAG, FLOAT_TAG, STR_TAG, SEQ_TAG, MAP_TAG, SET_TAG }
export { MERGE_TAG, BINARY_TAG, OMAP_TAG, PAIRS_TAG }

#: The package version, which `jaithon.package.json` also carries.
pub let VERSION = "0.1.0"

#: Parse the first document of `stream` and return its value.
#:
#: Mappings become `dict[any, any]`, sequences `list[any]`, and scalars
#: null, bool, int, float or str by the core schema. An empty stream is
#: null. Raises a `YAMLError` subclass, with its line and column, on
#: anything malformed, and `ComposerError` when a second document follows.
#:
#: ```
#: safe_load("a: 1\nb: [x, y]\n")     # {"a": 1, "b": ["x", "y"]}
#: safe_load("- 0x10\n- .inf\n- ~")   # [16, inf, null]
#: safe_load("")                       # null
#: ```
pub fn safe_load(stream: str) -> any { return Constructor(stream).get_single_data() }

#: Parse every document of `stream` and return their values in order.
#:
#: PyYAML returns a generator; this returns a list, since the whole text is
#: already in memory.
#:
#: ```
#: safe_load_all("a: 1\n---\nb: 2\n")   # [{"a": 1}, {"b": 2}]
#: safe_load_all("")                     # []
#: ```
pub fn safe_load_all(stream: str) -> list[any] {
    let constructor = Constructor(stream)
    let documents: list[any] = []
    while constructor.check_data() { documents.push(constructor.get_data()) }
    return documents
}

#: Render `data` as one YAML document.
#:
#: The options are PyYAML's, with its defaults: `default_flow_style` false
#: writes block style everywhere, true flow style everywhere, and null flow
#: style for collections of scalars only; `sort_keys` orders mapping keys;
#: `indent` (2 to 9) and `width` shape the layout; `allow_unicode` writes
#: non-ASCII as itself instead of `\u` escapes; `default_style` forces a
#: scalar style (`'`, `"`, `|` or `>`); `canonical`, `explicit_start`,
#: `explicit_end`, `line_break`, `version` and `tags` decorate the document.
#:
#: Raises `RepresenterError` for a value with no YAML form, such as a class
#: instance.
#:
#: ```
#: safe_dump({"b": 1, "a": [1, 2]})                          # "a:\n- 1\n- 2\nb: 1\n"
#: safe_dump([1, "two"], default_flow_style: true)           # "[1, two]\n"
#: safe_dump("é", allow_unicode: true)                       # "é\n...\n"
#: ```
pub fn safe_dump(
    data: any,
    default_style: str? = null,
    default_flow_style: bool? = false,
    canonical: bool = false,
    indent: int? = null,
    width: int? = null,
    allow_unicode: bool = false,
    line_break: str? = null,
    explicit_start: bool = false,
    explicit_end: bool = false,
    version: tuple[int, int]? = null,
    tags: dict[str, str]? = null,
    sort_keys: bool = true
) -> str {
    return safe_dump_all(
        [data],
        default_style: default_style,
        default_flow_style: default_flow_style,
        canonical: canonical,
        indent: indent,
        width: width,
        allow_unicode: allow_unicode,
        line_break: line_break,
        explicit_start: explicit_start,
        explicit_end: explicit_end,
        version: version,
        tags: tags,
        sort_keys: sort_keys
    )
}

#: Render each of `documents` as a YAML document in one stream.
#:
#: Documents after the first are introduced by `---`, as PyYAML writes them.
#: The options are `safe_dump`'s.
#:
#: ```
#: safe_dump_all([1, 2])   # "1\n---\n2\n...\n"
#: ```
pub fn safe_dump_all(
    documents: list[any],
    default_style: str? = null,
    default_flow_style: bool? = false,
    canonical: bool = false,
    indent: int? = null,
    width: int? = null,
    allow_unicode: bool = false,
    line_break: str? = null,
    explicit_start: bool = false,
    explicit_end: bool = false,
    version: tuple[int, int]? = null,
    tags: dict[str, str]? = null,
    sort_keys: bool = true
) -> str {
    let emitter = Emitter(canonical, indent, width, allow_unicode, line_break)
    let serializer = Serializer(emitter, explicit_start, explicit_end, version, tags)
    let representer = Representer(default_style, default_flow_style, sort_keys)
    serializer.open()
    for document in documents { serializer.serialize(representer.represent(document)) }
    serializer.close()
    return emitter.result()
}

#: The tokens of `stream`, STREAM-START to STREAM-END.
#:
#: ```
#: scan("a: 1").len()   # 8
#: ```
pub fn scan(stream: str) -> list[Token] {
    let scanner = Scanner(stream)
    let tokens: list[Token] = []
    while scanner.check_token() { tokens.push(scanner.get_token()) }
    return tokens
}

#: The events of `stream`, STREAM-START to STREAM-END.
#:
#: ```
#: parse("- 1").len()   # 7
#: ```
pub fn parse(stream: str) -> list[Event] {
    let parser = Parser(stream)
    let events: list[Event] = []
    while parser.check_event() { events.push(parser.get_event()) }
    return events
}

#: The root node of the first document of `stream`, or null when empty.
#:
#: ```
#: compose("a: 1").kind   # NodeKind.Mapping
#: ```
pub fn compose(stream: str) -> Node? { return Composer(stream).get_single_node() }

#: The root nodes of every document of `stream`.
pub fn compose_all(stream: str) -> list[Node] {
    let composer = Composer(stream)
    let nodes: list[Node] = []
    while composer.check_node() {
        let node = composer.get_node()
        if node is not null { nodes.push(node) }
    }
    return nodes
}

#: Write `node` as one document; the options are `safe_dump`'s.
pub fn serialize(
    node: Node,
    canonical: bool = false,
    indent: int? = null,
    width: int? = null,
    allow_unicode: bool = false,
    line_break: str? = null,
    explicit_start: bool = false,
    explicit_end: bool = false,
    version: tuple[int, int]? = null,
    tags: dict[str, str]? = null
) -> str {
    return serialize_all(
        [node],
        canonical: canonical,
        indent: indent,
        width: width,
        allow_unicode: allow_unicode,
        line_break: line_break,
        explicit_start: explicit_start,
        explicit_end: explicit_end,
        version: version,
        tags: tags
    )
}

#: Write each of `nodes` as a document in one stream.
pub fn serialize_all(
    nodes: list[Node],
    canonical: bool = false,
    indent: int? = null,
    width: int? = null,
    allow_unicode: bool = false,
    line_break: str? = null,
    explicit_start: bool = false,
    explicit_end: bool = false,
    version: tuple[int, int]? = null,
    tags: dict[str, str]? = null
) -> str {
    let emitter = Emitter(canonical, indent, width, allow_unicode, line_break)
    let serializer = Serializer(emitter, explicit_start, explicit_end, version, tags)
    serializer.open()
    for node in nodes { serializer.serialize(node) }
    serializer.close()
    return emitter.result()
}

#: Write a stream of `events`, as `parse` produces them, back as text.
#:
#: ```
#: emit(parse("a: 1\n"))   # "a: 1\n"
#: ```
pub fn emit(
    events: list[Event],
    canonical: bool = false,
    indent: int? = null,
    width: int? = null,
    allow_unicode: bool = false,
    line_break: str? = null
) -> str {
    let emitter = Emitter(canonical, indent, width, allow_unicode, line_break)
    for event in events { emitter.emit(event) }
    return emitter.result()
}
