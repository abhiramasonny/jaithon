#: The stage-by-stage surface: `scan`, `parse`, `compose`, `compose_all`,
#: `serialize`, `serialize_all`, `emit`, `resolve` and the base64 helpers.
import jaiyaml as yaml
from jaiyaml import EventKind, Node, NodeKind, SerializerError, Token, TokenKind
from std.test import assert_eq, assert_throws, assert_true

fn kinds_of(tokens: list[Token]) -> list[TokenKind] { return [token.kind for token in tokens] }

fn test_scan_produces_pyyaml_tokens() -> void {
    let tokens = yaml.scan("a: 1")
    assert_eq(
        kinds_of(tokens),
        [
            TokenKind.StreamStart,
            TokenKind.BlockMappingStart,
            TokenKind.Key,
            TokenKind.Scalar,
            TokenKind.Value,
            TokenKind.Scalar,
            TokenKind.BlockEnd,
            TokenKind.StreamEnd,
        ]
    )
    assert_eq(tokens[3].value, "a")
    assert_true(tokens[3].plain)
    assert_eq(tokens[3].style, null)
    assert_eq(tokens[3].start_mark.line, 0)
    assert_eq(tokens[3].start_mark.column, 0)
    assert_eq(tokens[5].start_mark.column, 3)
    assert_eq(tokens[5].end_mark.column, 4)
    assert_eq(tokens[2].id(), "?")
    assert_eq(tokens[3].id(), "<scalar>")
}

fn test_scan_flow_anchors_tags_and_directives() -> void {
    let tokens = yaml.scan("%YAML 1.2\n--- !!str &a [*b, 'q']")
    assert_eq(
        kinds_of(tokens),
        [
            TokenKind.StreamStart,
            TokenKind.Directive,
            TokenKind.DocumentStart,
            TokenKind.Tag,
            TokenKind.Anchor,
            TokenKind.FlowSequenceStart,
            TokenKind.Alias,
            TokenKind.FlowEntry,
            TokenKind.Scalar,
            TokenKind.FlowSequenceEnd,
            TokenKind.StreamEnd,
        ]
    )
    assert_eq(tokens[1].value, "YAML")
    assert_eq(tokens[1].major, 1)
    assert_eq(tokens[1].minor, 2)
    assert_eq(tokens[3].handle, "!!")
    assert_eq(tokens[3].suffix, "str")
    assert_eq(tokens[4].value, "a")
    assert_eq(tokens[6].value, "b")
    assert_eq(tokens[8].style, "'")
    let directive = yaml.scan("%TAG !e! tag:x.org,2000:\n---")[1]
    assert_eq(directive.handle, "!e!")
    assert_eq(directive.suffix, "tag:x.org,2000:")
}

fn test_parse_produces_pyyaml_events() -> void {
    let events = yaml.parse("- 1\n- &x b\n- *x")
    assert_eq(
        [event.kind for event in events],
        [
            EventKind.StreamStart,
            EventKind.DocumentStart,
            EventKind.SequenceStart,
            EventKind.Scalar,
            EventKind.Scalar,
            EventKind.Alias,
            EventKind.SequenceEnd,
            EventKind.DocumentEnd,
            EventKind.StreamEnd,
        ]
    )
    assert_eq(events[3].value, "1")
    assert_true(events[3].implicit)
    assert_eq(events[4].anchor, "x")
    assert_eq(events[5].anchor, "x")
    assert_eq(events[2].flow_style, false)
    assert_eq(yaml.parse("[1]")[2].flow_style, true)
    let tagged = yaml.parse("!!int '1'")[2]
    assert_eq(tagged.tag, "tag:yaml.org,2002:int")
    assert_eq(tagged.style, "'")
    assert_eq(tagged.implicit, false)
    let quoted = yaml.parse("'1'")[2]
    assert_eq(quoted.implicit, false)
    assert_eq(quoted.implicit_quoted, true)
}

fn test_compose_builds_resolved_nodes() -> void {
    let root = yaml.compose("a: [1, 'x', ~]")
    assert_true(root is not null)
    let mapping = root ?? Node.scalar("", "")
    assert_eq(mapping.kind, NodeKind.Mapping)
    assert_eq(mapping.tag, yaml.MAP_TAG)
    assert_eq(mapping.id(), "mapping")
    let (key, value) = mapping.pairs[0]
    assert_eq(key.tag, yaml.STR_TAG)
    assert_eq(key.value, "a")
    assert_eq(value.kind, NodeKind.Sequence)
    assert_eq(value.flow_style, true)
    assert_eq([item.tag for item in value.items], [yaml.INT_TAG, yaml.STR_TAG, yaml.NULL_TAG])
    assert_eq(value.items[1].style, "'")
    assert_eq(yaml.compose(""), null)
    assert_eq(yaml.compose_all("1\n---\n2").len(), 2)
    assert_eq(yaml.compose_all("").len(), 0)
}

fn test_serialize_writes_nodes() -> void {
    let node = Node.mapping(
        yaml.MAP_TAG,
        [
            (Node.scalar(yaml.STR_TAG, "a"), Node.scalar(yaml.INT_TAG, "1")),
            (
                Node.scalar(yaml.STR_TAG, "b"),
                Node.sequence(yaml.SEQ_TAG, [Node.scalar(yaml.STR_TAG, "12")])
            ),
        ]
    )
    assert_eq(yaml.serialize(node), "a: 1\nb:\n- '12'\n")
    assert_eq(
        yaml.serialize_all([node, Node.scalar(yaml.NULL_TAG, "null")]),
        "a: 1\nb:\n- '12'\n--- null\n...\n"
    )
    assert_eq(yaml.serialize(Node.scalar("!custom", "v")), "!custom 'v'\n")
    let composed = yaml.compose("x: {y: [1, 2]}") ?? Node.scalar("", "")
    assert_eq(yaml.serialize(composed), "x: {y: [1, 2]}\n")
}

fn test_the_serializer_is_driven_in_order() -> void {
    let serializer = yaml.Serializer(yaml.Emitter())
    assert_throws(SerializerError, || serializer.serialize(Node.scalar(yaml.STR_TAG, "a")))
    serializer.open()
    assert_throws(SerializerError, || serializer.open())
    serializer.close()
    assert_throws(SerializerError, || serializer.serialize(Node.scalar(yaml.STR_TAG, "a")))
}

fn test_emit_writes_events_back() -> void {
    for text in ["a: 1\n", "- 1\n- 2\n", "[1, {a: b}]\n", "- |\n  x\n", "'q'\n"] {
        assert_eq(yaml.emit(yaml.parse(text)), text)
    }
    assert_eq(yaml.emit(yaml.parse("a: 1\n---\nb: 2\n")), "a: 1\n---\nb: 2\n")
}

fn test_resolve_follows_the_core_schema() -> void {
    assert_eq(yaml.resolve(NodeKind.Scalar, "12", true), yaml.INT_TAG)
    assert_eq(yaml.resolve(NodeKind.Scalar, "12", false), yaml.STR_TAG)
    assert_eq(yaml.resolve(NodeKind.Scalar, "1.5", true), yaml.FLOAT_TAG)
    assert_eq(yaml.resolve(NodeKind.Scalar, "true", true), yaml.BOOL_TAG)
    assert_eq(yaml.resolve(NodeKind.Scalar, "", true), yaml.NULL_TAG)
    assert_eq(yaml.resolve(NodeKind.Scalar, "<<", true), yaml.MERGE_TAG)
    assert_eq(yaml.resolve(NodeKind.Scalar, "yes", true), yaml.STR_TAG)
    assert_eq(yaml.resolve(NodeKind.Sequence, null, true), yaml.SEQ_TAG)
    assert_eq(yaml.resolve(NodeKind.Mapping, null, true), yaml.MAP_TAG)
    for text in ["0", "-0", "+12", "0o7", "0x0", "0xAbC", "123456789012345678901234567890"] {
        assert_eq(yaml.resolve_plain(text), yaml.INT_TAG, text)
    }
    for text in ["0.", ".0", "1e5", "1E+5", "1.5e-3", "-.inf", "+.INF", ".nan", "1.", "00.5"] {
        assert_eq(yaml.resolve_plain(text), yaml.FLOAT_TAG, text)
    }
    for text in [".", "-", "+", "e5", "1e", "1.5.2", "0x", "0o8", "0o", "-.nan", "inf", "nan", "1,5", "1 "] {
        assert_eq(yaml.resolve_plain(text), yaml.STR_TAG, text)
    }
}

fn test_base64_helpers() -> void {
    assert_eq(yaml.base64_encode(bytes([104, 101, 108, 108, 111])), "aGVsbG8=\n")
    assert_eq(yaml.base64_encode(bytes([])), "")
    assert_eq(yaml.base64_encode(bytes([0, 255])), "AP8=\n")
    assert_eq(yaml.base64_decode("aGVsbG8="), bytes([104, 101, 108, 108, 111]))
    assert_eq(yaml.base64_decode("aGVs\nbG8="), bytes([104, 101, 108, 108, 111]))
    assert_eq(yaml.base64_decode("aGVsbG8"), bytes([104, 101, 108, 108, 111]))
    assert_eq(yaml.base64_decode("a"), null)
    let sample: list[int] = []
    for index in 0..100 { sample.push(index * 7 % 256) }
    let encoded = yaml.base64_encode(bytes(sample))
    assert_eq(encoded.split("\n")[0].len(), 76)
    assert_eq(yaml.base64_decode(encoded), bytes(sample))
}

fn test_marks_render_like_pyyaml() -> void {
    let mark = yaml.Mark("<unicode string>", 5, 1, 2, "ab\ncdefg", 5)
    assert_eq(mark.to_str(), "  in \"<unicode string>\", line 2, column 3:\n    cdefg\n      ^")
    assert_eq(yaml.Mark("n", 0, 0, 0, null, 0).to_str(), "  in \"n\", line 1, column 1")
    let long = "x".repeat(200)
    let snippet = yaml.Mark("n", 100, 0, 100, long, 100).get_snippet() ?? ""
    assert_true(snippet.starts_with("     ... "))
    assert_true(snippet.contains(" ... \n"))
}
