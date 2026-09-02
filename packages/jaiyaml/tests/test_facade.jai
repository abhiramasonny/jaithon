#: `import jaiyaml` itself, and the names the facade promises.
#:
#: A package's own tests import its files, so none of them ever exercised
#: the module a user reaches through `import jaiyaml`. This one does, and
#: touches every exported name once, so a facade that names a module that
#: does not exist, or forgets a name, fails here rather than in a program.
import jaiyaml
import jaiyaml as yaml
from jaiyaml import ComposerError, ConstructorError, EmitterError, MarkedYAMLError, ParserError
from jaiyaml import ReaderError, RepresenterError, ScannerError, SerializerError, YAMLError
from std.test import assert_eq, assert_throws, assert_true

fn test_the_facade_loads_and_dumps() -> void {
    assert_eq(jaiyaml.VERSION, "0.1.0")
    assert_eq(jaiyaml.safe_load("a: 1"), {"a": 1})
    assert_eq(jaiyaml.safe_load_all("1\n---\n2"), [1, 2])
    assert_eq(jaiyaml.safe_dump({"a": 1}), "a: 1\n")
    assert_eq(jaiyaml.safe_dump_all([1, 2]), "1\n--- 2\n...\n")
}

fn test_every_exported_name_is_reachable() -> void {
    assert_eq(yaml.scan("a").len(), 3)
    assert_eq(yaml.parse("a").len(), 5)
    assert_true(yaml.compose("a") is not null)
    assert_eq(yaml.compose_all("a").len(), 1)
    assert_eq(yaml.serialize(yaml.Node.scalar(yaml.STR_TAG, "a")), "a\n...\n")
    assert_eq(yaml.serialize_all([]), "")
    assert_eq(yaml.emit(yaml.parse("a\n")), "a\n...\n")
    assert_eq(yaml.resolve(yaml.NodeKind.Scalar, "1", true), yaml.INT_TAG)
    assert_eq(yaml.resolve_plain("x"), yaml.DEFAULT_SCALAR_TAG)
    assert_eq(yaml.base64_decode(yaml.base64_encode(bytes([1, 2, 3]))), bytes([1, 2, 3]))
    assert_eq(yaml.STRING_NAME, "<unicode string>")
    assert_eq(yaml.DEFAULT_SEQUENCE_TAG, yaml.SEQ_TAG)
    assert_eq(yaml.DEFAULT_MAPPING_TAG, yaml.MAP_TAG)
    assert_eq(
        [yaml.NULL_TAG, yaml.BOOL_TAG, yaml.FLOAT_TAG, yaml.SET_TAG, yaml.MERGE_TAG, yaml.BINARY_TAG].len(),
        6
    )
    assert_eq([yaml.OMAP_TAG, yaml.PAIRS_TAG].len(), 2)
    assert_eq(
        yaml.Token(yaml.TokenKind.Key, yaml.Mark("n", 0, 0, 0, null, 0), yaml.Mark("n", 0, 0, 0, null, 0)).id(),
        "?"
    )
    assert_eq(yaml.Event(yaml.EventKind.Scalar).kind, yaml.EventKind.Scalar)
    assert_eq(yaml.Reader("ab").peek(1), "b")
    assert_true(yaml.Scanner("a").check_token())
    assert_true(yaml.Parser("a").check_event())
    assert_true(yaml.Composer("a").check_node())
    assert_true(yaml.Constructor("a").check_data())
    assert_eq(yaml.Representer().represent(1).value, "1")
    let emitter = yaml.Emitter()
    let serializer = yaml.Serializer(emitter)
    serializer.open()
    serializer.close()
    assert_eq(emitter.result(), "")
}

fn test_every_error_class_is_a_yaml_error() -> void {
    assert_throws(ScannerError, || yaml.safe_load("a: b: c"))
    assert_throws(ParserError, || yaml.safe_load("[1"))
    assert_throws(ComposerError, || yaml.safe_load("*x"))
    assert_throws(ConstructorError, || yaml.safe_load("!!int x"))
    assert_throws(ReaderError, || yaml.safe_load("\u{1}"))
    assert_throws(RepresenterError, || yaml.safe_dump(yaml.Emitter()))
    assert_throws(SerializerError, || yaml.Serializer(yaml.Emitter()).close())
    assert_throws(EmitterError, || yaml.Emitter().emit(yaml.Event(yaml.EventKind.Scalar)))
    assert_throws(MarkedYAMLError, || yaml.safe_load("a: b: c"))
    assert_throws(YAMLError, || yaml.safe_load("a: b: c"))
    assert_throws(YAMLError, || yaml.safe_load("\u{1}"))
    assert_throws(YAMLError, || yaml.safe_dump(yaml.Emitter()))
    assert_throws(Error, || yaml.safe_load("a: b: c"))
}
