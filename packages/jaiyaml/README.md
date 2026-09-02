# jaiyaml

YAML reading and writing with PyYAML's API.

```jai
import jaiyaml as yaml
from jaiyaml import YAMLError

let config = yaml.safe_load("name: demo\nports: [80, 443]\n")
print(config["ports"][1])                       # 443
print(yaml.safe_dump({"a": 1, "b": [true, null]}))
try {
    yaml.safe_load("a: [1, 2")
} catch error: YAMLError {
    print(f"{error.line}:{error.column} {error.message}")
}
```

`safe_load`, `safe_load_all`, `safe_dump` and `safe_dump_all` take the same
arguments as PyYAML's and produce the same values and the same text. The
dumper is a method-for-method port of PyYAML's representer, serializer and
emitter, so `safe_dump` writes the bytes `yaml.safe_dump` writes for the same
value and options -- flow and block styles, quoting choices, folding at
`width`, `indent`, `sort_keys`, `allow_unicode`, `default_style`, `canonical`,
`explicit_start`, `explicit_end`, `line_break`, `version` and `tags`. The
loader is a port of PyYAML's reader, scanner, parser, composer and safe
constructor, so it accepts and rejects the same documents and raises the same
errors with the same messages: the problem, its line and column, the context
it appeared in, and a caret under the source line.

## What is ported

- Block and flow mappings and sequences, indentless sequences, complex `?`
  keys, empty values, duplicate keys (last wins).
- Plain, single-quoted and double-quoted scalars with every escape PyYAML
  knows (`\0 \a \b \t \n \v \f \r \e \  \" \\ \/ \N \_ \L \P \xNN \uNNNN
  \UNNNNNNNN`), and their line folding.
- Literal `|` and folded `>` block scalars with `-`, `+` and `1`-`9`
  indicators.
- Anchors `&`, aliases `*`, shared and self-referencing structures, `<<`
  merge keys in PyYAML's order.
- Comments, `---` and `...`, multiple documents, `%YAML` and `%TAG`
  directives, a leading byte order mark, `\r`, `\r\n`, `\x85`, ` ` and
  ` ` line breaks.
- Explicit tags `!!null`, `!!bool`, `!!int`, `!!float`, `!!str`, `!!seq`,
  `!!map`, `!!set` (a set), `!!binary` (bytes), `!!omap` and `!!pairs` (a list
  of pairs), verbatim `!<...>` tags and `%TAG` handles.
- Tag resolution for untagged plain scalars by the **YAML 1.2 core schema**:
  `null`/`Null`/`NULL`/`~`/empty, `true`/`True`/`TRUE`/`false`/`False`/`FALSE`,
  decimal, `0o` and `0x` ints, floats with an optional exponent, `.inf` with
  an optional sign and `.nan`, `<<`, and str for everything else.
- Every error class: `YAMLError`, `MarkedYAMLError`, `ReaderError`,
  `ScannerError`, `ParserError`, `ComposerError`, `ConstructorError`,
  `RepresenterError`, `SerializerError`, `EmitterError`, and `Mark`.
- The stage-by-stage surface: `scan`, `parse`, `compose`, `compose_all`,
  `serialize`, `serialize_all`, `emit`, and the `Reader`, `Scanner`,
  `Parser`, `Composer`, `Constructor`, `Representer`, `Serializer` and
  `Emitter` classes.

## What deliberately differs from PyYAML

PyYAML implements YAML 1.1; this package resolves by the 1.2 core schema, as
asked. Every consequence is listed here, and the module docs repeat the ones
that apply to them.

| Text | PyYAML | jaiyaml |
| --- | --- | --- |
| `yes`, `no`, `on`, `off` (any case) | bool | str |
| `010` | int 8 (octal) | int 10 |
| `0o17` | str | int 15 |
| `1e5`, `1E5` | str | float |
| `1_000`, `0b101`, `1:30` | int | str |
| `2001-12-14`, `2001-12-14 21:59:43.10 -5` | date, datetime | str |
| `=` | the `value` key | str |

Because the dumper uses the same resolver, it quotes `'0o17'` and `'1e5'`
where PyYAML writes them plain, and writes `yes` and `1_000` plain where
PyYAML quotes them. A document that uses none of those spellings dumps and
loads identically on both sides.

The rest are consequences of the language:

- A mapping key that is null raises `ConstructorError`; a dict here cannot
  hold one. PyYAML gives `{None: ...}`.
- An int outside 64 bits loads as a float; PyYAML widens to a long. A key of
  such a value is a float key.
- A scalar explicitly tagged `!!int`, `!!float` or `!!bool` whose text is not
  one raises `ConstructorError` with its mark; PyYAML lets `int()` or
  `float()` raise a bare `ValueError` (a `KeyError` for `!!bool`).
- `!!timestamp` is not constructed (there is no datetime to build); it raises
  `ConstructorError` like any other unknown tag.
- `safe_load_all` returns a list, not a generator.
- The input is a `str`; there is no file or bytes stream argument, and
  `safe_dump` always returns its text rather than writing to a stream.
- A `\uD800`-`\uDFFF` escape in a double-quoted scalar raises
  `ScannerError`; Python builds a lone surrogate.
- `Error.init` is called directly in the error classes rather than through
  `super`, because `super` here resolves from the object's own class, and a
  grandchild would otherwise call its parent's `init` from inside itself.
- Catch clauses need an unqualified class name: write
  `from jaiyaml import YAMLError` and `catch e: YAMLError`, not
  `catch e: yaml.YAMLError`.

`load`, `dump`, `full_load`, `unsafe_load`, the `Loader`/`Dumper` classes and
the `add_*` registration hooks are absent: they exist in PyYAML to construct
arbitrary Python objects, which there is nothing to construct here.

## Layout

| File | Covers |
| --- | --- |
| `error.jai` | `Mark`, the error classes, Python's `repr` of a string for messages |
| `tokens.jai`, `events.jai`, `nodes.jai` | the three intermediate forms |
| `reader.jai` | the character cursor and the printable check |
| `scanner.jai` | text to tokens, simple keys, indentation |
| `parser.jai` | tokens to events, the LL(1) grammar |
| `composer.jai` | events to nodes, anchors and aliases |
| `resolver.jai` | the core-schema tag resolution |
| `constructor.jai` | nodes to values, merge keys, base64 |
| `representer.jai` | values to nodes, aliases for shared objects, Python's key ordering |
| `serializer.jai` | nodes to events, anchor naming |
| `emitter.jai` | events to text, scalar analysis, folding |
| `mod.jai` | the facade: `safe_load`, `safe_dump` and the rest |

Nothing here is seeded; the package compiles from source on first import.

## Tests

```
./jaithon test packages/jaiyaml/tests
```

One file per concern: `test_load.jai`, `test_dump.jai`, `test_stream.jai`
and `test_facade.jai` (which imports `jaiyaml` itself, not its files). Every
expected value and every expected string came from running the same input
through PyYAML.

## The differential test

```
python3 packages/jaiyaml/tests/py/jaiyaml_diff.py
python3 packages/jaiyaml/tests/py/jaiyaml_diff.py --count 1000 --seed 7
```

The script re-executes itself under `uv run --python 3.13 --with pyyaml`
when PyYAML is not importable. Each of the 400 default cases is a random
nested value dumped by PyYAML with random options; it checks that jaiyaml
loads that text to the same structure, that jaiyaml's dump of what it loaded
equals PyYAML's dump byte for byte, that PyYAML loads jaiyaml's dump back to
the same structure, and that jaiyaml loads PyYAML's dump of that back to the
same structure again. Multi-document streams go through `safe_dump_all` and
`safe_load_all`. Each case is then mutated by one character and both sides
must agree on whether it raises, on the error class and full message when it
does, and on the structure when it does not.

The string pool holds only texts the 1.1 and 1.2 resolvers classify alike;
the mutation cases, which can produce any plain scalar, are loaded on the
Python side through the real `SafeLoader` configured with the core schema's
implicit resolvers. The two language deviations above -- a null key and a
bare `ValueError` from an explicitly tagged non-number -- are recognised and
counted as known deviations rather than failures.

Last runs: the default seed gave 400 documents, 392 mutants, 1985 checks, 7
known deviations, 0 failures; `--count 1000 --seed 3` gave 1000 documents,
982 mutants, 4947 checks, 35 known deviations, 0 failures.
