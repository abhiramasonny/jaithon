# jaitoml

TOML 1.0, read as Python's `tomllib` reads it and written as `tomli_w` writes it.

```jai
import jaitoml

let config = jaitoml.loads("[server]\nport = 8080\nhosts = [\"a\", \"b\"]")
print(config["server"]["port"])                      # 8080
print(jaitoml.dumps({"server": {"port": 8080}}))     # [server]\nport = 8080\n

let file = open("pyproject.toml")
defer { file.close() }
let project = jaitoml.load(file)["project"]
```

The surface is Python's, name for name: `loads` and `load` return a
`dict[str, any]`; `dumps` and `dump` take one and accept `multiline_strings`
and `indent`; `TOMLDecodeError` is what a malformed document raises. A program
written against `tomllib` and `tomli_w` translates line for line.

## What comes back

| TOML | Jaithon |
| --- | --- |
| table, inline table | `dict[str, any]`, in document order |
| array | `list[any]` |
| string, in any of the four forms | `str` |
| integer, in any base | `int` |
| float, `inf`, `nan` | `float` |
| boolean | `bool` |
| offset date-time | `jaitoml.DateTime` with `offset_minutes` set |
| local date-time | `jaitoml.DateTime` with `offset_minutes` null |
| local date | `jaitoml.Date` |
| local time | `jaitoml.Time` |

The three calendar classes are shaped like `datetime.datetime`, `datetime.date`
and `datetime.time`, and are the package's own: `std.time.DateTime` is always
a moment with an offset and so has nowhere to put a local date-time, a date
without a time, or a time without a date, all three of which TOML has. A
`DateTime` with an offset converts with `to_std()`. Fractional seconds are
microseconds; the seventh digit onward is dropped, as `tomllib` drops it.
`to_iso()` is Python's `isoformat()` and `to_str()` is Python's `str()`, which
is the form `dumps` writes. Equality and `compare` follow Python: offset
date-times compare by instant, a local one never equals an offset one, and
comparing the two raises `TypeError`.

## Errors

Every malformed document raises `TOMLDecodeError`, a `ValueError`, with the
message `tomllib` gives at the position `tomllib` reports:

```
Cannot overwrite a value (at line 2, column 6)
Cannot declare ('a', 'b') twice (at line 3, column 5)
Unescaped '\' in a string (at line 1, column 8)
Invalid value (at end of document)
```

Duplicate keys, a table declared twice, a dotted key that reopens a
`[table]`, a write into an inline table or an array, an `[[array]]` where a
table already stands: each is refused, and the message names the namespace as
a Python tuple, the way `tomllib` prints it. `lineno`, `colno`, `pos`, `msg`
and `doc` are fields on the error, which Python only added in 3.14; the
message text is 3.13's.

## Writing

`dumps` reproduces `tomli_w` 1.2 byte for byte, so its layout rules are
`tomli_w`'s: the scalars of a table first, then each nested dict as a
`[section]` after a blank line; a list of dicts as `[[sections]]` unless every
one of them fits an inline table in a hundred columns; every other array one
element per line with a trailing comma, indented `indent` spaces per level;
a key that is not bare quoted like a string; a string holding a newline as a
`"""` block when `multiline_strings` is on. A null, a set, a class instance
other than the three calendar types, or a key that is not a string raises
`TypeError` with `tomli_w`'s message.

## Deviations from Python

- **Integers are 64-bit.** An integer literal outside that range raises
  `TOMLDecodeError` (`Integer is outside the 64-bit range this runtime can
  hold`) where Python would return a big int. TOML itself promises only 64
  bits, so no valid document changes meaning; a document with a larger
  literal fails here, and it fails at that literal, which may be earlier than
  a later error `tomllib` would have reported first.
- **`load` and `dump` take a text `std.io.File`.** Python's `load` insists on
  a binary handle and decodes it as UTF-8 itself; `std.io` has already done
  that. Neither strips a byte-order mark, and neither does `tomllib`.
- **`TOMLDecodeError` carries `lineno`, `colno`, `pos`, `msg` and `doc`** from
  the start, as Python 3.14's does. Its message is exactly 3.13's.
- **A module-qualified name cannot follow `catch`.** Write
  `from jaitoml import TOMLDecodeError` and `catch e: TOMLDecodeError`; the
  checker does not accept `catch e: jaitoml.TOMLDecodeError`.
- **`parse_float` receives the literal with its underscores**, as in Python;
  its result may be anything but a list or a dict.

## Layout

| File | Covers |
| --- | --- |
| `datetime.jai` | `Date`, `Time`, `DateTime`, `MIN_YEAR`, `MAX_YEAR` |
| `decoder.jai` | `loads`, `load`, `TOMLDecodeError` |
| `encoder.jai` | `dumps`, `dump`, `MAX_LINE_LENGTH` |

`src/jaitoml/mod.jai` re-exports all of it. Each file also stays independently
importable — `from jaitoml.decoder import loads` keeps working.

## Tests

```
./jaithon test packages/jaitoml/tests
uv run --python 3.13 --with tomli-w python3 packages/jaitoml/tests/py/jaitoml_diff.py
```

The unit tests pin values and messages taken from Python. The differential
script is the refuter: it generates random documents with `tomli_w`, loads
them with `tomllib` and with jaitoml and diffs the results and the re-dumped
text exactly; mutates each document and requires the same verdict and the
same error message from both; and, when the official corpus is cloned at
`/tmp/toml-test` (`git clone --depth 1
https://github.com/toml-lang/toml-test /tmp/toml-test`), runs every TOML 1.0
`valid/` and `invalid/` file and reports pass counts. It exits 1 on any
mismatch.
