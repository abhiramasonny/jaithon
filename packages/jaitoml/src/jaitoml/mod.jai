#: TOML 1.0, read as Python's `tomllib` reads it and written as `tomli_w`
#: writes it.
#:
#: The four names are the four Python gives you: `loads` and `load` return a
#: `dict[str, any]`, `dumps` and `dump` take one. Every rule of the format —
#: the string forms and their escapes, the integer and float spellings, the
#: four date-time shapes, dotted keys, tables, arrays of tables, inline
#: tables, comments — parses to the same value, and every malformed document
#: raises `TOMLDecodeError` with the same message and the same line and
#: column `tomllib` reports.
#:
#: ```
#: import jaitoml
#:
#: let config = jaitoml.loads("[server]\nport = 8080\nhosts = [\"a\", \"b\"]")
#: config["server"]["port"]                              # 8080
#: jaitoml.dumps({"server": {"port": 8080}})            # "[server]\nport = 8080\n"
#: ```
#:
#: Dates and times are `Date`, `Time` and `DateTime` from `datetime.jai`,
#: shaped like Python's `datetime` classes: a `DateTime` with a null
#: `offset_minutes` is a local date-time, one with an offset is an offset
#: date-time, and `DateTime.to_std` converts the latter to
#: `std.time.DateTime`.
#:
#: What differs from Python, and why, is listed in the package README: an
#: integer outside 64 bits is an error rather than a big int, `load` and
#: `dump` take a text `std.io.File`, and `TOMLDecodeError` already carries
#: the `lineno` and `colno` Python only added in 3.14.
#:
#: This facade is the contract each module owes the package. Each file stays
#: independently importable too — `from jaitoml.decoder import loads` keeps
#: working — so the facade is for callers who want the package rather than
#: the file.
from .datetime import MAX_YEAR, MIN_YEAR, Date, DateTime, Time
from .decoder import TOMLDecodeError, load, loads
from .encoder import MAX_LINE_LENGTH, dump, dumps

export { loads, load, dumps, dump, TOMLDecodeError }
export { Date, Time, DateTime, MIN_YEAR, MAX_YEAR, MAX_LINE_LENGTH }

#: The package version, which `jaithon.package.json` also carries.
pub let VERSION = "0.1.0"
