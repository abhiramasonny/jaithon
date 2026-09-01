#: A frame as text: CSV and JSON on disk, and the printed table on screen.
#:
#: Every path here crosses the boundary between a device allocation and host
#: text, and the whole design is about crossing it once. A reader parses each
#: field into a per-column host lane and hands the finished lane to
#: `Buffer.upload` in a single call; a writer pulls each column down in a
#: single `download` and formats from that. Reaching for `Column.get` or
#: `Column.set` a cell at a time instead would be a device round trip per cell,
#: and a million-row file has enough cells to make that the whole program
#: rather than a detail of it.
#:
#: What is left on the host after that is the formatting itself, and it stays
#: there: the output of a writer is a host string, so turning ten million
#: values into ten million pieces of text is work no dispatch can take away.
#: The device does the two things that are reductions rather than renderings —
#: folding a validity lane into its payload so the writer makes one pass
#: instead of two, and counting the bytes a dictionary column's strings occupy
#: for `memory_usage(deep: true)`.
#:
#: Display never reads the whole frame. `to_string` picks the head and tail
#: rows it is going to show and downloads exactly those slices, so printing a
#: ten-million-row frame costs the same as printing a sixty-row one.
#:
#: ```
#: let frame = read_csv("sales.csv", parse_dates: ["day"])
#: defer { frame.free() }
#: print(to_string(frame))
#: to_csv(frame, "out.csv", float_format: ".2f")
#: to_json(frame, "out.json", orient: "split")     # dtypes and nulls survive
#: ```
from .io.csv import DEFAULT_NA_WORDS, read_csv, read_csv_text, to_csv, to_csv_text
from .io.display import DEFAULT_MAX_COLS, DEFAULT_MAX_ROWS, ELLIPSIS, to_markdown, to_string
from .io.json import JSON_ORIENTS, SCHEMA_MEMBER, read_json, read_json_text, to_json
from .io.json import to_json_text
from .io.memory import buffer_slots, column_bytes, index_slots, info, memory_usage

export { DEFAULT_NA_WORDS, DEFAULT_MAX_ROWS, DEFAULT_MAX_COLS, ELLIPSIS }
export { read_csv_text, read_csv, to_csv_text, to_csv }
export { JSON_ORIENTS, SCHEMA_MEMBER, to_json_text, to_json, read_json_text, read_json }
export { to_string, to_markdown }
export { buffer_slots, index_slots, column_bytes, memory_usage, info }
