#: What one checker decided, in a form the other can be diffed against. The C
#: writer is `src/sema/semadump.c` and the format is normative between them. A
#: record is `start end kind field value`, the span in BYTE offsets and `value`
#: the rest of the line, spaces included. Records are keyed on span, not node
#: index: two independently written front ends do not number AST nodes alike.
from std.algo.sort import natural_order, sort_by
from std.str import to_int

#: Format version. Must equal `JAI_SEMADUMP_VERSION` in `src/sema/semadump.h`.
pub const VERSION: int = 1

#: The four fields: `type`, `fold`, `guard` (`any->T`), `widen` (`int->float`).
pub let FIELDS: list[str] = ["fold", "guard", "type", "widen"]

let _HEADER_VERSION: str = "# jaithon sema dump v"
let _HEADER_FILE: str = "# file "

#: One decision, about the node covering `[span_start, span_end)`.
pub class Record {
    #: First byte of the node's span.
    pub let span_start: int
    #: One past the last byte of the node's span.
    pub let span_end: int
    #: The AST kind name, as `jaiAstKindName` spells it.
    pub let kind: str
    #: One of `FIELDS`.
    pub let field: str
    #: The decision, rendered. Never empty, and never contains a newline.
    pub let value: str

    #: Record that `kind` at `[span_start, span_end)` decided `field` = `value`.
    pub fn init(self, span_start: int, span_end: int, kind: str, field: str, value: str) {
        self.span_start = span_start
        self.span_end = span_end
        self.kind = kind
        self.field = field
        self.value = value
    }

    #: What identifies this decision, without saying what was decided.
    pub fn key(self) -> str {
        return f"{self.span_start} {self.span_end} {self.kind} {self.field}"
    }

    fn __str__(self) -> str {
        return f"{self.key()} {self.value}"
    }

    fn __eq__(self, other: any) -> bool {
        if other is not Record { return false }
        return self.key() == other.key() and self.value == other.value
    }
}

#: Every decision one checker made about one file.
pub class Dump {
    #: The path the header names.
    pub let source_path: str
    #: The records, in the order they were read.
    pub let records: list[Record]

    #: Hold `records` as the dump of `source_path`.
    pub fn init(self, source_path: str, records: list[Record]) {
        self.source_path = source_path
        self.records = records
    }
}

#: One decision the two checkers did not agree on. `ours` and `theirs` are null
#: when that side had no record for this key at all.
pub class Mismatch {
    pub let span_start: int
    pub let span_end: int
    pub let kind: str
    pub let field: str
    pub let ours: str?
    pub let theirs: str?

    pub fn init(
        self,
        span_start: int,
        span_end: int,
        kind: str,
        field: str,
        ours: str?,
        theirs: str?
    ) {
        self.span_start = span_start
        self.span_end = span_end
        self.kind = kind
        self.field = field
        self.ours = ours
        self.theirs = theirs
    }

    fn __str__(self) -> str {
        let mine = self.ours ?? "<none>"
        let other = self.theirs ?? "<none>"
        return f"{self.span_start}-{self.span_end} {self.kind} {self.field}: " +
            f"C says {other}, jai says {mine}"
    }
}

fn _order(a: Record, b: Record) -> int {
    if a.span_start != b.span_start { return natural_order(a.span_start, b.span_start) }
    if a.span_end != b.span_end { return natural_order(a.span_end, b.span_end) }
    if a.kind != b.kind { return natural_order(a.kind, b.kind) }
    if a.field != b.field { return natural_order(a.field, b.field) }
    return natural_order(a.value, b.value)
}

#: Read a dump. Raises `ValueError` on anything it does not understand, naming
#: the line. Skipping bad input would turn a format mismatch into a silent pass.
pub fn parse_dump(text: str) -> Dump {
    let lines = text.split("\n")
    if lines.len() < 2 {
        throw ValueError("a sema dump needs a version line and a file line")
    }

    if not lines[0].starts_with(_HEADER_VERSION) {
        throw ValueError(f"line 1: expected `{_HEADER_VERSION}N`, found `{lines[0]}`")
    }
    let version = to_int(lines[0][_HEADER_VERSION.len():])
    if version != VERSION {
        throw ValueError(f"line 1: this is a v{version ?? -1} dump and this reader is v{VERSION}")
    }

    if not lines[1].starts_with(_HEADER_FILE) {
        throw ValueError(f"line 2: expected `{_HEADER_FILE}PATH`, found `{lines[1]}`")
    }
    let source_path = lines[1][_HEADER_FILE.len():]

    var records: list[Record] = []
    for (offset, line) in lines[2:].enumerate() {
        let number = offset + 3
        if line == "" { continue }
        # At most five, so a value keeps whatever spaces it contains.
        let parts = line.split(" ", 4)
        if parts.len() != 5 {
            throw ValueError(f"line {number}: expected 5 fields, found {parts.len()}: `{line}`")
        }
        let span_start = to_int(parts[0])
        let span_end = to_int(parts[1])
        if span_start == null or span_end == null {
            throw ValueError(f"line {number}: `{parts[0]} {parts[1]}` is not a span")
        }
        records.push(Record(span_start, span_end, parts[2], parts[3], parts[4]))
    }
    return Dump(source_path, records)
}

#: Write a dump, sorted, so one program always produces one file.
pub fn render_dump(dump: Dump) -> str {
    var lines: list[str] = [f"{_HEADER_VERSION}{VERSION}", f"{_HEADER_FILE}{dump.source_path}"]
    for record in sort_by(dump.records, _order) {
        lines.push(f"{record}")
    }
    return "\n".join(lines) + "\n"
}

#: Every decision the two dumps do not agree on, in dump order. `ours` is the
#: self-hosted side, `theirs` the C one — the opposite of `--bootstrap-verify`.
pub fn diff_dumps(ours: Dump, theirs: Dump) -> list[Mismatch] {
    var mine: dict[str, str] = {}
    var other: dict[str, str] = {}
    var seen: dict[str, Record] = {}

    for record in ours.records {
        mine[record.key()] = record.value
        seen[record.key()] = record
    }
    for record in theirs.records {
        other[record.key()] = record.value
        if record.key() not in seen { seen[record.key()] = record }
    }

    var found: list[Mismatch] = []
    for record in sort_by([seen[key] for key in seen.keys()], _order) {
        let key = record.key()
        let a = if key in mine { mine[key] } else { null }
        let b = if key in other { other[key] } else { null }
        if a == b { continue }
        found.push(Mismatch(record.span_start, record.span_end, record.kind, record.field, a, b))
    }
    return found
}
