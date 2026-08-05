#: A release manifest, from disk to objects and back again.
#:
#: The round trip is the whole example. `std.io` supplies the filesystem —
#: `Path` for locations, `write_file`/`read_file` for text, `write_csv` for the
#: report — and `std.json` supplies the format, at all three of the levels it
#: offers:
#:
#:   - `parse` / `stringify` for a whole document,
#:   - `to_class` / `from_class` to move between JSON objects and real objects,
#:   - `Parser.next_event` to walk a document without building it, which is how
#:     you measure or validate something too large to hold in memory.
#:
#: It writes into a directory of its own (a temporary one unless you name
#: another), then reads back everything it wrote and checks it survived.
#:
#:     jaithon run examples/manifest.jai [output-dir]
import std.io
import std.json
import std.os
from std.core import Printable
from std.io import Path
from std.json import EventKind

#: Bytes per mebibyte, for the human-readable sizes in the report.
const MIB = 1048576

#: One package in the manifest.
#:
#: The field names are the JSON member names: that is the contract
#: `json.to_class` and `json.from_class` work from, so a rename here is a
#: format change.
class Package: Printable {
    pub var name: str
    pub var version: str
    pub var size_bytes: int
    pub var tags: list[str]

    fn init(self, name: str, version: str, size_bytes: int, tags: list[str]) {
        self.name = name
        self.version = version
        self.size_bytes = size_bytes
        self.tags = tags
    }

    #: Whether this package ships to end users rather than only to CI.
    pub fn is_published(self) -> bool { return not self.tags.contains("internal") }

    pub fn size_mib(self) -> float { return (float(self.size_bytes) / float(MIB)).round(2) }

    pub fn to_str(self) -> str { return f"{self.name}@{self.version}" }

    fn __str__(self) -> str { return self.to_str() }
}

#: The manifest as it is written on disk.
#:
#: Sizes are deliberately uneven so the report has something to say.
pub fn sample_manifest() -> dict[str, any] {
    return {
        "schema": 2,
        "release": {"name": "2026.08", "channel": "stable", "signed": true},
        "packages": [
            {
                "name": "core",
                "version": "3.0.1",
                "size_bytes": 1_204_992,
                "tags": ["stable", "runtime"],
            },
            {
                "name": "compiler",
                "version": "3.0.1",
                "size_bytes": 9_871_360,
                "tags": ["stable", "runtime"],
            },
            {
                "name": "docs",
                "version": "3.0.0",
                "size_bytes": 22_413_312,
                "tags": ["stable"],
            },
            {
                "name": "bench-harness",
                "version": "0.4.2",
                "size_bytes": 331_776,
                "tags": ["internal"],
            },
            {
                "name": "gui",
                "version": "3.0.1",
                "size_bytes": 5_242_880,
                "tags": ["stable", "runtime", "optional"],
            },
        ],
    }
}

#: Turn the `packages` array of a parsed manifest into real objects.
#:
#: Raises `ValueError` when the document is shaped wrongly — a manifest that
#: does not describe packages is not a manifest, and carrying on with an empty
#: list would hide the problem until the report came out blank.
pub fn load_packages(document: any) -> list[Package] {
    let raw = json.get_path(document, "packages")
    if not isinstance(raw, list) {
        throw ValueError("manifest has no `packages` array")
    }

    var packages: list[Package] = []
    for entry in raw {
        if not isinstance(entry, dict) {
            throw ValueError(f"package entry is {type_of(entry)}, expected an object")
        }
        packages.push(json.to_class(Package, entry))
    }
    return packages
}

#: Count the nodes of `text` without building the document.
#:
#: `next_event` is a pull parser: it reports one structural event at a time and
#: never holds more than the current nesting path, so the cost of this is the
#: depth of the document rather than its size.
pub fn survey(text: str) -> dict[str, int] {
    let parser = json.Parser(text)
    var objects = 0
    var arrays = 0
    var keys = 0
    var values = 0
    var depth = 0
    var deepest = 0

    loop {
        let event = parser.next_event()
        if event == null { break }
        match event.kind {
            EventKind.ObjectStart => {
                objects += 1
                depth += 1
                if depth > deepest { deepest = depth }
            }
            EventKind.ArrayStart => {
                arrays += 1
                depth += 1
                if depth > deepest { deepest = depth }
            }
            EventKind.ObjectEnd => { depth -= 1 }
            EventKind.ArrayEnd => { depth -= 1 }
            EventKind.Key => { keys += 1 }
            EventKind.Value => { values += 1 }
        }
    }
    return {"objects": objects, "arrays": arrays, "keys": keys, "values": values, "depth": deepest}
}

#: Build the summary document that gets written back out as JSON.
pub fn summarize(document: any, packages: list[Package]) -> dict[str, any] {
    let published = iter_published(packages)
    var total = 0
    for package in published { total += package.size_bytes }

    var largest = published[0]
    for package in published {
        if package.size_bytes > largest.size_bytes { largest = package }
    }

    return {
        "release": json.get_path(document, "release.name", "unknown"),
        "channel": json.get_path(document, "release.channel", "unknown"),
        "published": published.len(),
        "internal": packages.len() - published.len(),
        "total_bytes": total,
        "largest": json.from_class(largest),
    }
}

#: The rows of the CSV report, header first.
pub fn report_rows(packages: list[Package]) -> list[list[str]] {
    var rows = [["name", "version", "size_bytes", "size_mib", "published", "tags"]]
    for package in packages {
        rows.push(
            [
                package.name,
                package.version,
                package.size_bytes.to_str(),
                package.size_mib().to_str(),
                f"{package.is_published()}",
                "|".join(package.tags),
            ]
        )
    }
    return rows
}

fn iter_published(packages: list[Package]) -> list[Package] {
    var out: list[Package] = []
    for package in packages {
        if package.is_published() { out.push(package) }
    }
    if out.len() == 0 { throw ValueError("every package is internal; nothing to report") }
    return out
}

#: Parse a document that is wrong on purpose, to show what the reader says.
fn show_error_reporting() -> void {
    print("a malformed manifest, for contrast:")
    try {
        let _ignored = json.parse("{\n  \"schema\": 2,\n  \"packages\": [1, 2,]\n}")
        print("    unexpectedly accepted")
    } catch error: ValueError {
        for line in io.split_lines(error.message) { print(f"    {line}") }
    }
}

fn main(args: list[str]) -> int {
    let root = if args.len() > 1 { args[1] } else { os.temp_dir() }
    let directory = Path(root).join("jaithon-manifest")
    directory.mkdir(parents: true, exist_ok: true)
    print(f"working in {directory}\n")

    # --- write, then read back exactly what was written -------------------
    let manifest_path = directory.join("manifest.json")
    # `sort_keys` is what makes the rendering reproducible: without it members
    # keep insertion order, which is stable for one dict but not across two.
    # The trailing newline is not JSON's business, but a text file without one
    # is awkward for every tool that reads lines.
    let written = json.stringify(sample_manifest(), indent: 2, sort_keys: true) + "\n"
    manifest_path.write_text(written)

    let text = manifest_path.read_text()
    print(
        f"wrote and re-read {manifest_path.name()} ({text.len()} bytes on disk: " +
            f"{manifest_path.size()})"
    )
    if text != written { print("    ROUND TRIP FAILED: bytes differ") }

    let document = json.parse(text)
    let reparsed = json.stringify(document, indent: 2, sort_keys: true) + "\n"
    print(f"    parse -> stringify is byte-identical: {reparsed == written}\n")

    # --- walk it without building it --------------------------------------
    let counts = survey(text)
    print("streamed with Parser.next_event, building nothing:")
    print(
        f"    {counts["objects"]} objects, {counts["arrays"]} arrays, " +
            f"{counts["keys"]} keys, {counts["values"]} scalars, " +
            f"nesting depth {counts["depth"]}\n"
    )

    # --- JSON objects to real objects -------------------------------------
    let packages = load_packages(document)
    print(f"{packages.len()} packages:")
    for package in packages {
        let mark = if package.is_published() { " " } else { "-" }
        print(
            f"  {mark} {package.to_str().pad_right(22)} " +
                f"{package.size_mib().to_str().pad_left(7)} MiB  " +
                f"{"|".join(package.tags)}"
        )
    }
    print("")

    # --- and back out, as JSON and as CSV ---------------------------------
    let summary = summarize(document, packages)
    let summary_path = directory.join("summary.json")
    summary_path.write_text(json.stringify(summary, indent: 2, sort_keys: true) + "\n")

    let report_path = directory.join("report.csv")
    io.write_csv(report_path.to_str(), report_rows(packages))

    print(f"wrote {summary_path.name()} and {report_path.name()}")
    print("summary read back:")
    let summary_back = json.parse(summary_path.read_text())
    print(
        f"    release      {json.get_path(summary_back, "release")} " +
            f"({json.get_path(summary_back, "channel")})"
    )
    print(f"    published    {json.get_path(summary_back, "published")}")
    print(
        f"    largest      {json.get_path(summary_back, "largest.name")} at " +
            f"{json.get_path(summary_back, "largest.size_bytes")} bytes"
    )
    print(f"    missing key  {json.get_path(summary_back, "release.nope", "(default)")}")

    let rows = io.read_csv(report_path.to_str())
    print(f"    csv          {rows.len() - 1} data rows, columns {rows[0]}\n")

    show_error_reporting()
    return 0
}
