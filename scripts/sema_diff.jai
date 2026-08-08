#: Compare what the two front ends' checkers decided about one file.
#:
#: `--bootstrap-verify` answers in bytecode, which is the wrong vocabulary for a
#: type bug: it reports the first byte that differs, and the decision that was
#: wrong is somewhere upstream of it. This reports the decisions.
#:
#: ```text
#: jaithon run scripts/sema_diff.jai C_DUMP JAI_DUMP   # compare two dumps
#: jaithon run scripts/sema_diff.jai C_DUMP            # summarise one
#: ```
#:
#: Both dumps are produced by `jaithon check --dump-sema PATH FILE`. The
#: self-hosted front end cannot produce one until it has a checker (Phase 2)
#: and until `--front=jai` implements `check` (Phase 6), which is why one
#: argument is a supported mode and not a usage error: until then the useful
#: question is what the C checker decided, because that is the work the
#: self-hosted checker has to reproduce.
from jaithon.compile.check.dump import Dump, diff_dumps, parse_dump
from jaithon.compile.mod import read_text
from std.os import argv, exit

fn _load(path: str) -> Dump {
    return parse_dump(read_text(path))
}

#: How many records of each field a dump holds, in a stable order.
fn _by_field(dump: Dump) -> list[tuple[str, int]] {
    var counts: dict[str, int] = {}
    for record in dump.records {
        counts[record.field] = (if record.field in counts { counts[record.field] } else { 0 }) + 1
    }
    var out: list[tuple[str, int]] = []
    for field in counts.keys().sorted() {
        out.push((field, counts[field]))
    }
    return out
}

fn _summarise(path: str) -> void {
    let dump = _load(path)
    print(f"{dump.source_path}: {dump.records.len()} decisions")
    for (field, count) in _by_field(dump) {
        print(f"  {field}  {count}")
    }
}

fn _compare(theirs_path: str, ours_path: str) -> void {
    let theirs = _load(theirs_path)
    let ours = _load(ours_path)
    let found = diff_dumps(ours, theirs)
    if found.len() == 0 {
        print(
            f"ok   {theirs.source_path}: the two checkers agree on " +
                f"{theirs.records.len()} decisions"
        )
        return
    }
    for mismatch in found {
        print(f"{mismatch}")
    }
    print(f"{found.len()} mismatches")
    exit(1)
}

fn main() -> void {
    let args = argv()
    if args.len() == 2 {
        _summarise(args[1])
        return
    }
    if args.len() == 3 {
        _compare(args[1], args[2])
        return
    }
    print("usage: jaithon run scripts/sema_diff.jai C_DUMP [JAI_DUMP]")
    exit(2)
}
