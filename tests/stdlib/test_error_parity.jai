#: Every file in `tests/errors` must be rejected by the self-hosted front end,
#: with the same code the C front end reports.
#:
#: `--bootstrap-verify` cannot see this. It compares bytecode, and a file the C
#: rejects produces none — those 36 files are reported as "not comparable" and
#: contribute nothing. So the whole diagnostic surface of the self-hosted front
#: end was unmeasured: it could have accepted every one of them and the
#: bootstrap number would not have moved.
#:
#: The expected code is the one the file's own first line names, which is what
#: the C is held to as well (`scripts/run_tests.sh`, the diagnostics layer).
from std.test import assert_eq, assert_true
from jaithon.compile.mod import compile_source, import_cycles, read_text
from std.io import Path

#: Files the self-hosted front end does not yet reject, and why.
#:
#: Empty, and it stays a list rather than being deleted: the two entries it
#: held were the module graph's, and this is where the next gap gets recorded.
#: A file named here that starts being rejected fails the test until it is
#: removed, so the list cannot rot into a quietly weaker check.
pub let NOT_YET_REJECTED: list[str] = []

fn _expected_code(source: str) -> str {
    let first = source.split("\n")[0]
    let at = first.find("E0")
    if at < 0 { return "" }
    return first[at:at + 5]
}

#: The first error `jaithon check` would report for `path`.
#:
#: The import graph first and the front end second, which is the order
#: `jaiCheckFile` runs them in and the reason it runs them in that order: every
#: name a module in a cycle should have exported is missing, so the E0200 storm
#: that follows a cycle is a consequence rather than a second problem.
fn _first_error(path: str) -> str? {
    for diagnostic in import_cycles(path).items {
        if diagnostic.is_error() { return diagnostic.code }
    }
    let built = compile_source(read_text(path), path)
    if built.ok() { return null }
    for diagnostic in built.diagnostics.items {
        if diagnostic.is_error() { return diagnostic.code }
    }
    return null
}

fn _error_files() -> list[str] {
    var out: list[str] = []
    for entry in Path("tests/errors").glob("*.jai") {
        out.push(entry.text)
    }
    return out.sorted()
}

fn test_every_error_file_is_rejected_with_the_right_code() -> void {
    var checked = 0
    for path in _error_files() {
        let name = path.split("/")[path.split("/").len() - 1]
        let expected = _expected_code(read_text(path))
        if expected == "" { continue }
        let actual = _first_error(path)

        if name in NOT_YET_REJECTED {
            assert_true(
                actual is null,
                f"{name} is on the not-yet-rejected list but was rejected with {actual}; " +
                    "remove it from NOT_YET_REJECTED"
            )
            continue
        }

        assert_true(actual is not null, f"{name}: expected {expected}, but it was accepted")
        assert_eq(actual, expected, f"{name}")
        checked += 1
    }
    # A directory walk that silently found nothing would pass every assertion
    # above without testing anything.
    assert_true(checked >= 30, f"only {checked} error files were checked")
}

fn test_the_gap_list_names_only_files_that_exist() -> void {
    var present: set[str] = set()
    for path in _error_files() {
        present.add(path.split("/")[path.split("/").len() - 1])
    }
    for name in NOT_YET_REJECTED {
        assert_true(name in present, f"{name} is listed but no such error file exists")
    }
}
