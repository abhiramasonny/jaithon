#: `jaithon.tool.args` — the one command-line convention `fmt`, `test`, `doc` and
#: `bench` share. The point of these tests is that both spellings of a
#: value-taking option mean the same thing, whichever side of the CLI boundary
#: wrote them, and that everything else is a usage error the user hears about
#: rather than a flag that quietly does nothing.
from std.test import assert_eq, assert_true, assert_false
from jaithon.tool.args import Argument, Arguments

fn test_a_path_is_not_a_flag() -> void {
    assert_false(Argument("lib/std").is_flag())
    assert_eq(Argument("lib/std").name, "")
    assert_true(Argument("lib/std").value is null)
    assert_false(Argument("-").is_flag())
    assert_false(Argument("--").is_flag())
}

fn test_both_spellings_split_to_the_same_name() -> void {
    assert_eq(Argument("--out").name, "--out")
    assert_eq(Argument("--out=docs").name, "--out")
    assert_true(Argument("--out").value is null)
    assert_eq(Argument("--out=docs").value, "docs")
    assert_eq(Argument("--out=").value, "")
    assert_eq(Argument("--filter=a=b").value, "a=b")
}

fn test_a_value_may_be_attached_or_follow() -> void {
    let attached = Arguments(["--out=docs", "lib"])
    assert_eq(attached.value_of(attached.next()), "docs")
    assert_eq(attached.next().text, "lib")
    assert_true(attached.error is null)

    let separate = Arguments(["--out", "docs", "lib"])
    assert_eq(separate.value_of(separate.next()), "docs")
    assert_eq(separate.next().text, "lib")
    assert_true(separate.error is null)
}

fn test_a_missing_value_is_a_usage_error() -> void {
    let empty = Arguments(["--out"])
    assert_true(empty.value_of(empty.next()) is null)
    assert_eq(empty.error, "--out needs a value")

    let flag = Arguments(["--filter", "--verbose"])
    assert_true(flag.value_of(flag.next()) is null)
    assert_eq(flag.error, "--filter needs a value")
    assert_eq(flag.next().name, "--verbose")
}

fn test_a_value_on_a_boolean_flag_is_a_usage_error() -> void {
    let reader = Arguments([])
    assert_true(reader.is_bare(Argument("--html")))
    assert_true(reader.error is null)
    assert_false(reader.is_bare(Argument("--html=yes")))
    assert_eq(reader.error, "--html takes no value")
}

fn test_only_the_first_error_is_kept() -> void {
    let reader = Arguments([])
    reader.unknown(Argument("--bogus"))
    reader.unknown(Argument("--worse"))
    assert_eq(reader.error, "unknown option \"--bogus\"")
}

fn test_the_cursor_runs_out_rather_than_wrapping() -> void {
    let reader = Arguments(["a"])
    assert_eq(reader.next().text, "a")
    assert_true(reader.next() is null)
    assert_true(reader.next() is null)
}
