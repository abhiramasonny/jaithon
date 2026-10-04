#: Source that is not well-formed UTF-8 is refused, not compiled.
#:
#: The lexer scans the source's bytes and slices the `str` by offsets it derives
#: from them, which is only sound for whole UTF-8 sequences. Every reader in the
#: language validates, but the driver hands a file's raw bytes to the front end,
#: so the lexer validates first. Without that a stray continuation byte passed as
#: part of a name and the program ran, and a lone lead byte made the lexer slice
#: through a sequence and fail inside the VM with an internal error.
#:
#: Only a file read by the driver can carry such bytes, so each case is written
#: to disk and given to this same executable, which inherits the JIT settings.
from std.test import assert_eq, assert_contains, assert_true
from std.io import Path
from std.os import argv, temp_dir, pid, run
from jaithon.compile.mod import write_bytes

fn _ascii(text: str) -> list[int] {
    var out: list[int] = []
    for b in text.bytes() { out.push(b) }
    return out
}

fn _file(name: str, data: list[int]) -> str {
    let path = f"{temp_dir()}/jaithon_utf8_{pid()}_{name}.jai"
    write_bytes(path, data)
    return path
}

#: `check` and, when `run_too`, `run` both refuse `data`, naming the encoding.
fn _refused(name: str, data: list[int], run_too: bool = false) -> void {
    let path = _file(name, data)
    defer { Path(path).remove() }
    let checked = run([argv()[0], "check", "--no-cache", path])
    assert_true(checked.exit_code != 0, f"check accepted {name}")
    assert_contains(checked.stderr, "invalid UTF-8")
    if run_too {
        let ran = run([argv()[0], "run", "--no-cache", path])
        assert_true(ran.exit_code != 0, f"run accepted {name}")
        assert_eq(ran.stdout, "")
        assert_contains(ran.stderr, "invalid UTF-8")
    }
}

fn test_a_continuation_byte_inside_a_name_is_refused() -> void {
    _refused(
        "continuation",
        _ascii("let x") + [0x80] + _ascii(" = 41\nprint(x") + [0x80] + _ascii(" + 1)\n"),
        run_too: true
    )
}

fn test_an_invalid_lead_byte_inside_a_string_is_refused() -> void {
    _refused("bad_lead", _ascii("let a = \"") + [0xff] + _ascii("\"\nprint(a)\n"), run_too: true)
}

fn test_an_encoded_surrogate_is_refused() -> void {
    _refused("surrogate", _ascii("let ") + [0xed, 0xa0, 0x80] + _ascii(" = 1\nprint(1)\n"))
}

fn test_an_overlong_encoding_is_refused() -> void {
    _refused("overlong", _ascii("print(\"") + [0xc0, 0xaf] + _ascii("\")\n"))
}

fn test_a_sequence_truncated_by_the_end_of_the_file_is_refused() -> void {
    _refused("truncated", _ascii("print(1)\n") + [0xe2, 0x82])
}

fn test_a_lone_lead_byte_inside_a_comment_is_refused() -> void {
    _refused("comment", _ascii("# lone lead ") + [0xc3] + _ascii("\nprint(1)\n"))
}

fn test_a_code_point_past_u_10ffff_is_refused() -> void {
    _refused("beyond_max", _ascii("print(\"") + [0xf4, 0x90, 0x80, 0x80] + _ascii("\")\n"))
}

fn test_well_formed_non_ascii_source_still_runs() -> void {
    let path = _file(
        "valid",
        _ascii("let caf") +
            [0xc3, 0xa9] +
            _ascii(" = 1 # ") +
            [0xe2, 0x82, 0xac] +
            _ascii("\nprint(\"") +
            [0xf0, 0x9f, 0x98, 0x80] +
            _ascii("\", caf") +
            [0xc3, 0xa9] +
            _ascii(")\n")
    )
    defer { Path(path).remove() }
    let ran = run([argv()[0], "run", "--no-cache", path])
    assert_eq(ran.exit_code, 0)
    assert_eq(ran.stdout, "😀 1\n")
}
