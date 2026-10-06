#: Deeply nested expressions parse once the parser itself runs compiled.
#:
#: The recursive descent alternates between compiled and interpreted bodies,
#: and each compiled-to-interpreted call re-enters the interpreter loop. A
#: fixed cap of 128 re-entries made 32 nested calls or 45 nested parentheses
#: raise RecursionError out of the front end once enough of the parser was
#: compiled, far below the parser's own nesting limit and the interpreter's
#: frame limit. The warm-up makes the parser hot before the deep parse.
#:
#: A cold parser runs interpreted, so there the bound is the interpreter's
#: frame limit and every frame the descent spends per nesting level counts:
#: moving the call arm into a helper of its own cost one frame per level and
#: made 59-62 nested calls fail in a fresh process where they had parsed.
from std.test import assert_eq, assert_true
from jaithon.compile.diag import DiagnosticBag
from jaithon.compile.parse.wire import parse

fn _warm_source() -> str {
    var lines: list[str] = ["fn f(x: any) -> any { return x }", "let a = [1]"]
    for i in 0..400 {
        lines.push(
            f"let w{i} = f(a[0] + {i} * 2 - (3 + {i}) / 1.5) == 4 and not (1 < {i} or {i} >= 9)"
        )
    }
    return "\n".join(lines)
}

fn _nested(open: str, close: str, depth: int) -> str {
    var text = ""
    for _ in 0..depth { text += open }
    text += "1"
    for _ in 0..depth { text += close }
    return "let data = " + text + "\n"
}

fn test_deep_nesting_parses_once_the_parser_is_hot() -> void {
    let warm = _warm_source()
    for _ in 0..3 { assert_true(parse(warm) != null, "the warm-up parses") }
    let shapes = [["(", ")"], ["[", "]"], ["g(", ")"], ["(1 + ", ")"]]
    for shape in shapes {
        for depth in [24, 32, 45] {
            let bag = DiagnosticBag()
            let program = parse(_nested(shape[0], shape[1], depth), diagnostics: bag)
            assert_true(program != null, f"{shape[0]} nested {depth} deep")
            assert_true(not bag.has_errors(), f"{shape[0]} nested {depth} deep")
        }
    }
}

fn _spawn(argv: list[str]) -> dict[str, any] {
    return __prim__.os_spawn(argv, {})
}

#: In a child, so the parser is cold: nothing in this process has warmed it.
fn test_sixty_nested_calls_parse_in_a_cold_process() -> void {
    let made = _spawn(["/usr/bin/mktemp", "-d"])
    assert_eq(made["exit_code"], 0)
    let dir = f"{made["stdout"]}".strip()
    let path = dir + "/nested_calls.jai"
    let handle = __prim__.io_open(path, "wb")
    __prim__.io_write(handle, "fn g(x: int) -> int { return x }\n" + _nested("g(", ")", 60))
    __prim__.io_close(handle)

    let result = _spawn([__prim__.os_argv()[0], "check", "--no-cache", path])
    __prim__.io_remove(path)
    __prim__.io_remove(dir)
    assert_eq(result["stderr"], "")
    assert_eq(result["exit_code"], 0)
}
