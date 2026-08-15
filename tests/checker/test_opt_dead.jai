#: `jaithon.compile.opt.dead` — pass 2 of spec §8, the Jaithon half of
#: `jaiOptDeadCode`. These tests assert on emitted bytecode rather than on what
#: a program prints, because the failure this pass has is not a wrong answer:
#: it is different-but-working code, which nothing but `make bootstrap` would
#: notice.
#:
#: Most cases are checked at every `--O` level, which is both the property §8
#: states about this pass and the only level range at which the pass is on its
#: own. Above `--O0` the peephole calls `mark_unreachable` as well, so a test
#: that only looked at `--O2` would keep passing with this file stubbed out.
#: Every expectation below was read back from `jaithon disasm -O0` and from
#: the retired differential oracle, so an assertion that stops holding is a divergence
#: from the C front end rather than a change of taste.
#:
#: Three things are asserted, because the pass has three ways to be wrong: the
#: opcode sequence says what survived, `decode` from the pass substrate says
#: the rewrite is well formed (it answers null unless every branch operand,
#: exception-table offset and default-thunk offset still lands on an
#: instruction boundary), and the exception table and `default_offsets` are
#: read directly, since those are the two tables a relayout corrupts silently.
import jaithon.compile.opt.chunk as chunk

from std.test import assert_eq, assert_len, assert_true
from jaithon.compile import compile_source
from jaithon.compile.emit import ConstTag, FuncProto, Op, op_name, opcode
from jaithon.compile.jaic import disassemble_instruction

fn function_in(source: str, name: str = "f", level: int = 2) -> FuncProto {
    let built = compile_source(source, "<test>", false, 0, level)
    assert_true(built.ok(), built.report())
    let body = built.body
    assert_true(body != null, "nothing was emitted")
    let root = body ?? FuncProto("<none>", 0, 0)
    if name == "" { return root }
    let found = _named(root, name)
    assert_true(found != null, f"no function named `{name}` was emitted")
    return found ?? root
}

fn _named(proto: FuncProto, name: str) -> FuncProto? {
    for value in proto.chunk.constants {
        if value.tag != ConstTag.Func { continue }
        if value.value.name == name { return value.value }
        let nested = _named(value.value, name)
        if nested != null { return nested }
    }
    return null
}

fn ops_of(proto: FuncProto) -> list[str] {
    var names: list[str] = []
    var offset = 0
    while offset < proto.chunk.code.len() {
        let step = disassemble_instruction(proto.chunk, offset)
        if step[1] <= offset { break }
        names.push(op_name(proto.chunk.code[offset]))
        offset = step[1]
    }
    return names
}

fn assert_ops_at_every_level(source: str, expected: list[str], name: str = "f") -> void {
    for level in 0..4 {
        let proto = function_in(source, name, level)
        assert_eq(ops_of(proto), expected, f"at -O{level}")
        assert_true(chunk.decode(proto) != null, f"the chunk does not decode at -O{level}")
    }
}

fn test_the_implicit_epilogue_after_a_return_is_deleted() -> void {
    assert_ops_at_every_level("fn f() -> int {\n    return 1\n}\n", ["OP_INT", "OP_RETURN"])
    let f = function_in("fn f() -> int {\n    return 1\n}\n")
    assert_eq(f.chunk.code, [opcode(Op.Int), 1, 0, opcode(Op.Return)])
}

#: The peephole's `POP; RETURN_NULL` fusion only runs at -O1 and above, so
#: this is the one case in this file where -O0's op list is not the other
#: levels' -- everywhere else the pass under test has nothing to fuse.
fn test_a_body_that_can_fall_off_the_end_keeps_its_epilogue() -> void {
    let source = "fn f() -> void {\n    print(1)\n}\n"
    assert_eq(
        ops_of(function_in(source, "f", 0)),
        ["OP_GET_GLOBAL", "OP_INT", "OP_CALL", "OP_POP", "OP_RETURN_NULL"],
        "at -O0"
    )
    for level in 1..4 {
        assert_eq(
            ops_of(function_in(source, "f", level)),
            ["OP_GET_GLOBAL", "OP_INT", "OP_CALL", "OP_POP_RETURN_NULL"],
            f"at -O{level}"
        )
    }
}

fn test_an_empty_body_is_a_bare_epilogue() -> void {
    assert_ops_at_every_level("fn f() -> void {\n}\n", ["OP_RETURN_NULL"])
    assert_ops_at_every_level("fn f() -> void {\n    return\n}\n", ["OP_RETURN_NULL"])
}

fn test_statements_after_a_return_are_deleted() -> void {
    assert_ops_at_every_level(
        "fn f() -> int {\n    return 1\n    let x = 2\n    print(x)\n}\n",
        ["OP_INT", "OP_RETURN"]
    )
}

fn test_the_epilogue_after_a_throw_is_deleted() -> void {
    assert_ops_at_every_level(
        "fn f() {\n    throw Error(\"x\")\n}\n",
        ["OP_GET_GLOBAL", "OP_CONST", "OP_CALL", "OP_THROW"]
    )
}

fn test_a_nested_function_is_optimised_too() -> void {
    let source = "fn outer() -> int {\n" +
        "    fn inner() -> int {\n" +
        "        return 1\n" +
        "    }\n" +
        "    return inner()\n" +
        "}\n"
    assert_ops_at_every_level(source, ["OP_INT", "OP_RETURN"], "inner")
}

fn test_a_dead_try_takes_its_handler_and_its_table_entry_with_it() -> void {
    let source = "fn f() -> int {\n" +
        "    return 1\n" +
        "    try {\n" +
        "        print(1)\n" +
        "    } catch e: Error {\n" +
        "        print(2)\n" +
        "    }\n" +
        "}\n"
    assert_ops_at_every_level(source, ["OP_INT", "OP_RETURN"])
    assert_len(function_in(source).chunk.exceptions, 0)
}

fn test_a_dead_finally_goes_the_same_way() -> void {
    let source = "fn f() -> int {\n" +
        "    return 1\n" +
        "    try {\n" +
        "        print(1)\n" +
        "    } finally {\n" +
        "        print(2)\n" +
        "    }\n" +
        "}\n"
    assert_ops_at_every_level(source, ["OP_INT", "OP_RETURN"])
    assert_len(function_in(source).chunk.exceptions, 0)
}

fn test_a_live_try_keeps_its_handler() -> void {
    let source = "fn f() -> int {\n" +
        "    try {\n" +
        "        return 1\n" +
        "    } catch e: Error {\n" +
        "        return 2\n" +
        "    }\n" +
        "}\n"
    assert_ops_at_every_level(
        source,
        ["OP_INT", "OP_RETURN", "OP_GET_EXC", "OP_BIND", "OP_INT", "OP_RETURN"]
    )
    let f = function_in(source)
    assert_len(f.chunk.exceptions, 1)
    assert_true(chunk.decode(f) != null, "the exception table no longer names boundaries")
}

fn test_a_dead_try_nested_in_a_live_one_leaves_the_live_one_alone() -> void {
    let source = "fn f() -> int {\n" +
        "    try {\n" +
        "        return 1\n" +
        "        try {\n" +
        "            print(1)\n" +
        "        } catch e: Error {\n" +
        "            print(2)\n" +
        "        }\n" +
        "    } catch e: Error {\n" +
        "        return 3\n" +
        "    }\n" +
        "}\n"
    assert_ops_at_every_level(
        source,
        ["OP_INT", "OP_RETURN", "OP_GET_EXC", "OP_BIND", "OP_INT", "OP_RETURN"]
    )
    assert_len(function_in(source).chunk.exceptions, 1)
}

fn test_a_dead_try_does_not_resurrect_its_enclosing_loop() -> void {
    let source = "fn f(n: int) -> int {\n" +
        "    while n > 0 {\n" +
        "        return n\n" +
        "        try {\n" +
        "            print(1)\n" +
        "        } catch e: Error {\n" +
        "            print(2)\n" +
        "        }\n" +
        "    }\n" +
        "    return 0\n" +
        "}\n"
    let f = function_in(source, "f", 0)
    assert_eq(
        ops_of(f),
        [
            "OP_GET_LOCAL",
            "OP_INT",
            "OP_GT",
            "OP_JUMP_IF_FALSE",
            "OP_GET_LOCAL",
            "OP_RETURN",
            "OP_INT",
            "OP_RETURN",
        ]
    )
    assert_len(f.chunk.exceptions, 0)
    assert_true(chunk.decode(f) != null, "the loop exit no longer lands on an instruction")
}

fn test_a_loop_whose_exit_edge_is_live_keeps_everything() -> void {
    let source = "fn f() -> int {\n    while true {\n        print(1)\n    }\n    return 0\n}\n"
    let f = function_in(source, "f", 0)
    assert_eq(
        ops_of(f),
        [
            "OP_TRUE",
            "OP_JUMP_IF_FALSE",
            "OP_GET_GLOBAL",
            "OP_INT",
            "OP_CALL",
            "OP_POP",
            "OP_LOOP",
            "OP_INT",
            "OP_RETURN",
        ]
    )
}

fn test_a_default_thunk_is_an_entry_point() -> void {
    let source = "fn f(a: int = 3) -> int {\n    return a\n}\n"
    assert_ops_at_every_level(
        source,
        ["OP_JUMP", "OP_INT", "OP_RETURN", "OP_GET_LOCAL", "OP_RETURN"]
    )
    let f = function_in(source)
    assert_len(f.default_offsets, 1)
    assert_eq(f.default_offsets[0], 3)
}

fn test_branches_still_resolve_after_the_relayout() -> void {
    let source = "fn f(a) {\n    if a {\n        return 1\n    }\n    return 2\n}\n"
    assert_ops_at_every_level(
        source,
        ["OP_GET_LOCAL", "OP_JUMP_IF_FALSE", "OP_INT", "OP_RETURN", "OP_INT", "OP_RETURN"]
    )
    let f = function_in(source)
    let decoded = chunk.decode(f)
    assert_true(decoded != null, "the rewritten chunk does not decode")
    let code = decoded ?? chunk.Code(f)
    assert_eq(code.data[1].target, 4)
}
