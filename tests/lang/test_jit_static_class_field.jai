#: `Klass.STATIC_NAME` -- a static field read through a class receiver known
#: only at compile time, not through `self`.
#:
#: The receiver comes from OP_GET_GLOBAL's `globalClass` arm, which pushes a
#: SLOT_CLASS entry (no register, class pointer pinned by the module version
#: check). Before this, OP_GET_FIELD admitted only SLOT_INST and
#: SLOT_MAYBE_INST receivers, so a class receiver hit neither arm and the
#: whole enclosing function declined at a bare "OP_GET_FIELD", no matter how
#: simple the read -- see docs/agents/fix-1.md for the shape this came from
#: (`TokenFlags.AFTER_NEWLINE` in Lexer._push, lib/jaithon/compile/lexer.jai).
#:
#: The static's storage is `klass->statics`, an ordinary mutable table: a
#: static is reassignable at any point after the class is defined (fix-1.md
#: has the two call sites, checker and interpreter, that make `let` a
#: courtesy rather than a promise here). So the compiled form may bake the
#: table ENTRY's address, never the value, and must re-read it behind a
#: runtime tag guard on every access -- exactly OP_GET_GLOBAL's existing plan
#: for a plain module global, ported to a second table.
#:
#: What has to hold, and what each test below checks:
#:
#: - The value read matches the interpreter's, across the field kinds the
#:   arm classifies (int, bool, str).
#: - A static reassigned between two calls of an already-compiled reader is
#:   picked up, not answered from a value baked at compile time.
#: - A static reassigned to a DIFFERENT KIND deoptimises to the interpreter
#:   rather than reading the new bits as the old kind. `VARIABLE` is declared
#:   `any` so this is reachable through ordinary source: a declared `int`
#:   field cannot actually receive a `str` at runtime (the checker rejects a
#:   literal of the wrong kind outright, and codegen's own type guard on the
#:   store rejects one arriving through an `any` parameter), so `AFTER_NEWLINE`
#:   below only ever demonstrates the address-not-value half.
#: - A body reading statics off two different classes still answers right
#:   (the plan is one guarded table per function body, so the second class's
#:   read is expected to decline and interpret -- this only checks the
#:   ANSWER, not that both compile).
#:
#: The expectations below are the interpreter's, so `JAITHON_NO_JIT=1` on
#: this file proves them and the plain run proves the compiled code agrees.
#: `JAITHON_JIT_DEOPT_STRESS=1` is what actually exercises the tag guard on
#: every iteration rather than only when a real mismatch happens to land.
#:
#: Names are deliberately distinctive (`jflag_...`): `jaithon test` also
#: loads the self-hosted compiler, whose own front end happens to define a
#: `_push` -- a generic name here could be credited to that instead of the
#: function this file means to exercise (scripts/jit_compile_check.py).
# jit-compiles: jflag_sum_static_int_reads, jflag_bool_and_str_reads, jflag_read_variable
from std.test import assert_eq

class TokenFlags2 {
    pub static let AFTER_NEWLINE: int = 1
    pub static let HAS_ESCAPE: int = 2
    pub static let RAW_STRING: int = 4
    pub static let IS_KEYWORD: bool = true
    pub static let LABEL: str = "flag"
    #: Untyped on purpose: only an `any`-declared static can hold values of
    #: different kinds across two reassignments through ordinary source.
    pub static let VARIABLE: any = 1
}

#: A second class, for the two-classes-in-one-body test below.
class OtherFlags2 {
    pub static let ONE: int = 100
}

fn jflag_sum_static_int_reads(reps: int) -> int {
    var total = 0
    var i = 0
    while i < reps {
        total = total + TokenFlags2.AFTER_NEWLINE
        total = total + TokenFlags2.HAS_ESCAPE
        total = total + TokenFlags2.RAW_STRING
        i += 1
    }
    return total
}

fn jflag_bool_and_str_reads(reps: int) -> int {
    var trues = 0
    var chars = 0
    var i = 0
    while i < reps {
        if TokenFlags2.IS_KEYWORD { trues += 1 }
        chars = chars + TokenFlags2.LABEL.len()
        i += 1
    }
    return trues * 1000 + chars
}

#: Reads a static off each of two different classes. The staticsTable guard
#: is one table per function body, so this is expected to decline and run
#: interpreted rather than compile -- deliberately absent from the
#: `jit-compiles` marker above.
fn jflag_sum_two_classes(reps: int) -> int {
    var total = 0
    var i = 0
    while i < reps {
        total = total + TokenFlags2.AFTER_NEWLINE
        total = total + OtherFlags2.ONE
        i += 1
    }
    return total
}

fn jflag_read_variable() -> any {
    return TokenFlags2.VARIABLE
}

fn test_static_int_fields_match_the_interpreter() -> void {
    var grand = 0
    var i = 0
    while i < 100 {
        grand = grand + jflag_sum_static_int_reads(10)
        i += 1
    }
    assert_eq(grand, 100 * 10 * (1 + 2 + 4))
}

fn test_static_bool_and_str_fields_match_the_interpreter() -> void {
    var grand = 0
    var i = 0
    while i < 100 {
        grand = grand + jflag_bool_and_str_reads(10)
        i += 1
    }
    assert_eq(grand, 100 * (10 * 1000 + 10 * 4))
}

fn test_two_classes_statics_in_one_body_still_answers_right() -> void {
    var grand = 0
    var i = 0
    while i < 100 {
        grand = grand + jflag_sum_two_classes(10)
        i += 1
    }
    assert_eq(grand, 100 * 10 * (1 + 100))
}

#: The JaiEntry's ADDRESS is baked, never the value, so a call after a
#: reassignment must see the new number rather than one cached from an
#: earlier compile.
fn test_reassigned_static_is_re_read_not_stale() -> void {
    var i = 0
    var last: any = null
    while i < 100 {
        last = jflag_read_variable()
        i += 1
    }
    assert_eq(last, 1)
    TokenFlags2.VARIABLE = 10
    var j = 0
    while j < 5 {
        last = jflag_read_variable()
        j += 1
    }
    assert_eq(last, 10)
    TokenFlags2.VARIABLE = 1
}

#: `jflag_read_variable` is called enough above and below to be well past
#: JAI_JIT_THRESHOLD by the time this reassignment lands: the tag guard
#: behind the load must deoptimise to the interpreter here rather than read
#: the new bit pattern (now VAL_OBJ, a string) as the VAL_INT it was compiled
#: against.
fn test_reassigned_static_of_a_different_kind_deopts() -> void {
    var i = 0
    var last: any = null
    while i < 100 {
        last = jflag_read_variable()
        i += 1
    }
    assert_eq(last, 1)
    TokenFlags2.VARIABLE = "relabeled"
    var j = 0
    while j < 5 {
        last = jflag_read_variable()
        j += 1
    }
    assert_eq(last, "relabeled")
    TokenFlags2.VARIABLE = 1
}
