#: `__prim__.f64_sqrt(x)` and its kin -- a NATIVE reached through a MODULE
#: receiver, inside a compiled body.
#:
#: `__prim__` is a synthetic namespace jaiDefineNative builds the first time a
#: dotted name registers (namespaceFor/makeNamespace, builtins.c) -- an
#: ObjModule sitting in vm.builtins, not a real import. lib/std/math.jai's
#: `sqrt`, `floor`, `ceil`, `sin`, ... are each nothing but one call through
#: it, so OP_GET_GLOBAL "__prim__" used to end the walk for every one of them:
#: `globalNative` correctly refuses it (it is a module, not a native) and
#: `globalSlot` cannot find it either (it lives in vm.builtins's globals, not
#: the CURRENT module's), so the site fell all the way to "not a compiled
#: global function" and everything from there on ran interpreted -- the
#: negative-argument check compiled, the call that does the actual work
#: never did.
#:
#: Two things needed an arm, matching `math.sqrt(x)` itself
#: (test_jit_module_call.jai) one level down:
#:
#:   THE GLOBAL. `globalNamespace` resolves `__prim__` BY VALUE off
#:   vm.builtins, same shape as `globalNative`, and OP_GET_GLOBAL pushes it as
#:   a register-resident SLOT_OBJ loaded from a baked CONSTANT rather than
#:   from a JaiEntry -- nothing rebinds the name `__prim__` the way an import
#:   can be rebound, so there is no table to read one back out of.
#:
#:   THE INVOKE. `emitModuleNativeCall` is emitModuleCall's sibling for a
#:   native member instead of a compiled closure: jaiModuleMethod hands the
#:   native back unwrapped and resolveInvokeTarget drops the receiver, so the
#:   same `jitCallOut` descriptor call that already reaches a closure reaches
#:   a native too, with no changes of its own. What differs is what comes
#:   back -- a native carries no jitReturnKind to ask, so kPrimNativeResults
#:   answers it by (name, argc) -- and the guard that matters: not the
#:   receiver's identity (a baked constant cannot hold anything else), but
#:   `__prim__`'s own `version`, because `__prim__.f64_sqrt = v` is real
#:   syntax (jaiSetProperty's IS_MODULE arm) for a name reachable as a bare
#:   identifier, and jaiModuleSet bumps it on exactly that kind of write.
#:
#: The answers here are the interpreter's -- `JAITHON_NO_JIT=1` proves the
#: expectations and the plain run proves the compiled code agrees. Both error
#: messages are asserted verbatim: `sqrt`'s own domain check (ordinary
#: Jaithon, ahead of the primitive) and `f64_sqrt`'s (raised from inside the
#: native the descriptor calls) are two different strings from two different
#: places, and a compiled call must raise exactly what the interpreter raises
#: for each.
#:
#: JAITHON_JIT_MODULE_NATIVE=0 turns off both halves of the arm at once; every
#: assertion below holds either way, because all of them are questions about
#: the answer, not about whether it compiled.
#:
#: scripts/jit_compile_check.py's marker does NOT discriminate the switch for
#: THIS arm specifically, and that is a property of OP_GET_GLOBAL, not a hole
#: in this test: a refusal there falls into emitUnarmedDeopt (function tier
#: only -- `!e->osr` gates it off for OSR), which deoptimises from the
#: refusing instruction ONWARD rather than declining the whole function, so
#: `sqrt_direct` still prints `[jit] compiled sqrt_direct ...` with the switch
#: OFF -- as one giant deopt stub that interprets its entire body every call.
#: The marker still earns its keep as a regression guard (it would catch the
#: body dropping below JAI_JIT_THRESHOLD and failing to reach ANY tier), but
#: the real evidence the switch does something is the interpreted-instruction
#: count, not this line: a 2,000,000-call loop over `sqrt_direct`/
#: `floor_direct` reads 357,706 instructions with the switch on against
#: 20,024,420 off -- both answer 12000000.0.
# jit-compiles: sqrt_direct, floor_direct
from std.test import assert_eq, assert_near, assert_throws
import std.math

#: `__prim__.f64_sqrt` called directly, bypassing math.sqrt's own guard --
#: this is the call emitModuleNativeCall must reach for the body to compile
#: whole, not the Jaithon-level check ahead of it.
fn sqrt_direct(x: float) -> float {
    return __prim__.f64_sqrt(x)
}

fn floor_direct(x: float) -> float {
    return __prim__.f64_floor(x)
}

#: Entry count, not loop count: the function tier compiles at
#: JAI_JIT_THRESHOLD (64) calls, so each of these is called by its own
#: OUTER loop rather than looping inside itself.
fn warm_sqrt_direct() -> void {
    var call = 0
    while call < 200 {
        let _ = sqrt_direct(4.0)
        call += 1
    }
}

fn warm_floor_direct() -> void {
    var call = 0
    while call < 200 {
        let _ = floor_direct(4.7)
        call += 1
    }
}

pub fn test_prim_sqrt_direct_agrees_with_the_interpreter() -> void {
    warm_sqrt_direct()
    assert_eq(sqrt_direct(4.0), 2.0, "perfect square")
    assert_eq(sqrt_direct(0.0), 0.0, "zero")
    assert_near(sqrt_direct(2.0), 1.4142135623730951, message: "irrational")
    var call = 0
    while call < 200 {
        assert_eq(sqrt_direct(9.0), 3.0, "steady state")
        call += 1
    }
}

pub fn test_prim_floor_direct_agrees_with_the_interpreter() -> void {
    warm_floor_direct()
    assert_eq(floor_direct(4.7), 4.0, "fraction down")
    assert_eq(floor_direct(-4.3), -5.0, "negative")
    assert_eq(floor_direct(4.0), 4.0, "already whole")
    var call = 0
    while call < 200 {
        assert_eq(floor_direct(4.7), 4.0, "steady state")
        call += 1
    }
}

#: The error path. Two different messages from two different places:
#: math.jai's own domain check raises before the primitive is ever reached,
#: and f64_sqrt raises its own domain error when called past that check. A
#: compiled sqrt_direct must produce the second, verbatim -- there is no
#: Jaithon-level guard inside sqrt_direct to intercept it first.
pub fn test_math_sqrt_raises_its_own_message() -> void {
    warm_sqrt_direct()
    let err = assert_throws(ValueError, fn() -> any { return math.sqrt(-1.0) })
    assert_eq(
        err.message,
        "sqrt is undefined for a negative argument, got -1.0",
        "math.sqrt's own domain check"
    )
}

pub fn test_prim_f64_sqrt_raises_its_own_message() -> void {
    warm_sqrt_direct()
    let err = assert_throws(ValueError, fn() -> any { return sqrt_direct(-1.0) })
    assert_eq(
        err.message,
        "f64_sqrt() expects a non-negative argument, got -1",
        "f64_sqrt's own domain error"
    )
    #: Called directly too, not only through the wrapper -- the same call
    #: emitModuleNativeCall's descriptor makes, without a Jaithon frame around
    #: it to blur which message came from where.
    let err2 = assert_throws(ValueError, fn() -> any { return __prim__.f64_sqrt(-1.0) })
    assert_eq(
        err2.message,
        "f64_sqrt() expects a non-negative argument, got -1",
        "f64_sqrt's own domain error, called bare"
    )
}

#: `__prim__.f64_sqrt = v` is real syntax (jaiSetProperty's IS_MODULE arm) for
#: a name lib/std/math.jai itself reaches as a bare identifier. Without the
#: m->version guard emitModuleNativeCall emits, the compiled body would keep
#: calling the ObjNative it baked at compile time forever -- this rebinds to
#: a DIFFERENT already-registered native (not a closure, so this exercises
#: emitModuleNativeCall specifically rather than emitModuleCall's sibling
#: arm) and checks the compiled call picks it up.
pub fn test_a_rebound_prim_member_is_picked_up() -> void {
    warm_sqrt_direct()
    assert_eq(sqrt_direct(4.0), 2.0, "baseline")

    let original = __prim__.f64_sqrt
    __prim__.f64_sqrt = __prim__.f64_floor
    for _r in 0..20000 { let _ = sqrt_direct(4.7) }
    assert_eq(sqrt_direct(4.7), 4.0, "picked up the rebind")

    __prim__.f64_sqrt = original
    assert_eq(sqrt_direct(4.0), 2.0, "restored")
}
