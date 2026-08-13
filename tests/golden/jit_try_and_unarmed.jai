#: A hot loop inside a `try`, an opcode the tier has no arm for, and a
#: parameter it never modelled.
#:
#: Three things meet here and none of them existed before 2026-08-13.
#:
#: 1. An opcode with no arm in `src/vm/jit/jit_func.c` used to `return false`,
#:    which declines the WHOLE enclosing function. Every catch block holds
#:    OP_GET_EXC, so NO function containing a `try` could ever be compiled --
#:    `tests/bench/error_paths`' eight-million-iteration counted loop ran
#:    interpreted end to end, 124ms against 12ms compiled, because of a handler
#:    its own opcode histogram says executes zero times. An unarmed opcode now
#:    deoptimises at its own offset, so the cold path is interpreted and the
#:    loop around it compiles.
#:
#: 2. That made a raise from COMPILED code inside a `try` reachable for the
#:    first time, and it was not sound: the function tier has no interpreter
#:    frame at all and the OSR tier's frame still points at the loop head, so
#:    `vmThrow` looks the handler up against the wrong offset and the function's
#:    own `catch` is skipped. An overflow inside a protected region now resumes
#:    at its instruction and lets the interpreter raise it -- which means the
#:    result must NOT have reached its home register when the guard fires, and
#:    no operand may still be a borrow of a local's register.
#:
#: 3. It also made a LATENT defect reachable on nearly every deopt. A parameter
#:    the tier sampled as `null` is SLOT_OPAQUE -- "the body never reads this"
#:    -- so jitArgIn passes a raw 0 for it and checks nothing. A deopt hands the
#:    frame to the INTERPRETER, which does read it, and the record wrote its
#:    null straight over the caller's real argument. That turned
#:    `ctx.enter_foreign(module)` into `enter_foreign(null)` inside the
#:    self-hosted compiler, and every `-O0` golden then failed with
#:    `unknown type Box`.
#:
#: A GOLDEN rather than a `tests/lang` unit test, and that is not a style
#: choice: under `jaithon test` none of these bodies gets hot enough to compile,
#: so the same assertions passed with all three fixes reverted. `run_tests.sh`
#: runs a golden at four optimisation levels and again under `--gc-stress` and
#: `JAITHON_JIT_DEOPT_STRESS=1`, through `jaithon run`, which is where the tier
#: actually engages. Every number below was taken from the interpreter.
#:
#: Each `*_step` is small and called past JAI_JIT_THRESHOLD (64) so that the
#: FUNCTION tier compiles it -- deterministic, unlike the sampler-driven OSR
#: tier, which is what made an earlier version of this file toothless.
import std.io

#: Each returns the value the HANDLER saw, and each observes the local the
#: overflowing instruction writes rather than an untouched parameter: a body
#: that wrote its result home before testing V hands the handler the WRAPPED
#: value, and one that raises without resuming at the instruction never reaches
#: the handler at all.

#: OP_ADD_BIND -- `t += ...`, whose destination is the local itself.
fn add_bind_step(total: int, i: int, boom: bool) -> int {
    var t = total
    try {
        t += i % 3
        if boom { t += 9223372036854775807 }
        return t
    } catch e: OverflowError {
        return t
    }
}

#: OP_SUB_BIND.
fn sub_bind_step(total: int, i: int, boom: bool) -> int {
    var t = total
    try {
        t -= i % 3
        if boom { t -= 9223372036854775807 }
        return t
    } catch e: OverflowError {
        return t
    }
}

#: OP_MUL_BIND.
fn mul_bind_step(total: int, i: int, boom: bool) -> int {
    var t = i % 3 + 1
    try {
        t *= 2
        if boom { t *= 9223372036854775807 }
        return t
    } catch e: OverflowError {
        return t
    }
}

#: OP_INC_LOCAL, the fused `j += 1`, whose home used to be written BEFORE the
#: overflow test rather than after it.
fn inc_local_step(total: int, i: int, boom: bool) -> int {
    var j = if boom { 9223372036854775807 } else { i % 3 }
    try {
        j += 1
        return j
    } catch e: OverflowError {
        return j - 9223372036854775807
    }
}

#: Plain OP_ADD, both operands on the operand stack. Its destination register IS
#: the first operand's, so an unprotected write there does not merely leave a
#: wrong local -- it hands the RESUMING INTERPRETER a wrong operand, and the
#: re-executed add may then not overflow at all. `(i % 5 + 1)` also arrives as a
#: BORROW of a local's register, which is what the instruction-start record has
#: to be told about.
fn add_expr_step(total: int, i: int, boom: bool) -> int {
    try {
        let big = if boom { 9223372036854775807 } else { 2 }
        return i % 5 + 1 + big
    } catch e: OverflowError {
        return -total
    }
}

#: Plain OP_MUL, same argument.
fn mul_expr_step(total: int, i: int, boom: bool) -> int {
    try {
        let big = if boom { 4611686018427387904 } else { 2 }
        return (i % 5 + 1) * big
    } catch e: OverflowError {
        return -total
    }
}

fn drive_add_bind(n: int) -> int {
    var acc = 1
    var i = 0
    while i < n {
        acc = add_bind_step(acc, i, i == n - 1)
        i += 1
    }
    return acc
}

fn drive_sub_bind(n: int) -> int {
    var acc = 1
    var i = 0
    while i < n {
        acc = sub_bind_step(acc, i, i == n - 1)
        i += 1
    }
    return acc
}

fn drive_mul_bind(n: int) -> int {
    var acc = 1
    var i = 0
    while i < n {
        acc = mul_bind_step(acc, i, i == n - 1)
        i += 1
    }
    return acc
}

fn drive_inc_local(n: int) -> int {
    var acc = 1
    var i = 0
    while i < n {
        acc = inc_local_step(acc, i, i == n - 1)
        i += 1
    }
    return acc
}

fn drive_add_expr(n: int) -> int {
    var acc = 1
    var i = 0
    while i < n {
        acc = add_expr_step(acc, i, i == n - 1)
        i += 1
    }
    return acc
}

fn drive_mul_expr(n: int) -> int {
    var acc = 1
    var i = 0
    while i < n {
        acc = mul_expr_step(acc, i, i == n - 1)
        i += 1
    }
    return acc
}

#: An exception the handler does NOT match must still get out, rather than the
#: new resume point swallowing it.
fn wrong_handler_step(i: int, boom: bool) -> int {
    var t = i % 3
    try {
        if boom { t += 9223372036854775807 }
        return t
    } catch e: ValueError {
        return -1
    }
}

fn drive_wrong_handler(n: int) -> str {
    var i = 0
    try {
        while i < n {
            wrong_handler_step(i, i == n - 1)
            i += 1
        }
    } catch e: OverflowError {
        return "escaped"
    }
    return "swallowed"
}

#: `tests/bench/error_paths`' shape: a handler that never runs around a counted
#: loop. Nothing here can throw, so the whole answer is that the loop still
#: computes what it computed when it was interpreted.
fn calm(n: int) -> int {
    var total = 0
    var i = 0
    while i < n {
        try {
            total += i % 13
        } catch e: ValueError {
            total -= 1
        }
        i += 1
    }
    return total
}

#: An unarmed opcode (`{}` is OP_BUILD_DICT) in a branch taken once, with real
#: work after it. The walk cannot model the fall-through past an unconditional
#: deopt, so it skips to the next offset something can branch to -- if that skip
#: lands wrong, either the compile declines or this answer changes.
fn cold_unarmed(n: int) -> int {
    var total = 0
    var i = 0
    while i < n {
        if i == n - 1 {
            var d = {}
            d["k"] = i
            total += d["k"]
        }
        total += i % 7
        i += 1
    }
    return total
}

#: An unarmed opcode on the path every iteration takes: everything after it
#: runs interpreted, every iteration, and must still agree. `not` (OP_NOT)
#: rather than another `{}` on purpose -- this loop runs 300,000 times and the
#: golden is also run under `--gc-stress`, which collects on every allocation.
fn hot_unarmed(n: int) -> int {
    var total = 0
    var i = 0
    while i < n {
        total += i % 7
        let odd = not i % 2 == 0
        if odd { total += 1 }
        i += 1
    }
    return total
}

#: A `throw` from inside the protected region caught by the same function.
#: OP_THROW is itself unarmed, so this is an unconditional deopt on a path that
#: really runs, followed by an unwind the interpreter owns.
fn throw_and_catch(n: int) -> int {
    var mark = 0
    var i = 0
    while i < n {
        try {
            mark += i % 3
            if i % 1024 == 7 { throw ValueError("x") }
        } catch e: ValueError {
            mark = mark + 1
        }
        i += 1
    }
    return mark
}

#: A call inside the protected region. The tier refuses this one -- a callee
#: that raises has already had its effects, so the site cannot resume at the
#: call -- which is exactly what the whole function did before. It is here so
#: that a later attempt to admit it has to answer for the answer.
fn checked(x: int) -> int {
    if x % 128 == 3 { throw ValueError("bad") }
    return x + 1
}

fn try_with_call(n: int) -> int {
    var ok = 0
    var bad = 0
    var i = 0
    while i < n {
        try {
            ok += checked(i)
        } catch e: ValueError {
            bad += 1
        }
        i += 1
    }
    return ok - bad
}

class Marker {
    pub var n: int

    fn init(self, n: int) -> void { self.n = n }
}

#: `origin` is null for every call that decides the compile, so it is
#: SLOT_OPAQUE. The tuple is an opcode with no arm, so the body deoptimises on
#: the entry path of EVERY call and the interpreter finishes each one --
#: reading `origin`. Without the deopt record's skip mask this answers -10.
fn enter(origin: any?, n: int) -> int {
    let saved = (n, n)
    if origin is null { return -1 }
    return n
}

fn drive_opaque() -> int {
    var warm = 0
    while warm < 100 {
        enter(null, 1)
        warm += 1
    }
    let m = Marker(7)
    var acc = 0
    var i = 0
    while i < 10 {
        acc += enter(m, 3)
        i += 1
    }
    return acc
}

fn main() -> int {
    #: First, deliberately: `enter`'s compile is decided by its 65th call, and
    #: running the rest of this file first moved which call that was.
    print(drive_opaque())
    print(drive_add_bind(200))
    print(drive_sub_bind(200))
    print(drive_mul_bind(200))
    print(drive_inc_local(200))
    print(drive_add_expr(200))
    print(drive_mul_expr(200))
    print(drive_wrong_handler(200))
    print(calm(300000))
    print(cold_unarmed(300000))
    print(hot_unarmed(300000))
    print(throw_and_catch(300000))
    print(try_with_call(300000))
    return 0
}
