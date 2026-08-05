#: Shapes that made an optimisation pass disagree with -O0 (spec/BYTECODE.md §8).
#: Every one of these must print the same thing at every -O level; the test
#: driver runs this file at all four.

# --- fused superinstructions on a local whose type is not statically int -----
#
# The peephole reads bytecode, so it fuses GET_LOCAL/INT/ADD over an `any` local
# exactly as over an `int` one. The fused opcode is therefore a fast path and
# never a restriction: anything but two ints has to reach the same handler the
# unfused sequence would have.
fn add_one(x) {
    var y = x
    y = y + 1  # -> OP_INC_LOCAL
    return y
}

# `y - 1` must NOT become OP_INC_LOCAL with a negated immediate. The two are
# the same instruction only for numbers: everything else raises either way, but
# the message has to name the operator the source wrote.
fn minus_one(x) {
    var y = x
    y = y - 1  # stays OP_SUB; see fuseAt
    return y
}

fn plus_one(x) {
    var y = x
    return y + 1  # -> OP_ADD_INT_CONST
}

fn under_three(x) {
    var y = x
    return y < 3  # -> OP_CMP_LOCAL_CONST_LT
}

fn pair_add(x, z) {
    var a = x
    var b = z
    return a + b  # -> OP_ADD_LOCALS
}

print(add_one(41))
print(minus_one(41))
print(plus_one(41))
print(under_three(1))
print(under_three(9))
print(pair_add(20, 22))

# A float takes the generic path: comparison and float+float succeed, and an
# int mixed with a float widens there exactly as it does in the unfused opcode
# (spec/LANGUAGE.md §2.5), so the fused form must not answer differently.
print(under_three(1.5))
print(pair_add(1.5, 2.5))
print(add_one(1.5))
print(minus_one(1.5))
print(plus_one(1.5))
try { print(minus_one("s")) } catch e { print(e.message) }

# strings and lists concatenate; the message on a real mismatch is the generic one
print(pair_add("a", "b"))
print(pair_add([1], [2]))
try { print(add_one("s")) } catch e { print(e.message) }
try { print(under_three([1])) } catch e { print(e.message) }

# --- a local compared against a constant and branched on ---------------------
#
# OP_JUMP_IF_CMP_LOCAL_K reads the local out of the frame and the constant out
# of the pool, so neither ever reaches the stack. Everything but two ints has to
# fall back to the comparison the unfused sequence would have run, with the same
# message, and the guard against a literal too wide for an i16 immediate has to
# fuse as readily as `n < 2` does.
fn guard_big(x) -> int {
    var n = 0
    while n < 5_000_000 { n = n + 4_999_999 }
    if x < 5_000_000 { return 1 }
    return 2
}

fn guard_small(x) -> int {
    if x < 2 { return 1 }
    return 2
}

fn guard_ne(x) -> int {
    if x != "stop" { return 1 }
    return 2
}

print(guard_big(1))
print(guard_big(9_000_000))
print(guard_small(1))
print(guard_small(9))
print(guard_small(1.5))
print(guard_ne("go"))
print(guard_ne("stop"))
try { print(guard_small([1])) } catch e { print(e.message) }
try { print(guard_big("s")) } catch e { print(e.message) }

# --- `not` folded into the jump it feeds -------------------------------------
#
# Erasing the OP_NOT erases the check it would have run, and the jump's own
# check words the same failure differently. Only a producer that is a bool by
# construction may be folded through.
fn negated(x) -> int {
    if not x { return 1 }
    return 2
}

fn compared(a: int, b: int) -> int {
    if not a == b { return 1 }  # the comparison is bool: still folded
    return 2
}

print(negated(true))
print(negated(false))
print(compared(1, 2))
print(compared(2, 2))
try { print(negated(0)) } catch e { print(e.message) }

# --- unreachable try/finally ------------------------------------------------
#
# A handler is reachable only when its protected region is. Treating the
# exception table as an unconditional entry point resurrected the handler of a
# `try` inside a block the front end had proved dead, and the backward branch at
# the end of the enclosing loop body dragged the loop's OP_FOR_ITER back with
# it — without the setup code that produces the iterator.
fn dead_try() -> int {
    if false {
        for i in 2..6 {
            try { print(i) } finally { }
        }
    }
    return 1
}
print(dead_try())

# --- a `finally` inside a loop, reached only through a handler ---------------
#
# The `try` below throws unconditionally, so once the peephole has folded that
# away the only path into the loop is the catch handler. The finally region's
# own handler then sits under the loop's iterator, one deep, and not at the
# frame base the way an exception-table entry assumes.
fn only_via_handler() -> int {
    var total = 0
    try {
        throw ValueError("first")
    } catch _e: ValueError {
        for i in 1..4 {
            try {
                total = total + i
            } finally {
                total = total + 10
            }
        }
    }
    return total
}
print(only_via_handler())

# --- loop-invariant global hoisting -----------------------------------------
#
# The pass moves the read into a pre-header, so a loop that runs zero times must
# still behave as if it never read the global, and a global reassigned between
# calls must be re-read.
var SCALE = 3

fn scaled(n: int) -> int {
    var s = 0
    var i = 0
    while i < n {
        s = s + SCALE * SCALE - SCALE
        i = i + 1
    }
    return s
}
print(scaled(0))
print(scaled(4))
SCALE = 10
print(scaled(4))

# --- slot coalescing against captures ---------------------------------------
#
# Disjoint scopes make slots reusable; a slot an upvalue points at is not, even
# when this function never names it itself.
fn coalesce_and_capture() -> fn() -> int {
    {
        var a = 1
        var b = 2
        print(a + b)
    }
    {
        var c = 3
        var d = 4
        print(c + d)
    }
    var captured = 42
    return fn() -> int { return captured }
}
print(coalesce_and_capture()())

fn per_iteration() -> list[fn() -> int] {
    var out: list[fn() -> int] = []
    for i in 0..3 {
        let snapshot = i * 10
        out.push(fn() -> int { return snapshot })
    }
    return out
}
for f in per_iteration() { print(f()) }

# --- slot coalescing across a dead local -------------------------------------
#
# A local that is never read has a live range one instruction long, so several
# of them compete for the same number and one number takes tenant after tenant.
# The pass has to remember how far the *furthest* tenant of a number reaches,
# not who the current one is: reading one tenant's range hands the number out
# again while an earlier tenant is still live, and the live local silently
# takes the dead one's value. The three below print 2, 20 and 7 at -O0 and did
# not from -O1 up.
fn dead_between() -> int {
    let _a = 1
    let b = 2
    let _c = 3
    return b
}

# The same, with the live local's range spanning several dead ones so that the
# number it was given is the first one every later candidate tries.
fn dead_around() -> int {
    let _p = 1
    let live = 20
    let _q = 3
    let _r = 4
    let _s = 5
    return live
}

# Two live locals whose ranges are disjoint may share a number; a third that
# outlives both may not have it. `first` dies before `second` starts, so they
# coalesce, and `keep` — read last — has to survive both.
fn disjoint_then_long() -> int {
    let keep = 7
    let first = 100
    print(first)
    let second = 200
    print(second)
    return keep
}

print(dead_between())
print(dead_around())
print(disjoint_then_long())

print("done")
