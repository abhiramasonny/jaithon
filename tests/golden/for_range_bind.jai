#: `for x in a..b` lowered to OP_ITER_RANGE/OP_FOR_RANGE_BIND (spec §3.4).
#:
#: The fused pair builds neither the ObjRange nor the ObjIter the general
#: lowering does, and — this is what makes it different from every other fused
#: form in the tree — it puts NOTHING on the operand stack where the iterator
#: used to sit. So the interesting failures are not arithmetic, they are stack
#: depth: a `break` that pops one value too many takes an enclosing temporary
#: with it, and a `continue` that pops one too few leaves the frame growing by
#: one per iteration. Neither shows up in a loop that simply runs to the end,
#: which is why most of this file leaves early instead.
#:
#: Everything here must print the same at every -O level (the coalescer
#: renumbers the two hidden slots like any other local), under --gc-stress, and
#: under JAITHON_JIT_DEOPT_STRESS=1.

# --- the bounds themselves --------------------------------------------------
# An off-by-one in `end` shows here and almost nowhere else. Inclusive and
# exclusive differ by exactly the last element, and the empty forms have to
# produce no iterations rather than one or 2^64.
fn total(lo: int, hi: int) -> int {
    var t = 0
    for i in lo..hi { t = t + i }
    return t
}

fn total_inclusive(lo: int, hi: int) -> int {
    var t = 0
    for i in lo..=hi { t = t + i }
    return t
}

fn count(lo: int, hi: int) -> int {
    var n = 0
    for _ in lo..hi { n = n + 1 }
    return n
}

fn count_inclusive(lo: int, hi: int) -> int {
    var n = 0
    for _ in lo..=hi { n = n + 1 }
    return n
}

print([total(0, 5), total(0, 0), total(5, 0), total(-3, 3), total(-3, -1)])
print(
    [
        total_inclusive(0, 5),
        total_inclusive(0, 0),
        total_inclusive(5, 0),
        total_inclusive(-3, 3),
        total_inclusive(3, 3),
    ]
)
print([count(0, 5), count(0, 0), count(5, 0), count(-3, 3)])
print([count_inclusive(0, 5), count_inclusive(0, 0), count_inclusive(5, 0), count_inclusive(-3, 3)])

# The whole 64-bit range, at both ends. `end` is computed wrapping, so an
# inclusive range ending at the largest int has to stop rather than wrap round
# and start again; one that begins at the smallest has to start there.
const INT_MAX = 9223372036854775807
const INT_MIN = (-9223372036854775808)
var big: list[int] = []
for i in INT_MAX - 2..=INT_MAX { big.push(i - INT_MAX) }
for i in INT_MIN..INT_MIN + 3 { big.push(i - INT_MIN) }
for i in INT_MAX..=INT_MIN { big.push(999) }  # empty: hi < lo
print(big)

# --- leaving early ----------------------------------------------------------
# `break` and `continue` unwind the operand stack to a depth the emitter alone
# knows, and the fused loop's depth is one shallower than the general one's. A
# marker value sits under each loop so that a pop of the wrong size corrupts
# something printable rather than going unnoticed.
fn early(stop_at: int) -> str {
    let marker = "under"
    var seen: list[int] = []
    for i in 0..10 {
        if i == 3 { continue }
        if i == stop_at { break }
        seen.push(i)
    }
    return f"{marker}:{seen}"
}
print(early(7))
print(early(0))
print(early(100))

# Two levels, both fused, leaving from the inner one.
fn nested_break() -> str {
    let marker = -1
    var acc: list[int] = []
    for i in 0..4 {
        for j in 0..4 {
            if j == 2 { break }
            if i == 3 { continue }
            acc.push(i * 10 + j)
        }
    }
    return f"{marker}:{acc}"
}
print(nested_break())

# `return` from inside, at both nesting levels, leaving the frame rather than
# the loop.
fn returns_from_inside() -> int {
    for i in 0..10 {
        for j in 0..10 {
            if i * j == 12 { return i * 100 + j }
        }
    }
    return -1
}
print(returns_from_inside())

# --- protected regions ------------------------------------------------------
# `break` out of a `try` inlines the `finally` chain first and then pops, and
# the unwinder restores the stack itself on the exception path: two separate
# routes out of the same loop.
fn protected() -> str {
    var log: list[str] = []
    for i in 0..5 {
        try {
            if i == 2 { break }
            if i == 1 { throw ValueError("one") }
            log.push(f"body{i}")
        } catch e: ValueError {
            log.push(f"caught{i}")
        } finally {
            log.push(f"fin{i}")
        }
    }
    return f"{log}"
}
print(protected())

fn deferred() -> str {
    var log: list[str] = []
    fn inner(log: list[str]) -> void {
        defer { log.push("deferred") }
        for i in 0..4 {
            if i == 2 { return }
            log.push(f"{i}")
        }
    }
    inner(log)
    return f"{log}"
}
print(deferred())

fn escaping() -> str {
    var log: list[str] = []
    try {
        for i in 0..5 {
            log.push(f"{i}")
            if i == 2 { throw ValueError("out") }
        }
        log.push("unreached")
    } catch e: ValueError {
        log.push("handled")
    }
    return f"{log}"
}
print(escaping())

# --- the loop variable ------------------------------------------------------
# One closure per iteration over the loop variable's own slot: OP_CLOSE_UPVALUE
# still has to detach each iteration's binding even though the loop's counter
# now lives beside it in the frame.
var made: list[fn() -> int] = []
for i in 0..4 { made.push(fn() -> int { return i * i }) }
var squares: list[int] = []
for f in made { squares.push(f()) }
print(squares)

# The loop variable's slot reused afterwards by something of another kind,
# which is what the slot coalescer does with the hidden counter as well.
fn reused() -> str {
    var out: list[str] = []
    for i in 0..3 { out.push(f"i{i}") }
    for s in ["x", "y"] { out.push(s) }
    for i in 0..2 { out.push(f"j{i}") }
    return f"{out}"
}
print(reused())

# --- bounds that are not literals -------------------------------------------
# Both ends are ordinary expressions, evaluated once at entry. A bound that
# changes under the loop must not change the number of iterations.
fn moving_bound() -> str {
    var n = 4
    var seen: list[int] = []
    for i in 0..n {
        seen.push(i)
        n = 1
    }
    return f"{n}:{seen}"
}
print(moving_bound())

fn call_bounds(xs: list[int]) -> int {
    var t = 0
    for i in xs.len() - 3..xs.len() { t = t + xs[i] }
    return t
}
print(call_bounds([1, 2, 3, 4, 5]))

# A range reaching the loop through a variable keeps the general lowering, and
# has to keep agreeing with the fused one.
fn via_variable(lo: int, hi: int) -> int {
    let r = lo..hi
    var t = 0
    for i in r { t = t + i }
    return t
}
print([via_variable(0, 5), via_variable(5, 0), via_variable(-3, 3)])

# A non-int bound raises the same TypeError, with the same words, as
# OP_BUILD_RANGE always did.
fn bad_bound(hi) -> str {
    try {
        for i in 0..hi { print(i) }
    } catch e: TypeError {
        return f"{e}"
    }
    return "no error"
}
print(bad_bound(2.5))
print(bad_bound("nine"))

# --- hot enough to compile --------------------------------------------------
# The arms above run interpreted at these sizes. These run long enough for the
# function tier and the OSR tier to take them, so the compiled loop counter,
# the compiled exit branch and the deopt that hands the loop back are all
# exercised — and under JAITHON_JIT_DEOPT_STRESS every guard inside them fails.
fn hot_sum(n: int) -> int {
    var t = 0
    for i in 0..n {
        for j in i..n { t = t + i * j }
    }
    return t
}
print(hot_sum(300))

fn hot_early(n: int) -> int {
    var t = 0
    for i in 0..n {
        for j in 0..n {
            if j > i { break }
            t = t + 1
        }
    }
    return t
}
print(hot_early(400))

# An OSR entry: one loop, entered once, running long enough to be compiled from
# the middle. The counter has to survive the handover in both directions.
var osr = 0
for i in 0..2_000_000 { osr = osr + i % 7 }
print(osr)

# The loop variable taking a value the body then uses as a float, so the
# compiled arm's kind adoption has something to disagree with.
fn floaty(n: int) -> float {
    var t = 0.0
    for i in 1..n { t = t + 1.0 / float(i) }
    return t
}
print(f"{floaty(100000):.6f}")

# --- labels, last in the file on purpose ------------------------------------
# `labelled` leaves two fused loops at once and skips the rest of an outer body;
# `mixed` nests a fused loop inside a general one and then the other way round,
# so the two unwind conventions meet at every break and a labelled break crosses
# from one to the other. They are last because `jaithon fmt` deletes the comment
# run that follows a function containing a labelled loop, and a comment between
# these two would be eaten by it.
fn labelled() -> str {
    let marker = "keep"
    var acc: list[int] = []
    'outer: for i in 0..5 {
        for j in 0..5 {
            if j == 2 { continue 'outer }
            if i == 3 { break 'outer }
            acc.push(i * 10 + j)
        }
        acc.push(900 + i)
    }
    return f"{marker}:{acc}"
}
print(labelled())

fn mixed() -> str {
    let marker = "m"
    var acc: list[str] = []
    'outer: for name in ["a", "b", "c"] {
        for i in 0..4 {
            if i == 2 { continue 'outer }
            if name == "c" { break 'outer }
            acc.push(f"{name}{i}")
        }
    }
    'other: for i in 0..3 {
        for name in ["p", "q", "r"] {
            if name == "q" { continue 'other }
            if i == 2 { break 'other }
            acc.push(f"{i}{name}")
        }
    }
    var k = 0
    while k < 4 {
        for i in 0..4 {
            if i == 1 { break }
            acc.push(f"w{k}{i}")
        }
        k = k + 1
    }
    return f"{marker}:{acc}"
}
print(mixed())
