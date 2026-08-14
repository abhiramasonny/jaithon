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

const INT_MAX = 9223372036854775807
const INT_MIN = (-9223372036854775808)
var big: list[int] = []
for i in INT_MAX - 2..=INT_MAX { big.push(i - INT_MAX) }
for i in INT_MIN..INT_MIN + 3 { big.push(i - INT_MIN) }
for i in INT_MAX..=INT_MIN { big.push(999) }
print(big)

# --- leaving early ----------------------------------------------------------
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
var made: list[fn() -> int] = []
for i in 0..4 { made.push(fn() -> int { return i * i }) }
var squares: list[int] = []
for f in made { squares.push(f()) }
print(squares)

fn reused() -> str {
    var out: list[str] = []
    for i in 0..3 { out.push(f"i{i}") }
    for s in ["x", "y"] { out.push(s) }
    for i in 0..2 { out.push(f"j{i}") }
    return f"{out}"
}
print(reused())

# --- bounds that are not literals -------------------------------------------
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

fn via_variable(lo: int, hi: int) -> int {
    let r = lo..hi
    var t = 0
    for i in r { t = t + i }
    return t
}
print([via_variable(0, 5), via_variable(5, 0), via_variable(-3, 3)])

fn hot_via_variable(n: int) -> int {
    let outer = 0..n
    var t = 0
    for i in outer {
        let inner = i..n
        for j in inner { t = t + i * j }
    }
    return t
}
print(hot_via_variable(300))

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

var osr = 0
for i in 0..2_000_000 { osr = osr + i % 7 }
print(osr)

fn floaty(n: int) -> float {
    var t = 0.0
    for i in 1..n { t = t + 1.0 / float(i) }
    return t
}
print(f"{floaty(100000):.6f}")

# --- labels, last in the file on purpose ------------------------------------
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
