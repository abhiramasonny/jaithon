#: Shapes that made an optimisation pass disagree with -O0 (spec/BYTECODE.md §8).
#: Every one of these must print the same thing at every -O level; the test
#: driver runs this file at all four.

# --- fused superinstructions on a local whose type is not statically int -----
fn add_one(x) {
    var y = x
    y = y + 1
    return y
}

fn minus_one(x) {
    var y = x
    y = y - 1
    return y
}

fn plus_one(x) {
    var y = x
    return y + 1
}

fn under_three(x) {
    var y = x
    return y < 3
}

fn pair_add(x, z) {
    var a = x
    var b = z
    return a + b
}

print(add_one(41))
print(minus_one(41))
print(plus_one(41))
print(under_three(1))
print(under_three(9))
print(pair_add(20, 22))

print(under_three(1.5))
print(pair_add(1.5, 2.5))
print(add_one(1.5))
print(minus_one(1.5))
print(plus_one(1.5))
try { print(minus_one("s")) } catch e { print(e.message) }

print(pair_add("a", "b"))
print(pair_add([1], [2]))
try { print(add_one("s")) } catch e { print(e.message) }
try { print(under_three([1])) } catch e { print(e.message) }

# --- a local compared against a constant and branched on ---------------------
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
fn negated(x) -> int {
    if not x { return 1 }
    return 2
}

fn compared(a: int, b: int) -> int {
    if not a == b { return 1 }
    return 2
}

print(negated(true))
print(negated(false))
print(compared(1, 2))
print(compared(2, 2))
try { print(negated(0)) } catch e { print(e.message) }

# --- unreachable try/finally ------------------------------------------------
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
fn dead_between() -> int {
    let _a = 1
    let b = 2
    let _c = 3
    return b
}

fn dead_around() -> int {
    let _p = 1
    let live = 20
    let _q = 3
    let _r = 4
    let _s = 5
    return live
}

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
