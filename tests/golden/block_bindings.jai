#: A block's bindings die when the block exits (spec §5.2, §6).
#:
#: Sibling blocks share slots — the resolver hands the next declaration the
#: slots the block just gave up — so a closure that outlived a block and still
#: aliased one of its slots would read whatever landed there afterwards. That is
#: the worst kind of defect: no error, just a wrong number. The emitter closes a
#: block's by-reference captures on every path out of it: fall-through, `return`
#: (before the inlined `finally` copies and the defer thunks, which borrow the
#: same slots), and, for a block inside `try`, at the handler the unwinder jumps
#: to. `break` and `continue` are covered by the loop's own iteration close,
#: which sits at or below every block base inside the loop; `tests/golden/
#: loop_bindings.jai` pins that half.
#:
#: A `let` is captured by value (spec §6) and needs no close, and a block that
#: captures nothing by reference emits no instruction at all — `caseBlockClose`
#: in tests/verify_chunk.c reads the emitted chunk and pins both.

# --- the escaped closure must not read the next declaration -----------------
fn escape_from_if(c: bool) -> fn() -> int {
    var g = fn() -> int { return 0 }
    if c {
        var a = 1
        g = fn() -> int { return a }
    }
    var b = 99  # takes the slot `a` held
    print(b)
    return g
}
print(escape_from_if(true)())

# --- nested blocks: the inner one closes its own base, and only that --------
#: `outer` is still live when the inner block ends, so it keeps aliasing the
#: frame and the later `outer = 10` is visible to a closure made before it.
fn nested() -> list[fn() -> int] {
    var out: list[fn() -> int] = []
    {
        var outer = 1
        {
            var inner = 2
            out.push(|| outer + inner)
        }
        var filler = 900  # takes the slot `inner` held
        print(filler)
        out.push(|| outer)
        outer = 10
    }
    var after = 999
    print(after)
    return out
}
print(nested().map(|f| f()))

# --- escaping via `return` from inside a block ------------------------------
fn escape_via_return() -> fn() -> int {
    if true {
        var a = 7
        return || a * 2
    }
    return || 0
}
print(escape_via_return()())

# --- `return` closes before the `finally` copy that reuses the slot ---------
#: The frame-wide close at OP_RETURN is too late: `finally` runs first and its
#: `junk` lands in the slot `a` held.
fn return_through_finally() -> fn() -> int {
    var g = fn() -> int { return 0 }
    try {
        var a = 5
        g = || a
        return g
    } finally {
        var junk = 77
        print(junk)
    }
}
print(return_through_finally()())

# --- and before the defer thunks, which borrow the same slots ---------------
fn return_through_defer() -> fn() -> int {
    defer {
        var junk = 88
        print(junk)
    }
    if true {
        var a = 6
        return || a
    }
    return || 0
}
print(return_through_defer()())

# --- capture inside a `try`, thrown out of ----------------------------------
#: The unwinder jumps straight to the handler, skipping the block close at the
#: bottom of the body, and the catch binding is a sibling of the try body's
#: scope — it lands in the same slot.
fn escape_from_try() -> fn() -> int {
    var g = fn() -> int { return 0 }
    try {
        var a = 3
        g = || a
        throw ValueError("boom")
    } catch e {
        print(e.message)
    }
    var b = 99
    print(b)
    return g
}
print(escape_from_try()())

# --- a block nested inside the `try` body, thrown out of --------------------
fn escape_from_nested_try() -> fn() -> int {
    var g = fn() -> int { return 0 }
    try {
        var pad = 1
        print(pad)
        if true {
            var a = 4
            g = || a
            throw ValueError("boom")
        }
    } catch _e {
    }
    var x = 11
    var y = 22
    var z = 33
    print(x + y + z)
    return g
}
print(escape_from_nested_try()())

# --- a `try` in a loop takes the dynamic-handler path -----------------------
fn caught_each_pass() -> list[fn() -> int] {
    var out: list[fn() -> int] = []
    for i in 0..3 {
        try {
            var v = i + 1
            out.push(|| v)
            if i == 1 { throw ValueError("x") }
        } catch _e {
            var junk = -1
            print(junk)
        }
    }
    return out
}
print(caught_each_pass().map(|f| f()))

# --- a block used for its value closes after the value is on the stack ------
fn block_value(c: bool) -> fn() -> int {
    let g = if c {
        var a = 4
        || a
    } else {
        || 0
    }
    var pad = 55
    print(pad)
    return g
}
print(block_value(true)())

# --- a match arm is a block like any other ----------------------------------
fn from_match_arm(n: int) -> fn() -> int {
    var g = fn() -> int { return 0 }
    match n {
        1 => {
            var a = 21
            g = || a
        }
        _ => { g = || -1 }
    }
    var pad = 66
    print(pad)
    return g
}
print(from_match_arm(1)())

# --- a `let` capture is a snapshot and needs no close -----------------------
fn by_value(c: bool) -> fn() -> int {
    var g = fn() -> int { return 0 }
    if c {
        let a = 8
        g = || a
    }
    var b = 99
    print(b)
    return g
}
print(by_value(true)())

# --- a block inside a loop body closes on every pass ------------------------
#: The loop's iteration close covers the body block itself; a block nested in
#: the body has a base of its own and closes when *it* ends, or the next pass's
#: bindings would be read instead.
fn nested_in_loop() -> list[fn() -> int] {
    var out: list[fn() -> int] = []
    for i in 0..3 {
        {
            var v = i + 1
            out.push(|| v)
        }
        var reuse = -1  # takes the slot `v` held
        print(reuse)
    }
    return out
}
print(nested_in_loop().map(|f| f()))
