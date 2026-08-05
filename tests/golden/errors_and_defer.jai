#: Exceptions, the built-in hierarchy, defer ordering, and assert (spec §5.4).

# --- catching by type -------------------------------------------------------
fn parse_positive(text: str) -> int {
    let parsed = text.to_int()
    if parsed == null { throw ParseError(f"not a number: {text}") }
    let n = parsed ?? 0
    if n <= 0 { throw ValueError(f"expected a positive number, got {n}") }
    return n
}

try {
    print(parse_positive("42"))
    print(parse_positive("-1"))
} catch e: ValueError {
    print(f"ValueError: {e.message}")
}

try {
    parse_positive("abc")
} catch e: ParseError {
    print(f"ParseError: {e.message}")
}

# A subclass is caught by its parent's handler.
try {
    throw FileNotFoundError("no such file: /nope")
} catch e: IOError {
    print(f"caught as IOError: {e.message}")
}

# Multiple types on one clause.
fn risky(which: int) -> void {
    if which == 0 { throw IndexError("bad index") }
    throw KeyError("bad key")
}

for which in 0..2 {
    try {
        risky(which)
    } catch e: IndexError | KeyError {
        print(f"lookup problem: {e.message}")
    }
}

# Catch-all, then rethrow.
fn outer() -> str {
    try {
        throw RuntimeError("inner failure")
    } catch e {
        return f"handled: {e.message}"
    }
}
print(outer())

# --- finally always runs ----------------------------------------------------
fn with_finally(fail: bool) -> str {
    try {
        if fail { throw ValueError("boom") }
        return "normal"
    } catch e: ValueError {
        return "caught"
    } finally {
        print("finally ran")
    }
}
print(with_finally(false))
print(with_finally(true))

# --- defer runs in reverse order, on every exit -----------------------------
fn defers_normal() -> int {
    defer { print("defer 1") }
    defer { print("defer 2") }
    print("body")
    return 7
}
print(defers_normal())

fn defers_on_throw() -> void {
    defer { print("cleanup A") }
    defer { print("cleanup B") }
    throw RuntimeError("failing")
}
try {
    defers_on_throw()
} catch e {
    print(f"after defers: {e.message}")
}

# --- built-in exceptions from the runtime -----------------------------------
try {
    let xs = [1, 2, 3]
    print(xs[10])
} catch e: IndexError {
    print("IndexError caught")
}

try {
    let d = {"a": 1}
    print(d["missing"])
} catch e: KeyError {
    print("KeyError caught")
}

try {
    print(1 / 0)
} catch e: DivisionByZeroError {
    print("DivisionByZeroError caught")
}

# --- assert -----------------------------------------------------------------
assert 1 + 1 == 2
try {
    assert 1 == 2, "arithmetic is broken"
} catch e: AssertionError {
    print(f"AssertionError: {e.message}")
}

print("done")
