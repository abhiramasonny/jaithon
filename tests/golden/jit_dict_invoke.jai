#: A built-in method on a receiver the compiled tier types only as "some
#: object", and the dict store that has to compile with it.
#:
#: `counts.get(key, 0)` was the largest single decline site in the benchmark
#: suite. Three separate things had to be answered for it, and each has a case
#: here.
#:
#: WHICH METHOD. Resolved from the observed receiver, because a built-in method
#: is a function of (receiver type, name) and the name is in the instruction.
#:
#: THAT IT IS STILL THAT TYPE. A SLOT_OBJ slot pins nothing, so the object type
#: is guarded before anything is consumed. `mixed_receiver` puts a dict and a
#: string through one site so the guard has to miss and hand back a stack the
#: interpreter can resume from -- the receiver and both arguments still on it.
#:
#: WHAT COMES BACK. A built-in's result kind is not derivable from its
#: arguments, so it is PREDICTED from what the site was observed to return and
#: guarded after the call, deoptimising to the instruction AFTER it. Each of
#: `int_default`, `bool_default`, `str_default` and `float_default` pins one
#: kind; `bool_default` in particular, because a bool's payload is ONE BYTE of
#: the Value union and an eight-byte load of it returns whatever the slot held
#: before -- which is how this first went wrong, inside the compiler itself.
#:
#: The prediction's own failure mode -- the site returning a kind other than
#: the one recorded -- is not a case here, because a statically typed dict will
#: not hold two value kinds at one key. `JAITHON_JIT_DEOPT_STRESS=1` is the
#: systematic version of that test: it forces every guard in this file down its
#: deopt path, and the output must not change.
#:
#: `discarded` invokes for effect only, where no prediction is needed at all,
#: and `dict_store` is the write half without which none of this compiles: the
#: loop's decline simply moved from the `get` to the `[k] =`.
fn int_default(n: int) -> int {
    var counts = {}
    var i = 0
    while i < n {
        let key = f"k{i % 64}"
        counts[key] = counts.get(key, 0) + 1
        i += 1
    }
    var total = 0
    for (_k, v) in counts.items() { total += v }
    return total + counts.len()
}

fn bool_default(n: int) -> int {
    var seen = {}
    var hits = 0
    var i = 0
    while i < n {
        let key = f"b{i % 32}"
        if seen.get(key, false) { hits += 1 }
        seen[key] = true
        i += 1
    }
    return hits
}

fn str_default(n: int) -> int {
    let names = {"a": "alpha", "b": "beta"}
    var out = {}
    var i = 0
    while i < n {
        let key = if i % 3 == 0 { "a" } elif i % 3 == 1 { "b" } else { "c" }
        out[key] = names.get(key, "?")
        i += 1
    }
    var width = 0
    for (_k, v) in out.items() { width += v.len() }
    return width
}

fn float_default(n: int) -> float {
    var weights = {"one": 1.5}
    var total = 0.0
    var i = 0
    while i < n {
        let key = if i % 2 == 0 { "one" } else { "two" }
        total += weights.get(key, 0.25)
        i += 1
    }
    return total
}

# --- the receiver's type changes under a compiled site ----------------------
fn size_of(thing: any) -> int {
    return thing.len()
}

fn mixed_receiver(n: int) -> int {
    let d = {"x": 1, "y": 2, "z": 3}
    let s = "hello"
    var total = 0
    var i = 0
    while i < n {
        total += size_of(if i % 128 == 127 { s } else { d })
        i += 1
    }
    return total
}

# --- invoked for effect, so the result needs no prediction ------------------
fn discarded(n: int) -> int {
    var log = []
    var i = 0
    while i < n {
        log.push(i % 8)
        if log.len() > 64 { log.clear() }
        i += 1
    }
    return log.len()
}

# --- the store half, on its own ---------------------------------------------
fn dict_store(n: int) -> int {
    var d = {}
    var i = 0
    while i < n {
        d[i % 100] = i
        i += 1
    }
    var total = 0
    for (_k, v) in d.items() { total += v }
    return total
}

fn main() -> int {
    print(int_default(200000))
    print(bool_default(200000))
    print(str_default(200000))
    print(f"{float_default(200000):.4f}")
    print(mixed_receiver(200000))
    print(discarded(200000))
    print(dict_store(200000))
    return 0
}
