#: Closures whose captured variable the compiled tier has an opinion about.
#:
#: Two separate things live here, and they meet on the same machinery.
#:
#: FIRST, a local captured BY REFERENCE is aliased by an ObjUpvalue pointing
#: straight into the VM's slot array, so the slot and the closure's view of it
#: are one piece of storage. The OSR tier caches locals in registers when they
#: fit, which quietly makes them two. `mutate_under_closure` below reads the
#: captured variable through the closure and writes it directly in the same
#: loop; with the slot in a register the read saw whatever it held when the
#: loop was entered, and since OSR is entered on a timer tick the answer was
#: different on every run and always short.
#:
#: SECOND, a call through a closure held in a variable is guarded on the
#: callee's ObjFunction, not on the closure -- one function, many closures,
#: each with its own captured cell. Once that guard has passed the body is
#: known and can be emitted in place of the call, reading its upvalue out of
#: the guarded register. `sum_via` is that shape: a fresh closure over a
#: different `step` every outer iteration, all sharing one ObjFunction.
#:
#: `pick` then puts TWO functions through one call site, so the guard has to
#: fail and hand the interpreter back a stack it can resume from.

# --- a captured local written in the loop that reads it ---------------------
fn mutate_under_closure(n: int) -> int {
    var base = 3
    let read = |x| x + base
    var acc = 0
    var i = 0
    while i < n {
        acc = read(acc)
        base += 1
        i += 1
    }
    return acc
}

# --- the same, with the closure doing the writing ---------------------------
fn write_through_closure(n: int) -> int {
    var tot = 0
    fn bump(d: int) -> int {
        tot += d
        return tot
    }
    var i = 0
    var last = 0
    while i < n {
        last = bump(2)
        i += 1
    }
    return tot + last
}

# --- a monomorphic indirect call site, one function and many closures -------
fn adder(step: int) -> fn(int) -> int {
    return |x| x + step
}

fn sum_via(f: fn(int) -> int, start: int, times: int) -> int {
    var acc = start
    var i = 0
    while i < times {
        acc = f(acc)
        i += 1
    }
    return acc
}

# --- a float upvalue, so the tag guard inside the inlined body has work -----
fn scaler(step: float) -> fn(float) -> float {
    return |x| x * step + 1.0
}

fn fsum_via(f: fn(float) -> float, start: float, times: int) -> float {
    var acc = start
    var i = 0
    while i < times {
        acc = f(acc)
        i += 1
    }
    return acc
}

# --- two different functions reaching one call site -------------------------
fn twice(x: int) -> int {
    return x * 2
}

fn pick(which: int) -> fn(int) -> int {
    if which == 0 {
        return |x| x + 1
    }
    return twice
}

fn polymorphic(times: int) -> int {
    var acc = 1
    var i = 0
    while i < times {
        let g = pick(i % 7)
        acc = g(acc) % 1000003
        i += 1
    }
    return acc
}

fn main() -> int {
    print(mutate_under_closure(200000))
    print(write_through_closure(100000))

    var total = 0
    var k = 1
    while k <= 20 {
        total = (total + sum_via(adder(k), 0, 20000)) % 1000000007
        k += 1
    }
    print(total)

    var ft = 0.0
    var j = 1
    while j <= 10 {
        ft = ft + fsum_via(scaler(1.0000001), 1.0, 20000)
        j += 1
    }
    print(f"{ft:.6f}")

    print(polymorphic(200000))
    return 0
}
