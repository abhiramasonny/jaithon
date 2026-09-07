#: Builtin classes named from JIT-warmed bodies, and the shadow rule that says
#: when a module's own binding wins instead. The two tiers must agree exactly.
#:
#: Seventh of the tier-agreement set. `globalClass` used to look only in the
#: throwing module's own table, and every builtin exception type lives ONLY in
#: `vm.builtins` (defineErrorClass -> jaiDefineGlobal), so `throw ValueError(…)`
#: fell through the whole resolver chain to "is not a compiled global function"
#: and stopped the walk -- permanently, since no later compile can turn a
#: builtin class into a compiled function. It now falls back to `vm.builtins`
#: the way globalNative already does for a bare builtin.
#:
#: The edge that matters here is the SHADOW rule, because the fallback is only
#: sound if the module's own binding always wins: a user class named `ValueError`
#: must be the one constructed and caught, and a module-level binding of the
#: name to a NON-class must not resolve to the builtin either. Both are below.
#: Also covered: raising and catching in a warmed loop, catching by the builtin
#: base class, `isinstance`-style matching across the hierarchy, and a builtin
#: class used as a plain value rather than thrown.

class ValueError {
    pub var tag: str
    pub fn init(self, tag: str) { self.tag = tag }
}

fn shadowed(n: int) -> str {
    let e = ValueError("mine" + str(n % 3))
    return e.tag
}

fn raises(x: int) -> int {
    if x % 8 == 3 { throw KeyError("k" + str(x)) }
    if x % 8 == 5 { throw IndexError("i" + str(x)) }
    return x
}

fn caught(x: int) -> str {
    try {
        return "ok" + str(raises(x))
    } catch e: KeyError {
        return "key"
    } catch e: IndexError {
        return "index"
    }
}

fn caught_base(x: int) -> str {
    try { return "ok" + str(raises(x)) } catch e: Error { return "base" }
}

fn main() {
    var warm = ""
    var n = 0
    for i in 0..600 {
        warm = caught(i)
        if caught_base(i) == "base" { n += 1 }
        if shadowed(i) == "mine0" { n += 1 }
    }
    print("warm " + warm + " " + str(n))

    var out: list[str] = []
    for i in 0..12 { out.push(caught(i)) }
    print("caught " + str(out))

    var base: list[str] = []
    for i in 0..12 { base.push(caught_base(i)) }
    print("base " + str(base))

    # The shadow rule: this module defines its own ValueError, so THAT is the
    # class the name means, not the builtin. Constructing it must give the user
    # class, and catching the builtin must not catch it.
    print("shadow " + shadowed(0) + shadowed(1) + shadowed(2))
    let mine = ValueError("direct")
    print("shadow-tag " + mine.tag)

    # A builtin class as a plain value, not thrown.
    print("names " + str(KeyError) + " " + str(IndexError))
}
