#: A name in call position binds to the innermost binding (spec §3, §6, §7).
#:
#: Parameters and locals are ordinary bindings: they win over a method of the
#: enclosing class or trait, over the enclosing function's own name, and over a
#: module-level function. A call through such a binding is checked against the
#: binding's own `fn` type — its arity, its parameter types — and never against
#: whatever declaration happens to share the name.
fn twice(n: int) -> int { return n * 2 }

# --- a parameter beats a method of the same class ---------------------------
class Ledger {
    let entries: list[int]

    pub fn init(self, entries: list[int]) { self.entries = entries }

    #: The class has its own one-argument `weigh`; the parameter is a
    #: two-argument function and the call must take the parameter's shape.
    pub fn weigh(self, value: int) -> int { return value }

    pub fn fold(self, weigh: fn(int, int) -> int) -> int {
        var total = 0
        for entry in self.entries { total = weigh(total, entry) }
        return total
    }
}

# --- a parameter beats a method of an implemented trait ---------------------
trait Ranked {
    fn rank(self) -> int

    #: A trait default body: `rank` here is the parameter, not the requirement
    #: declared two lines above it (core.jai's `Iterator.min_by` shape).
    fn scaled(self, rank: fn(int) -> int) -> int { return rank(self.rank()) }
}

class Card: Ranked {
    let value: int

    pub fn init(self, value: int) { self.value = value }

    pub fn rank(self) -> int { return self.value }
}

# --- a parameter beats the enclosing function's own name --------------------
fn apply(apply: fn(int) -> int, n: int) -> int { return apply(n) }

# --- a local beats a module-level function ----------------------------------
fn shadowed() -> int {
    let twice = fn(n: int) -> int { return n + 100 }
    return twice(1)
}

let ledger = Ledger([1, 2, 3])
print(ledger.fold(fn(a: int, b: int) -> int { return a + b }))
print(ledger.weigh(9))
print(Card(7).scaled(twice))
print(apply(twice, 5))
print(shadowed())
print(twice(1))
