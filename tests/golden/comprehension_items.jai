#: `[... for (k, v) in d.items()]` gets the same lazy-pair fusion as the `for`
#: statement (tests/golden/iter_items.jai): `_comprehension` in emit.jai now
#: reuses `_items_iterable`/`_is_items_call` for a clause's iterable exactly
#: as `_for` does, instead of always emitting `X.items(); OP_GET_ITER`. On a
#: dict receiver that walks the table lazily instead of building an
#: N-element list of N fresh 2-tuples first.
#:
#: The match is on SHAPE alone (`_is_items_call`), so the opcode decides at
#: RUNTIME whether the receiver is really a dict. Everything below must
#: behave exactly as it did before the fusion: a user class's own `items()`
#: is still called, a comprehension with no `.items()` clause is untouched,
#: and a comprehension that mixes an `.items()` clause with an ordinary one
#: still produces the same values in the same order.
let d: dict[str, int] = {}
d["a"] = 1
d["b"] = 2
d["c"] = 3

let pairs = [(k, v) for (k, v) in d.items()]
print(pairs)

let doubled = {k: v * 2 for (k, v) in d.items() if v > 1}
print(doubled)

let values = {v for (_k, v) in d.items()}
print(values)

let whole = [pair for pair in d.items()]
print(whole)

let e: dict[str, int] = {}
e["x"] = 10
e["y"] = 20
let cross = [(n, k, v) for n in [1, 2] for (k, v) in e.items()]
print(cross)

let plain = [i * 3 for i in [1, 2, 3]]
print(plain)

class Bag {
    pub var made: int

    fn init(self) -> void { self.made = 0 }

    pub fn items(self) -> list[tuple[str, int]] {
        self.made += 1
        return [("p", 100), ("q", 200)]
    }
}

let b = Bag()
let from_bag = [v for (_k, v) in b.items()]
print(from_bag)
print(b.made)

var total = 0
for (_k, v) in d.items() { total += v }
print(total)
pairs.push(("z", 99))
print(pairs.len())
print(total + pairs.len())
