#: `for … in X.items()` iterates lazily when X is a dict.
#:
#: dict.items() builds an N-element list of N fresh 2-tuples and keeps all N
#: alive at once, so every collection during the loop marks the lot. The loop
#: form takes ITER_DICT_ITEMS instead, which walks the table one pair at a time.
#: Measured on tests/bench/dict_iter: peak RSS 37.1 MB -> 30.3 MB (-18.2%),
#: instructions -22%. Wall time was unchanged on its own -- the per-entry tuple,
#: which this does not remove, was the dominant cost. OP_FOR_ITER_PAIR removes
#: that one (tests/golden/for_iter_pair.jai) and the two together are what took
#: the benchmark from 176ms to 117ms.
#:
#: The dispatch is at RUNTIME, so everything below must behave exactly as it did
#: before: `items()` called on anything else is still that object's own method,
#: and `items()` used as a value is still a list.
let d: dict[str, int] = {}
d["a"] = 1
d["b"] = 2
d["c"] = 3

# The loop form: lazy, same values, same order as the eager list.
var total = 0
var keys: list[str] = []
for (k, v) in d.items() {
    total += v
    keys.push(k)
}
print(total)
print(keys)

# items() as a VALUE is still a list, with everything a list can do.
let pairs = d.items()
print(pairs.len())
print(pairs[0][0])

# Iterating that list gives the same thing as iterating the dict lazily.
var again = 0
for (_k, v) in pairs { again += v }
print(again)

# A user class defining its own `items()` must still have it called.
class Bag {
    pub var made: int

    fn init(self) -> void { self.made = 0 }

    pub fn items(self) -> list[tuple[str, int]] {
        self.made += 1
        return [("x", 10), ("y", 20)]
    }
}

let b = Bag()
var bag_total = 0
for (_k, v) in b.items() { bag_total += v }
print(bag_total)
print(b.made)

# A comprehension over items() takes the same path.
let doubled = [v * 2 for (_k, v) in d.items()]
print(doubled.len())

# Mutating the dict mid-iteration is still caught.
try {
    for (k, _v) in d.items() { d[f"{k}!"] = 0 }
    print("NOT REACHED")
} catch _e: RuntimeError {
    print("mutation caught")
}
