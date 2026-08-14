#: `for (a, b) in …` fused to OP_FOR_ITER_PAIR (spec/BYTECODE.md §3.3).
#:
#: The peephole rewrites FOR_ITER; UNPACK 2 255; BIND a; BIND b into one
#: instruction that binds both components without building a pair, which is
#: worth 4.8M allocations and 306 MB on tests/bench/dict_iter. Everything below
#: must print the same thing at every -O level, so this file is the proof that
#: it is a fast path and never a restriction.
#:
#: Every pair here has a `str` on one side and an `int` on the other, or two
#: distinguishable ints. That is deliberate: an opcode that swapped its two
#: operands passes silently on `(int, int)` pairs of equal value, and swapping
#: them is the single easiest mistake to make -- the loop pattern's elements
#: are not in source order relative to the slots the resolver assigns, so the
#: slots have to be read off the two BINDs, not off the pattern.

# --- a dict, the shape the opcode exists for --------------------------------
let d: dict[str, int] = {}
d["aa"] = 1
d["bbb"] = 2
d["cccc"] = 3

var keys: list[str] = []
var values: list[int] = []
for (k, v) in d.items() {
    keys.push(k)
    values.push(v)
}
print(keys)
print(values)

# --- a list of tuples, and a list of lists ----------------------------------
var seen: list[str] = []
for (name, n) in [("x", 1), ("y", 2)] { seen.push(f"{name}={n}") }
for (name, n) in [["p", 7], ["q", 8]] { seen.push(f"{name}={n}") }
print(seen)

# --- enumerate and zip, which yield pairs from something that is not a dict --
var enumerated: list[str] = []
for (i, s) in ["a", "b", "c"].enumerate() { enumerated.push(f"{i}{s}") }
print(enumerated)

# --- the loop variables really are separate slots ---------------------------
var lastKey = ""
var lastVal = 0
for (k, v) in d.items() {
    lastKey = k
    lastVal = v
}
print(lastKey)
print(lastVal)

# --- nested pair loops, sharing nothing -------------------------------------
var cross = 0
for (_k1, v1) in d.items() {
    for (_k2, v2) in d.items() { cross += v1 * v2 }
}
print(cross)

# --- shapes the window must NOT match ---------------------------------------
var triples = 0
for (a, b, c) in [(1, 2, 3), (4, 5, 6)] { triples += a * 100 + b * 10 + c }
print(triples)

var rests: list[str] = []
for (head, ...tail) in [[1, 2, 3], [4, 5]] { rests.push(f"{head}:{tail}") }
print(rests)

# --- a single-variable loop over pairs is unchanged --------------------------
var whole = 0
for pair in [(1, 2), (3, 4)] { whole += pair[0] * pair[1] }
print(whole)

# --- a user class supplying its own pairs ------------------------------------
class Bag {
    pub var made: int

    fn init(self) -> void { self.made = 0 }

    pub fn items(self) -> list[tuple[str, int]] {
        self.made += 1
        return [("u", 10), ("w", 20)]
    }
}

let b = Bag()
var bagKeys = ""
var bagTotal = 0
for (k, v) in b.items() {
    bagKeys = bagKeys + k
    bagTotal += v
}
print(bagKeys)
print(bagTotal)
print(b.made)

# --- a comprehension takes the same window ----------------------------------
print([f"{k}{v}" for (k, v) in d.items()])

# --- errors: word for word what UNPACK 2 raised ------------------------------
fn hide(x) { return x }

try {
    for (_a, _b) in hide([(1, 2, 3)]) { print("NOT REACHED") }
} catch e: ValueError {
    print(e.message)
}

try {
    for (_a, _b) in hide([(1,)]) { print("NOT REACHED") }
} catch e: ValueError {
    print(e.message)
}

try {
    for (_a, _b) in hide([1, 2]) { print("NOT REACHED") }
} catch e: TypeError {
    print(e.message)
}

try {
    for (_a, _b) in hide([["only"]]) { print("NOT REACHED") }
} catch e: ValueError {
    print(e.message)
}

var partialA = ""
var partialB = 0
try {
    for (pa, pb) in hide([("ok", 5), ("bad", 6, 7)]) {
        partialA = pa
        partialB = pb
    }
} catch e: ValueError {
    print(e.message)
}
print(partialA)
print(partialB)

# --- mutating the source under the loop is still caught ----------------------
try {
    for (k, _v) in d.items() { d[f"{k}!"] = 0 }
    print("NOT REACHED")
} catch _e: RuntimeError {
    print("dict mutation caught")
}

let growing = [(1, 2), (3, 4)]
try {
    for (_a, _b) in growing { growing.push((5, 6)) }
    print("NOT REACHED")
} catch _e: RuntimeError {
    print("list mutation caught")
}

# --- an empty source jumps straight out --------------------------------------
let empty: dict[str, int] = {}
var never = 0
for (_k, _v) in empty.items() { never += 1 }
print(never)

# --- hot enough to reach the compiled tier -----------------------------------
fn total_pairs(xs: list[tuple[int, int]]) -> int {
    var t = 0
    for (a, b) in xs { t += a * b }
    return t
}

fn spin(xs: list[tuple[int, int]], rounds: int) -> int {
    var acc = 0
    for _r in 0..rounds { acc += total_pairs(xs) }
    return acc
}

let ints = [(1, 2), (3, 4), (5, 6), (7, 8)]
print(spin(ints, 4000))

fn total_any(xs) -> float {
    var t = 0.0
    for (a, b) in xs { t = t + a * b }
    return t
}

fn spin_any(xs, rounds: int) -> float {
    var acc = 0.0
    for _r in 0..rounds { acc = acc + total_any(xs) }
    return acc
}

let mixed = [(1, 2), (3, 4.5), (5, 6)]
print(spin_any(mixed, 4000))

# --- the compiled arm's guards, failing for a real reason --------------------
fn sum_pairs(xs) -> int {
    var t = 0
    for (a, b) in xs { t = t + a * b }
    return t
}

fn spin_pairs(xs, rounds: int) -> int {
    var acc = 0
    for _r in 0..rounds { acc = acc + sum_pairs(xs) }
    return acc
}

let asTuples = hide([(1, 2), (3, 4), (5, 6), (7, 8)])
let asLists = hide([[1, 2], [3, 4], [5, 6], [7, 8]])
print(spin_pairs(asTuples, 3000))
print(spin_pairs(asLists, 3000))
print(spin_pairs(asTuples, 10))
