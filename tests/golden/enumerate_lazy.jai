#: `for (i, x) in xs.enumerate()` over a list, which OP_INVOKE turns into an
#: ITER_LIST_ENUM snapshot (JAITHON_LAZY_ENUMERATE, src/vm/vm.c) and both
#: compiled tiers arm (emitForIterPairEnum). Everything here must print the
#: same thing with the switch off -- the eager `list.enumerate()`, N tuples in
#: a list -- and under every tier configuration, so this file is the proof that
#: the snapshot is a fast path and never a change of meaning.
#:
#: The .expected was captured with JAITHON_LAZY_ENUMERATE=0 JAITHON_NO_JIT=1:
#: the interpreter running the eager call, which is the reference everything
#: else is measured against.
#:
#: Every function below is driven past the function tier's threshold, and the
#: last section walks a list long enough for the OSR tier to land on its head.

class Node {
    pub var k: int
    fn init(self, k: int) -> void { self.k = k }
    pub fn weight(self) -> int { return self.k }
}

class Leaf {
    pub var k: int
    fn init(self, k: int) -> void { self.k = k }
    pub fn weight(self) -> int { return self.k * 10 }
}

# --- the common shape: a `return` inside the loop ------------------------
fn find(xs: list[Node], want: int) -> int {
    for (i, x) in xs.enumerate() {
        if x.weight() == want { return i }
    }
    return -1
}

# --- break, continue, and a total that proves the index is the index ------
fn skim(xs: list[int]) -> int {
    var total = 0
    for (i, x) in xs.enumerate() {
        if x < 0 { continue }
        if i > 5 { break }
        total += i * 100 + x
    }
    return total
}

# --- every scalar storage a list can have ---------------------------------
fn floats(xs: list[float]) -> float {
    var acc = 0.0
    for (i, x) in xs.enumerate() { acc += x * (i + 1) }
    return acc
}

fn bools(xs: list[bool]) -> int {
    var acc = 0
    for (i, x) in xs.enumerate() { if x { acc += i } }
    return acc
}

fn strs(xs: list[str]) -> str {
    var acc = ""
    for (i, x) in xs.enumerate() { acc = acc + f"{i}{x}" }
    return acc
}

fn lists(xs: list[list[int]]) -> int {
    var acc = 0
    for (i, x) in xs.enumerate() { acc += i * x.len() }
    return acc
}

# --- nulls beside instances, dense and sparse ------------------------------
fn nully(xs: list[Node?]) -> int {
    var acc = 0
    for (i, x) in xs.enumerate() {
        if x is null { acc += i } else { acc += x.weight() }
    }
    return acc
}

# --- two classes in one list -------------------------------------------------
fn mixed(xs: list[any]) -> int {
    var acc = 0
    for (i, x) in xs.enumerate() { acc += x.weight() + i }
    return acc
}

# --- mutating the source inside the loop: the eager call copied, so must this
fn grows(xs: list[int]) -> int {
    var acc = 0
    for (i, x) in xs.enumerate() {
        xs.push(x + 100)
        acc += i + x
    }
    return acc
}

fn shrinks(xs: list[int]) -> int {
    var acc = 0
    for (i, x) in xs.enumerate() {
        if xs.len() > 0 { xs.pop() }
        acc += i * 10 + x
    }
    return acc
}

fn rewrites(xs: list[int]) -> int {
    var acc = 0
    for (i, x) in xs.enumerate() {
        if i + 1 < xs.len() { xs[i + 1] = -1 }
        acc += x
    }
    return acc
}

# --- nested, and a closure capturing both loop variables --------------------
fn nested(xs: list[int], ys: list[int]) -> int {
    var acc = 0
    for (i, x) in xs.enumerate() {
        for (j, y) in ys.enumerate() { acc += (i + 1) * (j + 1) * x * y }
    }
    return acc
}

fn captured(xs: list[int]) -> list[fn() -> int] {
    var fs: list[fn() -> int] = []
    for (i, x) in xs.enumerate() { fs.push(fn() -> int { return i * 1000 + x }) }
    return fs
}

# --- an exception thrown out of the loop --------------------------------------
fn throws(xs: list[int]) -> int {
    var acc = 0
    try {
        for (i, x) in xs.enumerate() {
            if x == 0 { throw ValueError(f"zero at {i}") }
            acc += x
        }
    } catch e: ValueError {
        return -acc
    }
    return acc
}

# --- shapes that must NOT take the snapshot ----------------------------------
fn asValue(xs: list[str]) -> str {
    let pairs = xs.enumerate()
    var acc = f"{pairs.len()}:"
    for (i, s) in pairs { acc = acc + f"{i}{s}" }
    return acc
}

fn withStart(xs: list[str]) -> str {
    var acc = ""
    for (i, s) in xs.enumerate(7) { acc = acc + f"{i}{s}" }
    return acc
}

fn whole(xs: list[str]) -> str {
    var acc = ""
    for pair in xs.enumerate() { acc = acc + f"{pair[0]}{pair[1]}" }
    return acc
}

class Bag {
    pub var made: int
    fn init(self) -> void { self.made = 0 }
    pub fn enumerate(self) -> list[tuple[int, str]] {
        self.made += 1
        return [(self.made, "own")]
    }
}

fn own(b: Bag) -> str {
    var acc = ""
    for (i, s) in b.enumerate() { acc = acc + f"{i}{s}" }
    return acc
}

from std.iter import iter

fn adapter(xs: list[str]) -> str {
    var acc = ""
    for (i, s) in iter(xs).enumerate() { acc = acc + f"{i}{s}" }
    return acc
}

fn empty(xs: list[int]) -> int {
    var acc = 7
    for (i, x) in xs.enumerate() { acc += i + x }
    return acc
}

#: `.enumerate()` whose result is used as a list rather than iterated: the
#: next opcode is not the GET_ITER, so the swap must not fire and the eager
#: list has to be there to be measured and indexed.
fn chained(xs: list[str]) -> str {
    let n = xs.enumerate().len()
    let second = xs.enumerate()[1]
    return f"{n}/{second}"
}

#: Emptied under the loop. The eager form copied the elements, so the walk
#: finishes on what was there when it started; the snapshot must agree.
fn cleared(xs: list[int]) -> str {
    var seen: list[int] = []
    for (i, x) in xs.enumerate() {
        seen.push(i * 100 + x)
        xs.clear()
    }
    return f"{seen}/{xs.len()}"
}

#: Rebound under the loop -- the name now holds a different list entirely.
fn rebound(xs: list[int]) -> str {
    var seen: list[int] = []
    var ys = xs
    for (i, x) in ys.enumerate() {
        seen.push(i * 100 + x)
        ys = [7, 7]
    }
    return f"{seen}/{ys}"
}

#: An index assignment under the loop: the element the walk yields is the one
#: the copy took, not the one that is there now.
fn overwritten(xs: list[int]) -> str {
    var seen: list[int] = []
    for (i, x) in xs.enumerate() {
        xs[i] = x + 1000
        seen.push(x)
    }
    return f"{seen}/{xs}"
}

#: The receivers that are NOT lists. `enumerate(x)` is the free function and
#: has no list native behind it, so every one of these keeps the slow path --
#: and a str, a dict, a range, a set and a tuple all reach it differently.
fn others(d: dict[str, int], s: str, st: set[int], tp: tuple[int, int, int]) -> str {
    var out = ""
    for (i, kv) in enumerate(d) { out += f"{i}{kv}" }
    for (i, c) in enumerate(s) { out += f"{i}{c}" }
    for (i, v) in enumerate(range(3, 6)) { out += f"{i}v{v}" }
    for (i, v) in enumerate(st) { out += f"{i}s{v}" }
    for (i, v) in enumerate(tp) { out += f"{i}t{v}" }
    return out
}

#: An enumerate loop that recurses through itself, so a frame holding one
#: snapshot always sits below another.
fn deep(xs: list[int], n: int) -> int {
    var acc = 0
    for (i, x) in xs.enumerate() {
        if n > 0 and i == 0 { acc += deep(xs, n - 1) }
        acc += i + x
    }
    return acc
}

fn main() -> void {
    var nodes: list[Node] = []
    for j in 0..40 { nodes.push(Node(j)) }
    var ints: list[int] = [3, -1, 4, 1, -5, 9, 2, 6, 5, 3]
    var fl: list[float] = [0.5, 1.25, 2.0]
    var bl: list[bool] = [true, false, true, true]
    var sl: list[str] = ["a", "b", "c"]
    var ll: list[list[int]] = [[1], [1, 2], []]
    var sparse: list[Node?] = []
    for j in 0..200 { sparse.push(if j == 100 { null } else { Node(j) }) }
    var dense: list[Node?] = []
    for j in 0..200 { dense.push(if j % 2 == 0 { null } else { Node(j) }) }
    var both: list[any] = [Node(1), Leaf(2), Node(3), Leaf(4)]
    var wide: list[int] = []
    for j in 0..3000 { wide.push(j % 11) }

    var r_find = 0
    var r_skim = 0
    var r_floats = 0.0
    var r_bools = 0
    var r_strs = ""
    var r_lists = 0
    var r_sparse = 0
    var r_dense = 0
    var r_mixed = 0
    var r_grows = 0
    var r_shrinks = 0
    var r_rewrites = 0
    var r_nested = 0
    var r_captured = 0
    var r_throws = 0
    var r_asValue = ""
    var r_withStart = ""
    var r_whole = ""
    var r_own = ""
    var r_adapter = ""
    var r_empty = 0
    var r_wide = 0
    var r_chained = ""
    var r_cleared = ""
    var r_rebound = ""
    var r_overwritten = ""
    var r_others = ""
    var r_deep = 0
    let dd: dict[str, int] = {"a": 1, "b": 2}
    let tt = (9, 8, 7)
    var ss: set[int] = {5}
    let bag = Bag()
    for pass in 0..100 {
        r_find = find(nodes, 37) * 1000 + find(nodes, 99)
        r_skim = skim(ints)
        r_floats = floats(fl)
        r_bools = bools(bl)
        r_strs = strs(sl)
        r_lists = lists(ll)
        r_sparse = nully(sparse)
        r_dense = nully(dense)
        r_mixed = mixed(both)
        r_grows = grows([1, 2, 3])
        r_shrinks = shrinks([1, 2, 3, 4])
        r_rewrites = rewrites([1, 2, 3, 4])
        r_nested = nested([1, 2], [3, 4, 5])
        var fs = captured([5, 6, 7])
        r_captured = 0
        for f in fs { r_captured += f() }
        r_throws = throws([1, 2, 0, 4])
        r_asValue = asValue(sl)
        r_withStart = withStart(sl)
        r_whole = whole(sl)
        r_own = own(bag)
        r_adapter = adapter(sl)
        r_empty = empty([])
        r_wide = 0
        for (i, x) in wide.enumerate() {
            if x == 3 { r_wide += i }
        }
        r_chained = chained(sl)
        r_cleared = cleared([10, 20, 30])
        r_rebound = rebound([10, 20, 30])
        r_overwritten = overwritten([10, 20, 30])
        r_others = others(dd, "xyz", ss, tt)
        r_deep = deep([3, 5, 7, 9], 40)
    }
    print(r_find)
    print(r_skim)
    print(r_floats)
    print(r_bools)
    print(r_strs)
    print(r_lists)
    print(r_sparse)
    print(r_dense)
    print(r_mixed)
    print(r_grows)
    print(r_shrinks)
    print(r_rewrites)
    print(r_nested)
    print(r_captured)
    print(r_throws)
    print(r_asValue)
    print(r_withStart)
    print(r_whole)
    print(r_own)
    print(r_adapter)
    print(r_empty)
    print(r_wide)
    print(r_chained)
    print(r_cleared)
    print(r_rebound)
    print(r_overwritten)
    print(r_others)
    print(r_deep)
}

#: The module frame, not a function frame -- the swap is in OP_INVOKE and has
#: no frame of its own to depend on, and this is the one loop here that runs
#: outside a call.
var top: list[str] = ["p", "q", "r"]
var tacc = 0
for (i, s) in top.enumerate() { tacc += i + s.len() }
print(tacc)
