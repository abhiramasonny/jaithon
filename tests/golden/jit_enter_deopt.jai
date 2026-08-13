#: A side effect must happen exactly once when a body deoptimises on the very
#: call that compiled it.
#:
#: `callClosure` reaches the tier two ways. Once a function has a `jitFunc` it
#: goes through a branch that handles all four verdicts. The FIRST time -- the
#: call that crosses JAI_JIT_THRESHOLD and triggers the compile -- it goes
#: through `jaiJitEnter` instead, and that used to answer a bool: DONE or "no".
#: A body that ran and then DEOPTED read as "no", so the caller pushed a frame
#: and ran the function again FROM THE TOP, repeating everything the compiled
#: body had already done.
#:
#: Nothing caught it while a fresh body deopting on its first compiled call was
#: a rarity. An unarmed opcode now emits an UNCONDITIONAL deopt, so it is the
#: normal outcome for every function that mentions one -- and a function that
#: writes before reaching it does the write twice. `bag.items` below is the
#: shape it was found in, `tests/lang/test_jit_dict_items_iter.jai` counting
#: 6002 calls for 6001: one extra, once, on the call that compiled it.
#:
#: Counting rather than asserting a value is the point. The doubled work here
#: is a store to a field nothing else reads, so every result stays right and
#: only the counter knows. That is what a repeated effect looks like.
#:
#: A GOLDEN, not `tests/lang`: this only happens at the threshold crossing, and
#: under `jaithon test` these bodies never reach it.
import std.io

#: `made` counts entries. The list of tuples after it is what makes the body
#: deopt every call -- OP_BUILD_TUPLE has no arm -- so the store is always on
#: the wrong side of the guard.
class Bag {
    pub var made: int

    fn init(self) -> void { self.made = 0 }

    pub fn items(self) -> list[tuple[str, int]] {
        self.made += 1
        return [("u", 10), ("w", 20)]
    }
}

#: An append rather than a counter, so the effect is visible in a value too.
class Log {
    pub var seen: list[int]

    fn init(self) -> void { self.seen = [] }

    pub fn note(self, n: int) -> list[tuple[str, int]] {
        self.seen.push(n)
        return [("n", n)]
    }
}

#: Straight past the threshold in one loop: the interesting call is the 65th.
fn drive_bag(rounds: int) -> int {
    let b = Bag()
    var i = 0
    while i < rounds {
        let pairs = b.items()
        i += pairs.len() - 1
    }
    return b.made
}

fn drive_log(rounds: int) -> int {
    let g = Log()
    var i = 0
    while i < rounds {
        let pairs = g.note(i)
        i += pairs.len()
    }
    return g.seen.len()
}

fn main() -> void {
    #: One entry per round, whatever the tier did with any of them.
    print(f"bag 70 {drive_bag(70)}")
    print(f"bag 200 {drive_bag(200)}")
    print(f"log 70 {drive_log(70)}")
    print(f"log 200 {drive_log(200)}")
}
