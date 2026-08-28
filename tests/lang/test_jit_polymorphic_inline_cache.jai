#: One call site, many receiver classes, hot enough to reach the compiled tier.
#:
#: The tier used to emit a descriptor against `jitInvokeByName` for every
#: invoke whose receiver class it could not pin, and that path resolves the
#: method by NAME through the program-wide megamorphic table -- six frames of C
#: for a call whose answer the site's own inline cache already held. Measured
#: on tests/bench/poly_dispatch, the same loop with one receiver class instead
#: of eight runs 1.62x faster, and that gap was the glue.
#:
#: The arm compares the receiver's shape against each way the cache recorded
#: and branches straight into that callee's compiled entry. A shape no way
#: holds falls through to exactly the descriptor the site emitted before, so a
#: miss costs one compare per way and nothing else. Worth 1.474x on
#: poly_dispatch, which is 91% of the 1.62x that was there to take.
#:
#: What these check is that every arm answers what the interpreter answers.
#: A polymorphic site is the one place a wrong branch is SILENT: calling the
#: wrong class's method reads fields at the wrong layout and lands inside the
#: instance's own allocation, so it returns a plausible number rather than
#: crashing. Run with JAITHON_JIT_PIC=0 and JAITHON_NO_JIT=1 too; all three
#: must agree.
# jit-compiles: sum_ints, any_flag, sum_reals
from std.test import assert_eq, assert_true

let ROUNDS = 300

trait Shape {
    fn area(self, n: int) -> int
}

class Sq: Shape {
    pub var k: int

    fn init(self, k: int) { self.k = k }

    pub fn area(self, n: int) -> int { return n * self.k }
}

class Tri: Shape {
    pub var k: int

    fn init(self, k: int) { self.k = k }

    pub fn area(self, n: int) -> int { return n + self.k }
}

class Hex: Shape {
    pub var k: int

    fn init(self, k: int) { self.k = k }

    pub fn area(self, n: int) -> int { return n - self.k }
}

class Oct: Shape {
    pub var k: int

    fn init(self, k: int) { self.k = k }

    pub fn area(self, n: int) -> int { return n * 2 + self.k }
}

#: A fifth and sixth class, so the site goes past the four ways the cache
#: holds. Those two must still get the right answer through the fall-through.
class Pent: Shape {
    pub var k: int

    fn init(self, k: int) { self.k = k }

    pub fn area(self, n: int) -> int { return n - self.k * 2 }
}

class Dec: Shape {
    pub var k: int

    fn init(self, k: int) { self.k = k }

    pub fn area(self, n: int) -> int { return n + self.k * 3 }
}

fn shapes(count: int) -> list[Shape] {
    var out: list[Shape] = []
    for index in 0..count {
        let k = index % 5 + 1
        if index % 6 == 0 {
            out.push(Sq(k))
        } elif index % 6 == 1 {
            out.push(Tri(k))
        } elif index % 6 == 2 {
            out.push(Hex(k))
        } elif index % 6 == 3 {
            out.push(Oct(k))
        } elif index % 6 == 4 {
            out.push(Pent(k))
        } else {
            out.push(Dec(k))
        }
    }
    return out
}

fn sum_ints(items: list[Shape], rounds: int) -> int {
    var total = 0
    var round = 0
    while round < rounds {
        for item in items { total += item.area(round) }
        round += 1
    }
    return total
}

#: The same shape returning a bool. The stub that finishes a deoptimised call
#: used to load its result eight bytes wide whatever the kind, and BOOL_VAL
#: writes only the union's one byte -- so a `false` returned into a slot that
#: last held a large int read back as `true`. Every class here writes a field,
#: which is what puts that stub on the path.
trait Flag {
    fn ask(self, n: int) -> bool
}

class Even: Flag {
    pub var seen: int

    fn init(self) { self.seen = 0 }

    pub fn ask(self, n: int) -> bool {
        self.seen += 1
        return n % 2 == 0
    }
}

class Never: Flag {
    pub var seen: int

    fn init(self) { self.seen = 0 }

    pub fn ask(self, n: int) -> bool {
        self.seen += 1
        return false
    }
}

class Third: Flag {
    pub var seen: int

    fn init(self) { self.seen = 0 }

    pub fn ask(self, n: int) -> bool {
        self.seen += 1
        return n % 3 == 0
    }
}

fn any_flag(flags: list[Flag], rounds: int) -> int {
    var seen = 0
    var round = 0
    while round < rounds {
        for flag in flags { if flag.ask(round) { seen += 1 } }
        round += 1
    }
    return seen
}

trait Real {
    fn scale(self, n: int) -> float
}

class Half: Real {
    fn init(self) { }

    pub fn scale(self, n: int) -> float { return n * 0.5 }
}

class Third2: Real {
    fn init(self) { }

    pub fn scale(self, n: int) -> float { return n * 0.25 }
}

fn sum_reals(reals: list[Real], rounds: int) -> float {
    var total = 0.0
    var round = 0
    while round < rounds {
        for real in reals { total = total + real.scale(round) }
        round += 1
    }
    return total
}

pub fn test_six_classes_through_one_site() -> void {
    let items = shapes(36)
    #: Computed the long way round, so the expectation does not come from the
    #: same dispatch it is checking.
    var expect = 0
    var round = 0
    while round < ROUNDS {
        var index = 0
        while index < 36 {
            let k = index % 5 + 1
            let m = index % 6
            if m == 0 {
                expect += round * k
            } elif m == 1 {
                expect += round + k
            } elif m == 2 {
                expect += round - k
            } elif m == 3 {
                expect += round * 2 + k
            } elif m == 4 {
                expect += round - k * 2
            } else {
                expect += round + k * 3
            }
            index += 1
        }
        round += 1
    }
    assert_eq(sum_ints(items, ROUNDS), expect)
}

pub fn test_a_bool_return_is_not_widened() -> void {
    var flags: list[Flag] = []
    for index in 0..30 {
        if index % 3 == 0 {
            flags.push(Even())
        } elif index % 3 == 1 {
            flags.push(Never())
        } else {
            flags.push(Third())
        }
    }
    var expect = 0
    var round = 0
    while round < ROUNDS {
        var index = 0
        while index < 30 {
            let m = index % 3
            if m == 0 and round % 2 == 0 {
                expect += 1
            } elif m == 2 and round % 3 == 0 {
                expect += 1
            }
            index += 1
        }
        round += 1
    }
    assert_eq(any_flag(flags, ROUNDS), expect)
}

pub fn test_a_float_return_lands_in_the_right_bank() -> void {
    var reals: list[Real] = []
    for index in 0..20 {
        if index % 2 == 0 { reals.push(Half()) } else { reals.push(Third2()) }
    }
    var expect = 0.0
    var round = 0
    while round < ROUNDS {
        expect = expect + float(round) * (10.0 * 0.5 + 10.0 * 0.25)
        round += 1
    }
    assert_eq(sum_reals(reals, ROUNDS), expect)
}

pub fn test_one_class_still_works() -> void {
    var only: list[Shape] = []
    for index in 0..12 { only.push(Sq(3)) }
    assert_eq(sum_ints(only, 100), 12 * 3 * (99 * 100 // 2))
    assert_true(sum_ints(only, 1) == 0)
}
