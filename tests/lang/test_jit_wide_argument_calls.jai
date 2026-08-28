#: Methods and constructors with more arguments than one call out of compiled
#: code used to carry.
#:
#: `JIT_MAX_ARGS_OUT` bounds a descriptor's argument array, and an invoke spends
#: one of those on the receiver -- so at 4 the tier refused every method of four
#: or more arguments, and refused it for the WHOLE enclosing body. The
#: self-hosted compiler checking `parser.jai` hit
#: "OP_INVOKE: 5 arguments, past the cap of 3" eighty times at a single osr
#: loop, and asked for six and seven elsewhere.
#:
#: A loop calling one five-argument method 4,000,000 times ran 52,000,846
#: interpreted instructions at a cap of 4 and 1,689 at a cap of 8 -- 1.57x on
#: wall clock (0.20s to 0.127s median of five).
# jit-compiles: drive_five, drive_seven, build_wide
from std.test import assert_eq

let CALLS = 100

class Acc {
    var total: int

    fn init(self) -> void { self.total = 0 }

    pub fn add5(self, a: int, b: int, c: int, d: int, e: int) -> int {
        return a + b + c + d + e
    }

    pub fn add7(self, a: int, b: int, c: int, d: int, e: int, f: int, g: int) -> int {
        return a + b + c + d + e + f + g
    }
}

#: A constructor storing more fields than the cap: `simpleInitFields` writes
#: them straight into the instance and is bounded by the same number.
class Wide {
    var a: int
    var b: int
    var c: int
    var d: int
    var e: int

    fn init(self, a: int, b: int, c: int, d: int, e: int) -> void {
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
    }

    pub fn sum(self) -> int { return self.a + self.b + self.c + self.d + self.e }
}

fn drive_five(rounds: int) -> int {
    let acc = Acc()
    var sum = 0
    var i = 0
    while i < rounds {
        sum += acc.add5(i, i + 1, i + 2, i + 3, i + 4)
        i += 1
    }
    return sum
}

fn drive_seven(rounds: int) -> int {
    let acc = Acc()
    var sum = 0
    var i = 0
    while i < rounds {
        sum += acc.add7(i, 1, 2, 3, 4, 5, 6)
        i += 1
    }
    return sum
}

fn build_wide(rounds: int) -> int {
    var sum = 0
    var i = 0
    while i < rounds {
        sum += Wide(i, 1, 2, 3, 4).sum()
        i += 1
    }
    return sum
}

pub fn test_five_argument_methods() -> void {
    var total = 0
    var call = 0
    while call < CALLS {
        #: Ten iterations of 5i + 10, i in 0..9: 5*45 + 100.
        total += drive_five(10)
        call += 1
    }
    assert_eq(total, 325 * CALLS)
}

pub fn test_seven_argument_methods() -> void {
    var total = 0
    var call = 0
    while call < CALLS {
        #: i + 21, i in 0..9.
        total += drive_seven(10)
        call += 1
    }
    assert_eq(total, (45 + 210) * CALLS)
}

pub fn test_five_field_constructors() -> void {
    var total = 0
    var call = 0
    while call < CALLS {
        #: i + 10, i in 0..9.
        total += build_wide(10)
        call += 1
    }
    assert_eq(total, (45 + 100) * CALLS)
}
