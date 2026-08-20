#: `is null` and `is not null` from inside a compiled body.
#:
#: The operator had no arm at all, so it fell to the unarmed-opcode deopt and
#: a body containing one was slower compiled than interpreted. These check the
#: answers rather than the speed: every kind the arm decides statically, and
#: the nullable-instance case it decides at run time.
from std.test import assert_eq, assert_true

class Link {
    pub var next: Link?
    pub var tag: int

    pub fn init(self, next: Link?, tag: int) {
        self.next = next
        self.tag = tag
    }
}

fn count_null(items: list[Link], rounds: int) -> int {
    var nulls = 0
    for _round in 0..rounds {
        nulls = 0
        for item in items {
            if item.next is null { nulls += 1 }
        }
    }
    return nulls
}

fn test_a_nullable_field_is_tested_the_same_hot_as_cold() -> void {
    let tail = Link(null, 1)
    var items: list[Link] = []
    for index in 0..64 { items.push(Link(if index % 3 == 0 { null } else { tail }, index)) }

    let cold = count_null(items, 1)
    let hot = count_null(items, 3000)
    assert_eq(hot, cold, "a compiled `is null` disagreed with a cold one")

    var expected = 0
    for index in 0..64 {
        if index % 3 == 0 { expected += 1 }
    }
    assert_eq(hot, expected, "the count itself is wrong")
}

fn test_is_not_null_is_the_complement() -> void {
    let tail = Link(null, 1)
    var items: list[Link] = []
    for index in 0..64 { items.push(Link(if index % 4 == 0 { null } else { tail }, index)) }
    var both = 0
    for _round in 0..3000 {
        both = 0
        for item in items {
            if item.next is null { both += 1 }
            if item.next is not null { both += 1 }
        }
    }
    assert_eq(both, items.len(), "exactly one of the two has to hold for each")
}

fn test_a_field_that_turns_null_under_a_compiled_loop() -> void {
    let node = Link(Link(null, 2), 1)
    var seen = 0
    for round in 0..4000 {
        if round == 2500 { node.next = null }
        if node.next is null { seen += 1 }
    }
    assert_eq(seen, 1500, "the field turning null mid-loop was not seen")
}

fn test_kinds_that_can_never_be_null() -> void {
    #: The arm answers these without emitting a compare, so they are the ones
    #: a wrong constant would silently invert.
    var ints = 0
    var floats = 0
    var strings = 0
    var lists = 0
    for _round in 0..3000 {
        ints = 0
        floats = 0
        strings = 0
        lists = 0
        for index in 0..8 {
            let i = index
            let f = float(index)
            let s = "x"
            let l = [index]
            if i is not null { ints += 1 }
            if f is not null { floats += 1 }
            if s is not null { strings += 1 }
            if l is not null { lists += 1 }
        }
    }
    assert_eq(ints, 8, "an int is never null")
    assert_eq(floats, 8, "a float is never null")
    assert_eq(strings, 8, "a str is never null")
    assert_eq(lists, 8, "a list is never null")
}
