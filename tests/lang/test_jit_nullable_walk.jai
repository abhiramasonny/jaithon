#: The loop shape every list and tree walk has: a local seeded from a live
#: instance, then reassigned from a `-> T?` call.
#:
#: Three refusals used to stack up on it, each hiding the next. The local took
#: SLOT_INST from `var at = head` and the assignment wanted SLOT_MAYBE_INST, so
#: the loop tier declined -- and unlike the function tier it had no retry that
#: could seed the slot wider. Widening it then moved the refusal to
#: `at.follow()`, whose receiver was a maybe-instance where the invoke arm
#: wanted an instance. And the callee's own declared return kind,
#: SLOT_MAYBE_INST, was not on the list of kinds a call result may have.
#:
#: What these check is the ANSWER, not the speed: the guards the change adds
#: are a null test at the invoke and an object-or-null test on the call result,
#: and both have to be right when they fire, not only when they do not. Run
#: this file with and without JAITHON_NO_JIT, and under --gc-stress.
# jit-compiles: walk_the_chain
from std.test import assert_eq, assert_true

class Link {
    pub var v: int
    pub var next: Link?

    pub fn init(self, v: int, next: Link?) -> void {
        self.v = v
        self.next = next
    }

    #: The declared return is `Link?`, which is the kind under test.
    pub fn follow(self) -> Link? {
        return self.next
    }

    pub fn value(self) -> int {
        return self.v
    }
}

#: A second class reaching the same compiled sites, so the class guard has
#: something to fail on rather than being right by never being tested.
class Tag {
    pub var v: int
    pub var next: Tag?

    pub fn init(self, v: int, next: Tag?) -> void {
        self.v = v
        self.next = next
    }

    pub fn follow(self) -> Tag? {
        return self.next
    }

    pub fn value(self) -> int {
        return self.v
    }
}

fn build(n: int) -> Link {
    var head = Link(0, null)
    var i = 1
    while i < n {
        head = Link(i, head)
        i += 1
    }
    return head
}

fn build_tags(n: int) -> Tag {
    var head = Tag(0, null)
    var i = 1
    while i < n {
        head = Tag(i * 2, head)
        i += 1
    }
    return head
}

fn walk_the_chain(head: Link, rounds: int) -> int {
    var total = 0
    var r = 0
    while r < rounds {
        var at: Link? = head
        while at is not null {
            total += at.value()
            at = at.follow()
        }
        r += 1
    }
    return total
}

fn walk_the_tags(head: Tag, rounds: int) -> int {
    var total = 0
    var r = 0
    while r < rounds {
        var at: Tag? = head
        while at is not null {
            total += at.value()
            at = at.follow()
        }
        r += 1
    }
    return total
}

#: 100 links summing 0..99 is 4950 per round.
fn test_a_nullable_walk_sums_the_same_hot_as_cold() -> void {
    let head = build(100)
    assert_eq(walk_the_chain(head, 1), 4950)
    assert_eq(walk_the_chain(head, 4000), 4950 * 4000)
    assert_eq(walk_the_chain(head, 1), 4950)
}

fn test_a_second_class_on_the_same_sites_is_not_confused_for_the_first() -> void {
    let links = build(100)
    let tags = build_tags(100)
    assert_eq(walk_the_chain(links, 2000), 4950 * 2000)
    assert_eq(walk_the_tags(tags, 2000), 9900 * 2000)
    assert_eq(walk_the_chain(links, 2000), 4950 * 2000)
}

#: The null really does reach the receiver here, so the invoke's guard has to
#: raise rather than dereference zero.
fn value_without_checking(items: list[Link?]) -> int {
    var total = 0
    var i = 0
    while i < items.len() {
        var at: Link? = items[i]
        total += at.value()
        i += 1
    }
    return total
}

fn test_a_null_receiver_raises_rather_than_crashing() -> void {
    var items: list[Link?] = []
    var i = 0
    while i < 400 {
        if i % 5 == 0 { items.push(null) } else { items.push(Link(i, null)) }
        i += 1
    }
    var raised = false
    try {
        value_without_checking(items)
    } catch e: Error {
        raised = true
    }
    assert_true(raised, "a null receiver did not raise")
}
