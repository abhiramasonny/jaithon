#: A bool field read from inside a compiled body.
#:
#: `BOOL_VAL` writes only the one-byte member of the Value union, so a bool
#: field has to be loaded with `ldrb`. Reading it eight bytes wide picks up
#: whatever the field held before and comes back true from garbage -- the same
#: mistake, made once on locals, cost a day of debugging the front end.
from std.test import assert_eq, assert_true

class Flagged {
    pub var live: bool
    pub var dead: bool
    pub var count: int

    pub fn init(self, live: bool, dead: bool, count: int) {
        self.live = live
        self.dead = dead
        self.count = count
    }

    pub fn live_count(self) -> int {
        if self.live { return self.count }
        return 0
    }

    pub fn both(self) -> bool { return self.live and not self.dead }
}

fn test_a_bool_field_reads_the_same_hot_as_cold() -> void {
    var cells: list[Flagged] = []
    for index in 0..64 {
        #: The int beside the bool is what makes an eight-byte read visible:
        #: it leaves a nonzero pattern in the bytes above the bool.
        cells.push(Flagged(index % 3 != 0, index % 5 == 0, index * 1000 + 7))
    }
    var cold = 0
    for cell in cells { cold += cell.live_count() }

    #: Warm past the compile threshold, then check every element again.
    var hot = 0
    for _round in 0..2000 {
        hot = 0
        for cell in cells { hot += cell.live_count() }
    }
    assert_eq(hot, cold, "a compiled bool field read disagreed with a cold one")

    var trues = 0
    for _round in 0..2000 {
        trues = 0
        for cell in cells {
            if cell.both() { trues += 1 }
        }
    }
    var expected = 0
    for index in 0..64 {
        if index % 3 != 0 and not index % 5 == 0 { expected += 1 }
    }
    assert_eq(trues, expected, "a compiled `and` over two bool fields is wrong")
}

fn test_a_bool_field_that_changes_under_a_compiled_loop() -> void {
    let cell = Flagged(true, false, 42)
    var total = 0
    for round in 0..4000 {
        if round == 3000 { cell.live = false }
        total += cell.live_count()
    }
    assert_eq(total, 3000 * 42, "flipping the field mid-loop was not seen")
}
