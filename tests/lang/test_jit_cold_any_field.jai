#: A read of an `any` field on a branch the body has never taken.
#:
#: The field read arm specialises an `any` field to the kind it finds on the
#: live receiver. With no live receiver to look at -- the measuring walk never
#: reached the read -- and no declared kind to fall back on, it used to decline
#: the WHOLE function. The parser's `_parse_primary` was that shape: its float
#: literal arm reads `token.value ?? 0.0`, a clean parse of the compiler meets
#: almost no floats, and that one cold read kept the dispatch -- and through it
#: every rung of the expression ladder above it -- interpreted. Off the hot
#: path the read is now left to the interpreter (JAITHON_JIT_SOFT_ANY_FIELD).
#:
#: `classify_cell` below is that shape. It is called far past the threshold
#: with ints only, then the float, string and null arms run on the compiled
#: form, which has to hand each one back to the interpreter with the right
#: answer.
#:
# jit-compiles: classify_cell
from std.test import assert_eq

class Cell {
    pub let tag: int
    pub var value: any

    pub fn init(self, tag: int, value: any) {
        self.tag = tag
        self.value = value
    }
}

#: Hands the cell out through a method, so `classify_cell` holds it in a LOCAL
#: bound from a call: a parameter, or a field read off one, is live when the
#: body compiles and the arm samples it; a call's result is not.
class Slot {
    pub let cell: Cell

    pub fn init(self, cell: Cell) {
        self.cell = cell
    }

    pub fn get(self) -> Cell {
        return self.cell
    }
}

fn classify_cell(slot: Slot) -> float {
    let cell = slot.get()
    let tag = cell.tag
    if tag == 0 { return 1.0 }
    if tag == 1 {
        let number = cell.value ?? 0.0
        return number + 0.5
    }
    if tag == 2 { return 2.0 }
    return -1.0
}

fn test_cold_any_field_reads_answer_after_compiling() {
    var total = 0.0
    for at in 0..400 { total += classify_cell(Slot(Cell(at % 3 == 1 ? 2 : 0, at))) }
    assert_eq(total, 533.0)
    assert_eq(classify_cell(Slot(Cell(1, 2.25))), 2.75)
    assert_eq(classify_cell(Slot(Cell(1, null))), 0.5)
    assert_eq(classify_cell(Slot(Cell(1, 4.0))), 4.5)
    assert_eq(classify_cell(Slot(Cell(9, "x"))), -1.0)
    for at in 0..100 { assert_eq(classify_cell(Slot(Cell(1, 1.0 * at))), 1.0 * at + 0.5) }
}
