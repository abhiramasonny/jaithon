#: What a guard hands back to the interpreter when it fails.
#:
#: The record says where to resume and what the interpreter is holding there,
#: and "there" is the START of an instruction. The compiler's model is not that
#: state: `OP_GET_LOCAL` pushes its operand and only then guards the operand's
#: tag, so by the time a guard is written the model can be one or two entries
#: deeper than the interpreter's stack is at that offset. Handing the surplus
#: over as well leaves it stranded underneath everything the interpreter pushes
#: for itself, and the next `for` finds the loop variable where its iterator
#: should be:
#:
#:     TypeError: for-loop expected an iterator, not 'int'
#:
#: Every guard below is on a dynamic local -- a slot the compiler reuses for the
#: induction variables of loops that do not overlap, so it holds an int on one
#: path and an instance on another, lives in the frame with its tag, and is read
#: behind a check of that tag. Those are the reads whose guard sits after the
#: push, and each one here has a live iterator beneath it.
#:
#: A dynamic local's tag check never fires in a well-typed program, so this
#: reaches the resume path only under JAITHON_JIT_DEOPT_STRESS=1, where every
#: guard fails at once. The golden layer runs every case both ways.

# --- something for the slot to hold besides an int --------------------------
class Cell {
    pub var v: int

    fn init(self, v: int) { self.v = v }
}

# --- one range loop, one list loop, one slot --------------------------------
fn walk(cells: list[Cell], n: int) -> int {
    var total = 0
    for i in 0..n { total = total + i }
    for c in cells { total = total + c.v }
    return total
}

# --- nbody's `advance`, in miniature ----------------------------------------
fn pairs(cells: list[Cell], n: int) -> int {
    var total = 0
    for i in 0..n {
        let a = cells[i]
        for j in i + 1..n {
            let b = cells[j]
            total = total + a.v * b.v
        }
    }
    for c in cells { total = total + c.v * 2 }
    return total
}

fn spin(cells: list[Cell], rounds: int) -> int {
    var acc = 0
    for _r in 0..rounds {
        acc = acc + walk(cells, 5)
        acc = acc + pairs(cells, cells.len())
    }
    return acc
}

let cells = [Cell(1), Cell(2), Cell(3), Cell(4)]
print(walk(cells, 5))
print(pairs(cells, 4))
print(spin(cells, 3000))
