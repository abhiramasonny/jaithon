#: A deopt taken before a local has been assigned.
#:
#: The prologue starts every local past the arity at a bare zero, and a register
#: home carries no tag -- so what a local "holds" before its first assignment is
#: a zero payload and nothing else. The deopt stub writes every local out and
#: rebuilds each tag from the compile-time kind, which for an instance, a list
#: or any other object kind was an unconditional VAL_OBJ. Pairing that tag with
#: the prologue's zero handed the interpreter an object at address zero.
#:
#: Nothing in the program itself notices -- the checker guarantees definite
#: assignment before a read, and the collector null-checks -- so it survives all
#: the way to the next back-edge, where jaiJitEnterOsr's slot scan walks every
#: frame slot and IS_INSTANCE loads `Obj.type` off address 0. That is a hard
#: crash with no output at all, not a wrong answer.
#:
#: Every function below binds an object-kinded local inside a branch that never
#: runs, then keeps looping, so the sampler tries an OSR entry on the frame a
#: deopt rebuilt. Reaching the bad path needs the guard to fail BEFORE the
#: binding, which is what JAITHON_JIT_DEOPT_STRESS=1 makes certain of --
#: scripts/run_tests.sh runs every golden under it. Without the switch this is
#: an ordinary run of the same arithmetic, and the answers have to match.
class P {
    pub var x: int
    pub var y: int

    pub fn init(self, x: int = 0, y: int = 0) {
        self.x = x
        self.y = y
    }
}

#: SLOT_INST. `inner` is bound inside a while inside an `if` the accumulator
#: never satisfies, so the slot is still zero when the first guard fires.
fn walk_inst(pts: list[P], n: int) -> int {
    let head = pts[0]
    var acc = head.x
    var i = 1
    while i < n {
        let outer = pts[i]
        acc += outer.x
        if acc < 0 {
            var j = 0
            while j < i {
                let inner = pts[j]
                acc += inner.y
                j += 1
            }
        }
        i += 1
    }
    return acc
}

#: SLOT_LIST beside SLOT_OBJ. The scan's list arm derefs exactly as its instance
#: arm does; its object arm checks the tag alone, so a zero payload passes entry
#: and is loaded from inside the compiled loop instead.
fn walk_list(rows: list[list[int]], n: int) -> int {
    var acc = 0
    var i = 0
    while i < n {
        let row = rows[i]
        acc += row[0] + row[1]
        if acc < 0 {
            let spare = rows[0]
            let label = "unreachable"
            acc += spare[1] + label.len()
        }
        i += 1
    }
    return acc
}

#: The same slot left unassigned in a body whose OSR form pins a class, so the
#: shape loop that follows the kind scan reads it too. `first` is assigned; the
#: two below it never are.
fn walk_shape(pts: list[P], n: int) -> int {
    var acc = 0
    var i = 0
    while i < n {
        let first = pts[i]
        acc += first.x - first.y
        if first.x > 1000000 {
            let second = pts[0]
            let third = pts[1]
            acc += second.y + third.y
        }
        i += 1
    }
    return acc
}

fn main() -> int {
    var pts: list[P] = []
    for i in 0..64 { pts.push(P(i, i * 3)) }
    var rows: list[list[int]] = []
    for i in 0..64 { rows.push([i, i * 7]) }

    var a = 0
    var b = 0
    var c = 0
    for _r in 0..4000 {
        a += walk_inst(pts, 64)
        b += walk_list(rows, 64)
        c += walk_shape(pts, 64)
    }
    print(a)
    print(b)
    print(c)
    return 0
}
