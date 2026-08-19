#: A field-kind memo lets a read of a field this body just STORED skip its tag
#: guard: `self.n = self.n + 1; return self.n` reads back a kind the emitter
#: already knows, and without it that read's guard would be a bail after a heap
#: write, which the tier refuses -- so the memo is what makes the whole shape
#: compilable rather than declined. `Emit::known`, written by
#: `recordFieldStore` and read by `knownFieldKind`.
#:
#: Its comment used to justify itself with "nothing can have changed it (the
#: body can't call)". That was true when it was written and is not true now,
#: and the only invalidation it ever grew was another store to the same field
#: SLOT emitted by this same walk. Everything else that can make the claim
#: false was invisible to it. Four routes, every one of them confirmed as a
#: real miscompile before the fix -- the compiled tier disagreeing with
#: `JAITHON_NO_JIT=1` on the very next line:
#:
#:   * A CALL. The callee reaches the same object through the argument it was
#:     handed (`call_arg`), through a global the compiler cannot rule out
#:     (`call_global`), or as the receiver it was invoked on (`call_invoke`),
#:     and a store in there is one this walk never sees. `call_arg` returned
#:     40823337536 -- an ObjString pointer read as an int, because the memo
#:     still said SLOT_INT. `call_compound_int` is the same hole reached
#:     through the OTHER read site, `OP_GET_FIELD`, whose receiver arrives on
#:     the operand stack rather than in a local; a compound assignment is what
#:     emits it, and it returned 4609434218613702657, which is 1.5's bit
#:     pattern plus one. Retired now in `noteScratchClobber`, the one point
#:     every call out of the body passes through.
#:
#:     `call_compound_int` DECLINES to compile now, which is the fix working
#:     and not a hole in it: the read emits its guard, gets SLOT_FLOAT, and
#:     `float + int` is a mixed add this tier will not take. `call_compound`
#:     is its twin with a `1.0` on the right, and exists so the `OP_GET_FIELD`
#:     site is still compiled and still right after the fix rather than merely
#:     refused. Both print 2.5.
#:
#:   * A BRANCH. The store may sit on the arm the branch skipped
#:     (`branch_skipped`), or a back edge may re-enter above a store further
#:     down the body, so the read at the top meets on its second iteration the
#:     kind the bottom left behind (`back_edge`). Neither needs a call at all.
#:     Retired now on the same `offsetIsBranchTarget` test the ASCII-table
#:     proofs use, and for the same reason: the linear walk models one edge.
#:
#:   * A WRITE TO THE LOCAL the entry names (`local_rebound`). An entry is
#:     keyed on a slot number, and the slot can be made to hold a different
#:     object entirely. Retired now in `noteSlotWrite`, which the emitter
#:     already documents as the one place every local write passes through.
#:
#: The consequence is a wrong answer wherever the two payloads are readable as
#: each other, and a SIGSEGV where they are not: memoise SLOT_OBJ, let the
#: callee leave a small integer in the field, and the next `len()` dereferences
#: 12345 as an `Obj`. That shape killed the process with exit 139 against an
#: ordinary AttributeError from the interpreter. `obj_then_int` is it, kept
#: here because it compiles and exercises a SLOT_OBJ memo across a call --
#: though in this file's register allocation it happens to reach a guard and
#: deopt rather than fault, so the numbers above it are what fails loudly.
#:
#: Each shape gets a Box of its own on purpose. Sharing one made the field's
#: sampled type flap between iterations, and several of these bodies then
#: declined to compile for unrelated reasons -- which is a silent loss of
#: coverage, since a declined body agrees with the interpreter trivially.
#:
#: `bump` is the case that must keep working: it is the shape the memo exists
#: for. A fix that simply deleted the memo would leave every number here
#: correct and quietly stop compiling it, so this one is checked by eye with
#: `JAI_JIT_WHY=1 ./jaithon run tests/golden/jit_field_kind_memo.jai`, which
#: must still say `compiled bump`.
#:
#: The loop count only has to clear JAI_JIT_THRESHOLD (64 entries) so the
#: function tier compiles and the compiled body then runs; nothing here
#: allocates per iteration, so gc stress stays cheap.
class Box {
    pub var v: any

    fn init(self, v: any) -> void { self.v = v }

    pub fn retype(self) -> int {
        self.v = "invoked"
        return 1
    }
}

var shared: any = null

fn store_str(b: any) -> int {
    b.v = "argument"
    return 1
}

fn poke_shared() -> int {
    shared.v = "global"
    return 1
}

fn store_float(b: any) -> int {
    b.v = 1.5
    return 1
}

fn store_flip(b: any, flip: int) -> int {
    if flip == 1 {
        b.v = 12345
    } else {
        b.v = "12345678"
    }
    return 1
}

fn call_arg(b: any) -> any {
    b.v = 7
    let t = store_str(b)
    let x = b.v
    return x
}

fn call_global(b: any) -> any {
    b.v = 7
    let t = poke_shared()
    let x = b.v
    return x
}

fn call_invoke(b: any) -> any {
    b.v = 7
    let t = b.retype()
    let x = b.v
    return x
}

fn call_compound(b: any) -> any {
    b.v = 7
    let t = store_float(b)
    b.v += 1.0
    return b.v
}

fn call_compound_int(b: any) -> any {
    b.v = 7
    let t = store_float(b)
    b.v += 1
    return b.v
}

fn branch_skipped(b: any, flag: int) -> any {
    if flag == 1 {
        b.v = 7
    }
    let x = b.v
    return x
}

fn back_edge(b: any, n: int) -> any {
    b.v = 7
    var i = 0
    var x: any = null
    while i < n {
        x = b.v
        b.v = 1.5
        i = i + 1
    }
    return x
}

fn local_rebound(b: any, c: any) -> any {
    b.v = 7
    b = c
    let x = b.v
    return x
}

fn obj_then_int(b: any, flip: int) -> int {
    b.v = "abcdefgh"
    let t = store_flip(b, flip)
    let x = b.v
    return x.len()
}

fn bump(c: any) -> int {
    c.v = c.v + 1
    return c.v
}

let arg_box = Box(0)
let global_box = Box(0)
let invoke_box = Box(0)
let compound_box = Box(0)
let compound_int_box = Box(0)
let branch_box = Box("skipped")
let edge_box = Box(0)
let rebound_box = Box(0)
let other_box = Box("other")
let flip_box = Box("")
let counter = Box(0)
shared = global_box

var a1: any = null
var a2: any = null
var a3: any = null
var a4: any = null
var a4b: any = null
var a5: any = null
var a6: any = null
var a7: any = null
var a8 = 0
var a9 = 0

var k = 0
while k < 300 {
    a1 = call_arg(arg_box)
    a2 = call_global(global_box)
    a3 = call_invoke(invoke_box)
    a4 = call_compound(compound_box)
    a4b = call_compound_int(compound_int_box)
    branch_box.v = "skipped"
    a5 = branch_skipped(branch_box, 0)
    a6 = back_edge(edge_box, 3)
    a7 = local_rebound(rebound_box, other_box)
    a8 = obj_then_int(flip_box, 0)
    a9 = bump(counter)
    k = k + 1
}

print(a1)
print(a2)
print(a3)
print(a4)
print(a4b)
print(a5)
print(a6)
print(a7)
print(a8)
print(a9)
print(branch_skipped(branch_box, 1))

var caught = 0
try {
    print(obj_then_int(flip_box, 1))
} catch _e {
    caught = 1
}
print(caught)
