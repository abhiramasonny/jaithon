#: gc-stress-every: 50
#: A dynamic local read as a MAYBE-instance must confirm its class shape too,
#: not just its VAL_OBJ tag.
#:
#: `jit_dynamic_local_inst.jai` covers the same hole for `SLOT_INST`, and
#: `localIn` grew an arm for that kind and for `SLOT_LIST`. It never grew one
#: for `SLOT_MAYBE_INST`, whose tag is the same VAL_OBJ (see `localTagFor`),
#: so a dynamic local whose kind at the read was "an instance of Dog, or
#: null" was handed back with nothing checked at all -- and a `Bird` a
#: sibling write had left in the slot sailed through and was read at Dog's
#: field offsets.
#:
#: Two routes reach that kind, and both are here because they are set on
#: different passes and a fix for one need not cover the other:
#:
#: `plain` is the ordinary one. `adoptLocalKindSeen` gives a dynamic slot
#: whatever kind the current write has, so binding a nullable instance FIELD
#: (`p.link`, which is a Dog or null) to a slot that a later write also makes
#: dynamic is enough. One retry, no recursion.
#:
#: `nullable` is the parameter route. `jaiJitCompileFunc` retries up to four
#: times carrying two persistent per-slot flags. The merge step refuses to set
#: `nullable[i]` while `dynamic[i]` is already true, but never clears a
#: `nullable[i]` set on an EARLIER attempt once `dynamic[i]` becomes true on a
#: later one -- so a parameter flagged nullable by the self-call check on
#: attempt 1 and flagged dynamic by an ordinary write on attempt 2 arrives at
#: the successful attempt with both set, which `seedLocals` turns into exactly
#: the unguarded combination. (The trailing `a = c` is what keeps the entry
#: contract accepting a Dog: `jitParamKind` is read off the walk's FINAL kind.)
#:
#: Confirmed as a real miscompile before the fix: both printed 4027584 against
#: the correct 160000, reading `Bird.wings` (99) where `Dog.legs` (4) lives.
#:
#: `checked` is a third, independent route, for a second bug in the same
#: family: the function-entry prologue's frame-tag write for a spilled
#: `SLOT_MAYBE_INST` parameter used to be a compile-time CONSTANT tag
#: (`localTagFor`, always VAL_OBJ for this kind) rather than reading the
#: actual argument's payload. A literal `null` argument therefore arrived
#: in the frame as `{tag: VAL_OBJ, payload: 0}` -- an object tag over a
#: null pointer, which is not `NULL_VAL`'s own encoding. `localIn`'s new
#: null-check-then-deopt catches the payload being zero and bails, but the
#: deopt record is built by copying the frame's tag+payload bytes
#: verbatim, so the interpreter it resumes into inherited the same
#: malformed value: `OBJ_VAL(NULL)` instead of a proper null. `a == null`
#: on that value is false, and the next line dereferences a null pointer.
#: `checked(null, ...)` is called directly (not through the self-call that
#: exercises `plain`/`nullable`) specifically to go through the prologue's
#: own frame-seeding write for an external argument, not a self-call's.
class Dog {
    pub var legs: int = 4
    pub var link: any = null
}

class Bird {
    pub var wings: int = 99
    pub var legs: int = 2
}

fn plain(p: any, other: any, n: int) -> int {
    var a = p.link
    var t = 0
    var i = 0
    while i < n {
        t = a.legs + t
        a = other
        i = i + 1
    }
    return t
}

fn nullable(a: any, n: int, other: any) -> int {
    var c = a
    var t = 0
    var i = 0
    while i < n {
        t = a.legs + t
        if i > 100 {
            var q = a.link
            t = t + nullable(q, 0, other)
        }
        a = other
        i = i + 1
    }
    a = c
    return t
}

fn checked(a: any, n: int, other: any) -> int {
    var c = a
    var t = 0
    var i = 0
    while i < n {
        if a == null { return -1 }
        t = a.legs + t
        if i > 100 {
            var q = a.link
            t = t + checked(q, 0, other)
        }
        a = other
        i = i + 1
    }
    a = c
    if a == null { return -2 }
    return t
}

let dog = Dog()
dog.link = Dog()
let bird = Bird()

var one = 0
var two = 0
var three = 0
var k = 0
while k < 20000 {
    one = one + plain(dog, bird, 3)
    two = two + nullable(dog, 3, bird)
    three = three + checked(dog, 3, bird)
    k = k + 1
}
print(one)
print(two)
print(three)
print(checked(null, 3, bird))
