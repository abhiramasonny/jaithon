#: gc-stress-every: 50
#: `OP_GET_INDEX`'s `SLOT_INST` list-element arm must confirm `Obj.type ==
#: OBJ_INSTANCE` before reading `klass` -- the same check its `SLOT_LIST`
#: neighbour ten lines below it already has, and that
#: `OP_FOR_ITER_BIND`'s own `SLOT_INST` arms already have.
#:
#: The consequence of skipping it is worse than a wrong answer: a list
#: sampled as holding instances, later indexed while holding some other raw
#: object (a string, here), reads that object's header bytes as an
#: `ObjInstance`'s `klass` pointer and dereferences it -- `ObjString`'s
#: layout at that offset is not a pointer, so this segfaults rather than
#: silently misreading. Confirmed before this fix: the JIT-enabled run below
#: crashed (`SIGSEGV`, exit 139); the interpreter always ran it correctly.
class Dog {
    pub var v: int = 99
}

fn touch(xs: list, n: int) -> any {
    var acc = 0
    for i in 0..n {
        let e = xs[i]
        acc += 1
    }
    return acc
}

fn warm(xs: list) -> void {
    for _r in 0..8000 { touch(xs, 4) }
}

var xs: list = []
for i in 0..4 { xs.push(Dog()) }
warm(xs)
print(touch(xs, 4))
xs.push("boom")
print(touch(xs, 5))
