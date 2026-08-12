#: A field write that arrives through an `any` receiver must be checked at run
#: time. The checker cannot see it: with an `any` receiver it does not know the
#: class, so it does not know the field's declared type, and it emits a bare
#: OP_SET_FIELD identical in shape to a legitimate one. Only FieldInfo, filled
#: from OP_FIELD_DEF's bits 4-7, knows what the field was declared as.
import std.io

class Point {
    pub var x: float

    fn init(self, x: float) -> void { self.x = x }
}

fn poison(p: any, v: any) -> void { p.x = v }

let p = Point(1.5)

try {
    poison(p, "not a float")
    print("NOT REACHED")
} catch e: TypeError {
    print(f"caught: {e.message}")
}

# The field must be unchanged: the guard rejects before it stores.
print(p.x)

# The same write with a correct value still succeeds through the same path.
poison(p, 2.5)
print(p.x)

# An int is accepted by a float field (LANGUAGE.md §2.5). Note it is stored as
# an int: the `any` path emits no OP_TO_FLOAT, because the checker never saw a
# float on the target side.
poison(p, 3)
print(p.x)

# The inline cache must be guarded too, not just the slow path. Every write
# above went through `poison`, so its cache is now warm and pointing straight
# at the slot -- a poisoned write here takes the fast path and never reaches
# jaiSetProperty. This is the case a guard on the slow path alone would miss.
try {
    poison(p, "still not a float")
    print("NOT REACHED (cached)")
} catch e: TypeError {
    print(f"caught cached: {e.message}")
}
print(p.x)
