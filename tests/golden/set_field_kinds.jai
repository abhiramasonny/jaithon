#: What the OP_SET_FIELD kind guard must NOT reject. Every one of these is
#: legal today, and a guard that is too eager turns correct code into a
#: TypeError -- which is far worse than the missed check it replaces.
import std.io

class Holder {
    pub var n: int
    pub var f: float
    pub var s: str
    pub var loose: any
    pub var maybe: int?
    pub var items: list[int]

    fn init(self) -> void {
        self.n = 0
        self.f = 0.0
        self.s = ""
        self.loose = null
        self.maybe = null
        self.items = []
    }
}

class Box[T] {
    pub var item: T

    fn init(self, item: T) -> void { self.item = item }
}

let h = Holder()

# Direct, statically-checked writes.
h.n = 7
h.f = 2.5
h.s = "ok"

# An int widening into a float field is legal (LANGUAGE.md §2.5).
h.f = 3

# An `any` field takes anything.
h.loose = "anything"
h.loose = 42

# A nullable field takes null and its element type.
h.maybe = null
h.maybe = 9

# Correct types arriving through `any` must still pass.
fn poke(o: any, v: any) -> void { o.n = v }
poke(h, 11)

# A generic field takes whatever the instantiation holds.
let b = Box("held")
let c = Box(3)

fn poke_item(o: any, v: any) -> void { o.item = v }
poke_item(b, "replaced")
poke_item(c, 4)

print(h.n)
print(h.f)
print(h.s)
print(h.loose)
print(h.maybe)
print(h.items)
print(b.item)
print(c.item)
