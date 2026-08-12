#: The declared kind of a field rides in bits 4-7 of OP_FIELD_DEF's info byte.
#: tests/vm/field_kind_disasm.sh checks the encoding itself; this file exists so
#: the shapes it names are real, and so every branch of the mapping is compiled
#: at every -O level and under gc/deopt stress.
#:
#: An `any` field, a nullable one, and a bare generic parameter all stay
#: FIELD_KIND_ANY -- a nullable holds null, and a generic parameter's type is
#: not known here. That is also what every image written before this encoding
#: existed decodes to, since its bits 4-7 are zero. (Every field must carry an
#: annotation, E0105, so there is no unannotated case to encode.)
import std.io

class Sample {
    pub var count: int
    pub var ratio: float
    pub var name: str
    pub var flag: bool
    pub var items: list[int]
    pub var table: dict[str, int]
    pub var maybe: int?
    pub var loose: any

    fn init(self) -> void {
        self.count = 0
        self.ratio = 0.0
        self.name = ""
        self.flag = false
        self.items = []
        self.table = {}
        self.maybe = null
        self.loose = null
    }
}

class Box[T] {
    pub var item: T
    pub var many: list[T]

    fn init(self, item: T) -> void {
        self.item = item
        self.many = []
    }
}

let s = Sample()
print(s.count)
print(s.ratio)
print(s.name == "")
print(s.flag)
print(s.items)
print(s.table)
print(s.maybe)
print(s.loose)

let b = Box("held")
print(b.item)
print(b.many)
