#: gc-stress-every: 50
#: This golden loops until the tier compiles, so bare --gc-stress collected on
#: every allocation of the SAME loop body and cost 92,775ms -- half this whole
#: suite, for one case. At N=50 it is 1,728ms and still collects 20,730 times,
#: which is far denser than the N=5000 the unit suites run at.
#: A field of object type must not decline the enclosing function, and must not
#: be read as the wrong type once admitted.
#:
#: `OP_GET_FIELD_LOCAL` and `OP_GET_FIELD` used to classify the observed field
#: value into int / float / maybe-instance and `return false` for everything
#: else -- and a `return false` from an opcode arm declines the whole function,
#: so a method reading a `str`, `list`, `dict` or `set` field could never be
#: compiled. `OP_SET_FIELD` refused to store one for the same reason. They go
#: through SLOT_OBJ now (a list through SLOT_LIST, which costs an OBJ_LIST
#: check because every SLOT_LIST consumer reads the header with no check of its
#: own).
#:
#: Four things have to hold, and the last three are invisible without the JIT:
#:   * a str field indexed per iteration reads the right bytes;
#:   * a list field iterated, indexed and pushed to sees the right elements;
#:   * a field whose object TYPE changes under an already-compiled body
#:     deoptimises rather than reading a string's header as a list's;
#:   * a SLOT_OBJ local live across a deopt is written back as itself and not
#:     as null -- the bug recorded at jit_func.c's deopt writer, which only
#:     becomes reachable once a body holding one can compile at all.
#:
#: The loop counts are what make the bodies hot enough to reach the compiled
#: tier; nothing here allocates per iteration, so --gc-stress stays cheap.
class Doc {
    pub var text: str
    pub var words: list[int]
    pub var tags: dict[str, int]
    pub var label: str

    fn init(self, text: str) -> void {
        self.text = text
        self.words = []
        self.tags = {}
        self.label = "unset"
    }

    pub fn spaces(self, n: int) -> int {
        var c = 0
        var i = 0
        while i < n {
            if self.text[i % 8] == " " { c = c + 1 }
            i = i + 1
        }
        return c
    }

    pub fn spaces_via_local(self, n: int) -> int {
        let s = self.text
        var c = 0
        var i = 0
        while i < n {
            if s[i % 8] == " " { c = c + 1 }
            i = i + 1
        }
        return c
    }

    pub fn fill(self, n: int) -> void {
        var i = 0
        while i < n {
            self.words.push(i % 7)
            i = i + 1
        }
    }

    pub fn total(self) -> int {
        var t = 0
        for w in self.words { t = t + w }
        return t
    }

    pub fn stride(self, n: int) -> int {
        var t = 0
        var i = 0
        while i < n {
            t = t + self.words[i]
            i = i + 1
        }
        return t
    }

    pub fn relabel(self, n: int) -> int {
        var i = 0
        while i < n {
            self.label = self.text
            i = i + 1
        }
        return self.label.len()
    }
}

class Loose {
    pub var v: any

    fn init(self, v: any) -> void { self.v = v }
}

fn measure(o: Loose, n: int) -> int {
    var t = 0
    var i = 0
    while i < n {
        t = t + o.v.len()
        i = i + 1
    }
    return t // n
}

fn walk_it(o: Loose, n: int) -> int {
    var t = 0
    var i = 0
    while i < n {
        for _x in o.v { t = t + 1 }
        i = i + 1
    }
    return t // n
}

let d = Doc("ab cd ef")
print(d.spaces(200000) / 200000)
print(d.spaces_via_local(200000) / 200000)
d.fill(200000)
print(d.total())
print(d.stride(200000))
print(d.relabel(200000))
print(d.text)
print(d.label)
print(d.words[0])
print(d.words[199999])
print(d.tags)

let a = Loose("abcd")
let b = Loose([1, 2, 3])
print(measure(a, 200000))
print(measure(b, 200000))
print(measure(a, 200000))

let c = Loose([7, 8, 9, 10])
print(walk_it(c, 100000))
print(walk_it(a, 100000))
print(walk_it(c, 100000))
