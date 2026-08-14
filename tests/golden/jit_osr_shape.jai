#: A compiled loop must be re-entered only with the CLASS it was compiled for.
#:
#: A field read compiles to a load at a baked slot index. Two classes declaring
#: the same field name put it at different indices whenever their declaration
#: order differs, and the OSR entry guard used to check only IS_INSTANCE -- so a
#: loop compiled against A and re-entered holding a B read the wrong field and
#: returned a plausible wrong number. Before the fix this printed 333, which is
#: B's `y`, where the interpreter printed 444.
import std.io

class A {
    pub var x: int
    pub var y: int

    fn init(self) -> void {
        self.x = 111
        self.y = 222
    }
}

class B {
    pub var y: int
    pub var x: int

    fn init(self) -> void {
        self.y = 333
        self.x = 444
    }
}

fn sum_x(o: any, n: int) -> int {
    var s = 0
    var i = 0
    while i < n {
        s = s + o.x
        i = i + 1
    }
    return s
}

print(sum_x(A(), 200000) / 200000)
print(sum_x(B(), 200000) / 200000)
print(sum_x(A(), 200000) / 200000)
