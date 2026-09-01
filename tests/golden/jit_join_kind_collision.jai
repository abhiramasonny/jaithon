#: A ternary whose two edges carry kinds that used to COLLIDE in the join check.
#:
#: `stackSignatureAt` packed `kind & 3` per entry -- a 15-kind enum hashed into
#: two bits -- so SLOT_FLOAT(1) and SLOT_OBJ(13) compared equal and a genuinely
#: disagreeing join passed. The body compiled, its return kind was whichever
#: edge the linear walk happened to arrive on, and a double's bits went back
#: tagged VAL_OBJ for the collector to trace. SIGSEGV under every JIT
#: configuration, correct only under JAITHON_NO_JIT=1.
#:
#: SLOT_BOOL(11) and SLOT_MAYBE_INST(3) collided the same way; the sibling case
#: is `n == 0 ? true : null`.
class Box {
    pub var v: int

    pub fn init(self, v: int) { self.v = v }
}

fn t_floatobj(n: int) -> any {
    return n == 0 ? 1.5 : "s"
}

pub fn main() {
    var acc = 0
    for i in 0..400 {
        let a = t_floatobj(0)
        let b = t_floatobj(1)
        acc = acc + 1
    }
    print(acc)
    print(t_floatobj(0))
    print(t_floatobj(1))
}
