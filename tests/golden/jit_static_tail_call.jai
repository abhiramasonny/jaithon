#: `return Klass.f()` is OP_GET_FIELD on the class then OP_TAIL_CALL, not the
#: OP_INVOKE the static-call arm speaks. The static method read as a value
#: becomes a SLOT_FUNC entry behind the static-call guards, and the callee --
#: called only from here, so it has not returned when this body first
#: compiles -- is retried once it has. The body also returns an int on one
#: edge and an instance on another, so its return kind is SLOT_DYNAMIC. This
#: is the shape of jaithon.ast.schema._default_field's `Span.none()` arm.

class Span {
    pub var file: int
    fn init(self, file: int) { self.file = file }
    pub static fn none() -> Span { return Span(-1) }
}

fn deflt(tag: str) -> any {
    if tag == "int" { return -1 }
    if tag == "span" { return Span.none() }
    return null
}

fn main() {
    var n = 0
    # The "span" arm is reached only after the body has crossed its own
    # threshold many times over, so its callee is cold at every early attempt.
    for i in 0..3000 {
        var tag = "x"
        if i % 3 == 0 { tag = "int" }
        if i >= 700 and i % 3 == 1 { tag = "span" }
        let v = deflt(tag)
        if v is Span { n = n + v.file }
        if v is int { n = n + v }
    }
    print(n)
}
