# expect: E0203
# `break` needs an enclosing loop.

fn f() -> void {
    break
}
f()
