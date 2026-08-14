#: The `is` type test (spec §4.2).
#:
#: `is` compares identity, except when its right operand denotes a type. That
#: reading is the one the checker narrows on (`typeOperand` in sema/check.c),
#: so the runtime has to agree with it: while it did not, `x is C` was `false`
#: for every instance of every class and every `if x is C` guard in the
#: language was dead code that silently took the wrong branch.
#:
#: A type on the right is a class, a trait, an enum, or one of the builtin type
#: names — which are bound as conversion *functions* (spec §9) and so have no
#: runtime type object at all, which is why `1 is int` was false too.
trait Greet {
    fn hello(self) -> str
}

class Base {
    pub fn init(self) { }
}

class Sub extends Base: Greet {
    pub fn init(self) { }

    pub fn hello(self) -> str {
        return "hi"
    }
}

enum Color { Red, Green }

# --- classes: exact, inherited, and the other way round ----------------------
let base = Base()
let sub = Sub()
print(base is Base)
print(sub is Base)
print(sub is Sub)
print(base is Sub)
print(base is not Sub)

# --- traits ------------------------------------------------------------------
print(sub is Greet)
print(base is Greet)

# --- enums -------------------------------------------------------------------
print(Color.Red is Color)
print(1 is Color)

# --- builtin type names ------------------------------------------------------
print(1 is int, 1.0 is float, 1 is float)
print("x" is str, true is bool, bytes("x") is bytes)
print([1] is list, {1: 2} is dict, {1} is set, (1, 2) is tuple)

# --- identity is untouched for everything else -------------------------------
let xs = [1, 2]
let ys = xs
let zs = [1, 2]
print(xs is ys, xs is zs)
print(5 is 5, 5 is 6)
print(null is null, 1 is null, null is not null)

# --- a class reached through a variable is still a type ----------------------
let alias = Base
print(sub is alias)

# --- and the guard narrows, which is the whole point -------------------------
fn describe(x: any) -> str {
    if x is Sub {
        return x.hello()
    }
    if x is Base {
        return "a base"
    }
    if x is int {
        return "an int"
    }
    return "something else"
}

print(describe(sub))
print(describe(base))
print(describe(7))
print(describe("s"))
