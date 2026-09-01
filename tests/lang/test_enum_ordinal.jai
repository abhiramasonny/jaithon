#: `ordinal()` is the variant's position, on unit variants and on variants
#: carrying data alike, and a method the enum declares under that name wins.
from std.test import assert_eq, assert_true

enum Color { Red, Green, Blue }

enum Expr { Num(value: float), Add(left: Expr, right: Expr) }

enum Shadow {
    A,
    B,

    fn ordinal(self) -> int { return 99 }
}

fn main() -> void {
    assert_eq(Color.Red.ordinal(), 0)
    assert_eq(Color.Green.ordinal(), 1)
    assert_eq(Color.Blue.ordinal(), 2)
    assert_eq(Expr.Num(1.5).ordinal(), 0)
    assert_eq(Expr.Add(Expr.Num(1.0), Expr.Num(2.0)).ordinal(), 1)
    assert_eq(Shadow.B.ordinal(), 99)
    let table = ["r", "g", "b"]
    assert_eq(table[Color.Blue.ordinal()], "b")
    let dynamic: any = Color.Green
    assert_eq(dynamic.ordinal(), 1)
    assert_true(dir(Color.Red).contains("ordinal"))
}

main()
