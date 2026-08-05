# expect: E0501
# A match over an enum must cover every variant or have a `_` arm.

enum Color { Red, Green, Blue }

fn code(c: Color) -> int {
    return match c {
        Color.Red   => 0,
        Color.Green => 1,
    }
}
print(code(Color.Blue))
