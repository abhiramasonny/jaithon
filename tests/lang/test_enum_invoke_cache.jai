#: An enum value's builtin methods are inline-cached at an `INVOKE` site, keyed
#: on the enum's shape -- so that `kind.ordinal()` in an interpreted loop stops
#: looking the name up and allocating a bound method on every call.
#:
#: The key has to be the enum, not one tag for every enum value: an enum's own
#: method of the same name wins, and a variant's payload field of that name
#: wins over both. Each call site below is shared by enums that disagree, and
#: is called many times, so a way filled for one of them and reused for
#: another fails an assertion.
from std.test import assert_eq

enum Color { Red, Green, Blue }

enum Size { Small, Large }

enum Shadow {
    A,
    B,

    pub fn ordinal(self) -> int { return 99 }
}

enum Holder { Empty, Has(ordinal: any) }

#: One invoke site, shared by every enum above.
fn ord(e: any) -> int {
    return e.ordinal()
}

fn test_one_site_keeps_enums_apart() -> void {
    var total = 0
    var i = 0
    while i < 200 {
        assert_eq(ord(Color.Blue), 2)
        assert_eq(ord(Size.Large), 1)
        assert_eq(ord(Shadow.A), 99)
        assert_eq(ord(Color.Red), 0)
        total += ord(Size.Small) + ord(Shadow.B)
        i += 1
    }
    assert_eq(total, 200 * 99)
}

fn test_a_payload_field_wins_at_a_shared_site() -> void {
    var i = 0
    while i < 100 {
        assert_eq(ord(Holder.Empty), 0)
        assert_eq(ord(Holder.Has(|| 42)), 42)
        assert_eq(ord(Color.Green), 1)
        i += 1
    }
}

fn test_a_megamorphic_site_still_answers() -> void {
    let values: list[any] = [Color.Red, Size.Large, Shadow.A, Color.Blue,
                             Holder.Empty, Size.Small, Shadow.B, Color.Green]
    var sum = 0
    for round in 0..50 {
        for v in values {
            sum += ord(v)
        }
    }
    assert_eq(sum, 50 * (0 + 1 + 99 + 2 + 0 + 0 + 99 + 1))
}

fn ord_color(c: Color) -> int {
    return c.ordinal()
}

fn ord_shadow(s: Shadow) -> int {
    return s.ordinal()
}

fn test_typed_receivers_take_their_own_method() -> void {
    var sum = 0
    for k in 0..300 {
        sum += ord_color(Color.Blue) + ord_shadow(Shadow.A)
    }
    assert_eq(sum, 300 * (2 + 99))
}
