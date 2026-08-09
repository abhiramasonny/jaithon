#: Reading a member off an enum is inline-cached, and these are the shapes that
#: cache can get wrong.
#:
#: Before it existed, 12.9 million of the 16.9 million slow-path property reads
#: in one `check lib/std` had an enum receiver -- the inline cache in
#: `OP_GET_FIELD_LOCAL` fired only for instances, and `vm.icMisses++` sat inside
#: that same branch, so enum reads were both uncached and invisible in the
#: reported hit rate.
#:
#: What makes caching them safe is that a variant's unit value and constructor
#: are built once and kept on the variant, the methods table is filled only
#: while the enum is being defined, and any addition bumps the enum's shapeId.
#: What makes it dangerous is a read site that sees more than one enum: the
#: names below collide on purpose, so a cache keyed on anything weaker than
#: identity hands back another enum's member and every assertion here fails.
from std.test import assert_eq

enum Color { Red, Green, Blue }
enum Fruit { Red, Green, Yellow }
enum Mood { Red, Green, Calm }
enum Sign { Red, Green, Stop }
enum Team { Red, Green, Blue }
enum Card { Red, Green, Heart }

#: One read site, deliberately shared by every enum above.
fn pick(e) {
    return e.Red
}

fn test_one_site_keeps_six_enums_apart() -> void {
    # Four ways of cache and six enums, so this walks monomorphic to
    # polymorphic to megamorphic in a single call.
    assert_eq(f"{pick(Color)}", "Color.Red")
    assert_eq(f"{pick(Fruit)}", "Fruit.Red")
    assert_eq(f"{pick(Mood)}", "Mood.Red")
    assert_eq(f"{pick(Sign)}", "Sign.Red")
    assert_eq(f"{pick(Team)}", "Team.Red")
    assert_eq(f"{pick(Card)}", "Card.Red")
}

fn test_answers_stay_right_after_the_site_goes_megamorphic() -> void {
    var i = 0
    while i < 8 {
        assert_eq(f"{pick(Color)}", "Color.Red")
        assert_eq(f"{pick(Card)}", "Card.Red")
        i = i + 1
    }
}

fn test_a_direct_site_is_unaffected() -> void {
    assert_eq(f"{Color.Blue}", "Color.Blue")
    assert_eq(f"{Card.Heart}", "Card.Heart")
    assert_eq(f"{Color.Blue}", "Color.Blue")
}

enum Shape {
    Circle(r: int)
    Dot

    pub fn describe(self) -> str {
        return match self {
            Shape.Circle(r) => f"circle {r}"
            Shape.Dot => "dot"
        }
    }
}

enum Blob {
    Circle(r: int)
    Dot

    pub fn describe(self) -> str {
        return "blob"
    }
}

#: A variant with a payload answers as a constructor, and one without answers as
#: a value. Both are built lazily on first read, which is exactly what a cache
#: would freeze at the wrong moment.
fn constructor_of(e) {
    return e.Circle
}

fn unit_of(e) {
    return e.Dot
}

fn test_payload_constructors_stay_with_their_own_enum() -> void {
    assert_eq(constructor_of(Shape)(3).describe(), "circle 3")
    assert_eq(constructor_of(Blob)(4).describe(), "blob")
    # Again, now that both ways are warm.
    assert_eq(constructor_of(Shape)(9).describe(), "circle 9")
    assert_eq(constructor_of(Blob)(1).describe(), "blob")
}

fn test_unit_variants_stay_with_their_own_enum() -> void {
    assert_eq(unit_of(Shape).describe(), "dot")
    assert_eq(unit_of(Blob).describe(), "blob")
    assert_eq(unit_of(Shape).describe(), "dot")
}

fn test_a_method_is_reachable_through_the_same_site() -> void {
    # Methods and variants share one lookup, so a cache that memoised the
    # variant path must not shadow the method path.
    assert_eq(Shape.Dot.describe(), "dot")
    assert_eq(Blob.Dot.describe(), "blob")
}
