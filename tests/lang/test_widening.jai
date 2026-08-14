#: int widens to float (spec §2.2, §2.5), and only in that direction.
from std.test import assert_eq, assert_true, assert_throws

fn test_every_arithmetic_operator_widens() -> void {
    let n = 3
    assert_eq(n + 1.5, 4.5)
    assert_eq(1.5 + n, 4.5)
    assert_eq(n - 0.5, 2.5)
    assert_eq(0.5 - n, -2.5)
    assert_eq(n * 2.0, 6.0)
    assert_eq(2.0 * n, 6.0)
    assert_eq(n / 2.0, 1.5)
    assert_eq(6.0 / n, 2.0)
    assert_eq(n // 2.0, 1.0)
    assert_eq(7.0 // n, 2.0)
    assert_eq(n % 2.0, 1.0)
    assert_eq(7.5 % n, 1.5)
    assert_eq(n ** 2.0, 9.0)
    assert_eq(2.0 ** n, 8.0)
}

fn test_the_result_of_a_mixed_operator_is_float() -> void {
    let n = 3
    assert_eq(type_of(n + 1.5), "float")
    assert_eq(type_of(n * 2.0), "float")
    assert_eq(type_of(n // 2.0), "float")
    assert_eq(type_of(n ** 2.0), "float")
    assert_eq(type_of(n + 1), "int")
    assert_eq(type_of(n // 2), "int")
    assert_eq(type_of(n / 2), "float")
}

fn test_a_widened_int_keeps_its_value() -> void {
    let n = 7
    assert_eq(n + 0.0, 7.0)
    assert_eq(float(n), n + 0.0)
    assert_eq(-7 // 2.0, -4.0)
    assert_eq(-7 % 3.0, 2.0)
}

fn test_the_celsius_table_needs_no_conversion() -> void {
    var celsius = 0
    var readings: list[float] = []
    while celsius <= 100 {
        readings.push(celsius * 9.0 / 5.0 + 32.0)
        celsius += 20
    }
    assert_eq(readings, [32.0, 68.0, 104.0, 140.0, 176.0, 212.0])
}

fn test_widening_at_a_binding() -> void {
    let n = 3
    let x: float = n
    assert_eq(x, 3.0)
    assert_eq(type_of(x), "float")

    let optional: float? = n
    assert_eq(optional, 3.0)
}

fn test_widening_at_an_assignment() -> void {
    let n = 3
    var x: float = 0.0
    x = n
    assert_eq(x, 3.0)
    x += n
    assert_eq(x, 6.0)
    x *= n
    assert_eq(x, 18.0)
    assert_eq(type_of(x), "float")
}

fn scale(value: float, by: float = 2.0) -> float { return value * by }

fn test_widening_at_an_argument() -> void {
    let n = 3
    assert_eq(scale(n), 6.0)
    assert_eq(scale(n, n), 9.0)
    assert_eq(scale(1.5, n), 4.5)
}

fn as_float(n: int) -> float { return n }

fn test_widening_at_a_return() -> void {
    assert_eq(as_float(4), 4.0)
    assert_eq(type_of(as_float(4)), "float")
}

fn test_widening_into_annotated_collections() -> void {
    let n = 3
    let items: list[float] = [n, 1.5, 2]
    assert_eq(items, [3.0, 1.5, 2.0])
    assert_eq(type_of(items[0]), "float")

    let members: set[float] = {n}
    assert_true(3.0 in members)

    let table: dict[str, float] = {"a": n, "b": 2.5}
    assert_eq(table["a"], 1.0 + 2.0)
    assert_eq(type_of(table["a"]), "float")

    let pair: tuple[float, float] = (n, 1.5)
    assert_eq(pair, (3.0, 1.5))
    assert_eq(type_of(pair[0]), "float")
}

class Sample {
    pub var ratio: float

    pub fn init(self, ratio: float) -> void { self.ratio = ratio }
}

fn test_widening_into_a_field() -> void {
    let n = 3
    let s = Sample(n)
    assert_eq(s.ratio, 3.0)
    s.ratio = 4
    assert_eq(s.ratio, 4.0)
    s.ratio += n
    assert_eq(s.ratio, 7.0)
}

fn test_widening_on_the_dynamic_path() -> void {
    let n: any = 3
    let x: any = 1.5
    assert_eq(n + x, 4.5)
    assert_eq(x * n, 4.5)
    assert_eq(type_of(n - x), "float")
}

fn dynamic_int() -> any { return 5 }

fn test_a_dynamic_int_widens_at_a_float_boundary() -> void {
    let x: float = dynamic_int()
    assert_eq(x, 5.0)
    assert_eq(type_of(x), "float")
}

fn test_widening_does_not_disturb_comparison_or_hashing() -> void {
    assert_true(1 == 1.0)
    assert_true(1 < 2.0)
    assert_true(2.0 > 1)
    assert_true(1 <= 1.0)
    assert_eq(hash(1), hash(1.0))

    var lookup: dict[any, str] = {}
    lookup[1.0] = "one"
    assert_eq(lookup[1], "one")
}

fn test_bitwise_does_not_widen() -> void {
    let x: any = 2.0
    assert_throws(TypeError, || x & 1)
    assert_throws(TypeError, || x | 1)
    assert_throws(TypeError, || x ^ 1)
    assert_throws(TypeError, || x << 1)
    assert_throws(TypeError, || x >> 1)
    assert_throws(TypeError, || ~x)
}

fn test_the_wrapping_operators_do_not_widen() -> void {
    let x: any = 2.0
    let n: any = 1
    assert_throws(TypeError, || n +% x)
    assert_throws(TypeError, || n -% x)
    assert_throws(TypeError, || n *% x)
}

fn test_a_huge_int_widens_to_the_nearest_double() -> void {
    let exact = 9007199254740993  # 2**53 + 1
    assert_eq(exact + 0.0, 9007199254740992.0)
    assert_true(exact + 0.0 == 9007199254740992.0)
}
