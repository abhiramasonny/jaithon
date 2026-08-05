#: What may be a dict key or a set element (spec §5.4, §5.5).
#:
#: The rule has two halves and both used to go unenforced on the operator
#: paths: `d[null] = 1` tripped an assertion inside the table, and an
#: unhashable key was dropped without a word, so the write looked like it had
#: happened until the read raised KeyError. `dict.set()` and `set.add()` always
#: checked; the subscript, the literals and the comprehensions did not.
from std.test import assert_eq, assert_true, assert_false, assert_throws

fn test_null_is_not_a_dict_key() -> void {
    var d = {}
    assert_throws(TypeError, fn() { d[null] = 1 })
    assert_eq(d.len(), 0)

    assert_throws(TypeError, fn() { let _bad = {null: 1} })
    assert_throws(TypeError, || d.set(null, 1))
    assert_throws(TypeError, fn() { let _bad = {k: 1 for k in [null]} })
}

fn test_null_is_not_a_set_element() -> void {
    var s = set()
    assert_throws(TypeError, || s.add(null))
    assert_eq(s.len(), 0)

    assert_throws(TypeError, fn() { let _bad = {null} })
    assert_throws(TypeError, fn() { let _bad = {v for v in [null]} })
}

fn test_a_mutable_container_is_not_a_key() -> void {
    var d = {}
    let key = [1, 2]
    assert_throws(TypeError, fn() { d[key] = "one" })
    assert_eq(d.len(), 0)  # the write must not have half happened

    assert_throws(TypeError, fn() { d[{"a": 1}] = "two" })
    assert_throws(TypeError, fn() { d[{1, 2}] = "three" })
    assert_throws(TypeError, fn() { let _bad = {key: "x"} })
    assert_throws(TypeError, fn() { let _bad = {key} })
}

fn test_a_tuple_is_a_key_only_when_its_parts_are() -> void {
    var d = {}
    d[1, "a"] = "ok"
    assert_eq(d[1, "a"], "ok")
    assert_throws(TypeError, fn() { d[1, [2]] = "no" })
    assert_eq(d.len(), 1)
}

enum Colour { Red, Green }

class Point {
    var x: int
    var y: int

    fn init(self, x: int, y: int) {
        self.x = x
        self.y = y
    }

    pub fn __hash__(self) -> int { return self.x * 31 + self.y }

    pub fn __eq__(self, other: any) -> bool {
        return other is Point and other.x == self.x and other.y == self.y
    }
}

class Opaque {
    fn init(self) { }
}

fn test_every_legal_key_kind_still_works() -> void {
    var d = {}
    d[7] = "int"
    d[1.5] = "float"
    d[true] = "bool"
    d["s"] = "str"
    d[bytes([120, 121])] = "bytes"
    d[1, 2] = "tuple"
    d[Colour.Red] = "enum"
    d[Point(1, 2)] = "instance"
    assert_eq(d.len(), 8)
    assert_eq(d[Point(1, 2)], "instance")  # __hash__ and __eq__ agree
    assert_eq(d[Colour.Red], "enum")
}

fn test_an_instance_without_hash_is_not_a_key() -> void {
    var d = {}
    assert_throws(TypeError, fn() { d[Opaque()] = 1 })
    assert_eq(d.len(), 0)
}

fn test_contains_answers_membership_for_dict_and_set() -> void {
    let seen = {1, 2, 3}
    assert_true(seen.contains(2))
    assert_true(seen.has(2))
    assert_false(seen.contains(9))

    let ages = {"ada": 36}
    assert_true(ages.contains("ada"))
    assert_true(ages.has("ada"))
    assert_false(ages.contains("bob"))
}
