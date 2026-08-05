#: `super` reaching a parent method written in C (spec §6.3, §7.2).
#:
#: Every method of `Error` is a native, so a custom exception is the case that
#: exercises this. The dispatch used to hand the native the arguments alone and
#: drop the receiver, which made `super(message)` report that `Error.init()`
#: needed an Error instance as its receiver and left no way at all to define an
#: exception that carries a message. The same slot mistake reached any built-in
#: method found through a class's own method table, so `e.traceback_string()` on
#: a plain ValueError failed on arity too.
from std.test import assert_eq, assert_true, assert_throws

class MyError extends ValueError {
    fn init(self, m: str) { super(m) }
}

class Loud extends Error {
    fn init(self, m: str) { super(m) }

    pub fn __str__(self) -> str { return "LOUD: " + super.__str__() }
}

fn test_a_custom_exception_carries_its_message() -> void {
    let caught = assert_throws(ValueError, fn() { throw MyError("mine") })
    assert_eq(caught.message, "mine")
    assert_eq(type_of(caught), "MyError")
}

fn test_a_custom_exception_is_caught_by_its_base() -> void {
    var seen = ""
    try {
        throw MyError("boom")
    } catch e: Error {
        seen = e.message
    }
    assert_eq(seen, "boom")
}

fn test_super_reaches_an_overridden_native_method() -> void {
    let l = Loud("hey")
    assert_eq(l.__str__(), "LOUD: hey")
    assert_eq(str(l), "LOUD: hey")
}

fn test_a_native_method_off_an_instance_gets_its_receiver() -> void {
    let e = ValueError("plain")
    assert_eq(e.message, "plain")
    assert_eq(e.__str__(), "plain")
    assert_true(e.traceback_string() is str)
}

class Holder {
    pub var f: fn(int) -> int

    fn init(self, f: fn(int) -> int) { self.f = f }
}

fn test_a_callable_field_is_still_a_plain_call() -> void {
    # `h.f(4)` looks like a method call and is not one: the field goes in slot
    # zero and gets no receiver, which is the case the fix must not disturb.
    let h = Holder(|x| x * 3)
    assert_eq(h.f(4), 12)
}
