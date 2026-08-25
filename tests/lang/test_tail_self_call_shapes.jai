#: What `return self.name(...)` lowers to, and why it depends on what `name` is.
#:
#: `Op.TailCall` reuses the caller's frame only when the callee passes
#: `IS_CLOSURE`. A METHOD reached through `self` puts a bound method there --
#: an ObjBound is not a closure -- so the tail form never reused a frame for
#: one, and paid for the privilege: the compiled tier reads `self.m` as a FIELD
#: and declines the whole enclosing body at `OP_GET_FIELD_LOCAL`. Those lower
#: to `Op.Invoke` now.
#:
#: A callable FIELD reached through `self` is a different animal. It IS a plain
#: closure, it DOES reuse the frame, and an invoke would resolve the name
#: against the class's method table first -- so a field shadowing an inherited
#: method would silently call the wrong one. Fields keep the tail form, which
#: is why the emitter asks whether the name is a method OF THE ENCLOSING CLASS
#: rather than merely whether the receiver is `self`.
from std.test import assert_eq, assert_true

class Doubler {
    pub fn twice(self, n: int) -> int {
        return n + n
    }

    #: The method case: an invoke after the change, the same answer before it.
    pub fn through_method(self, n: int) -> int {
        return self.twice(n)
    }
}

fn test_a_method_in_tail_position_answers_the_same() -> void {
    let d = Doubler()
    var total = 0
    var i = 0
    while i < 200_000 {
        total += d.through_method(3)
        i += 1
    }
    assert_eq(total, 1_200_000)
}

#: The frame reuse a callable field still has. 400000 is four hundred times the
#: ~1022 frames ordinary recursion gets, so this only returns at all because
#: the tail form is still emitted here.
class Countdown {
    pub var go: fn(int) -> int

    pub fn init(self) -> void {
        self.go = fn(n: int) -> int {
            if n <= 0 {
                return 0
            }
            return self.go(n - 1)
        }
    }

    pub fn start(self, n: int) -> int {
        return self.go(n)
    }
}

fn test_a_callable_field_still_reuses_its_frame() -> void {
    assert_eq(Countdown().start(400_000), 0)
}

#: And the shadowing the method table would get wrong. `pick` is a field here
#: and a method one class up; the field is what `self.pick()` means, and an
#: invoke would find the base method instead.
class Chooser {
    pub fn pick(self) -> str {
        return "base method"
    }
}

class Overrider extends Chooser {
    pub var pick: fn() -> str

    pub fn init(self) -> void {
        self.pick = fn() -> str { return "own field" }
    }

    pub fn choose(self) -> str {
        return self.pick()
    }
}

fn test_a_field_shadowing_an_inherited_method_wins() -> void {
    let o = Overrider()
    var seen = ""
    var i = 0
    while i < 100_000 {
        seen = o.choose()
        i += 1
    }
    assert_eq(seen, "own field")
}

#: An inherited method keeps the old shape -- it is not in the subclass's own
#: method list -- and still resolves to the base.
class Greeter {
    pub fn greet(self) -> str {
        return "hello"
    }
}

class Polite extends Greeter {
    pub fn speak(self) -> str {
        return self.greet()
    }
}

fn test_an_inherited_method_in_tail_position_resolves_to_the_base() -> void {
    let p = Polite()
    var seen = ""
    var i = 0
    while i < 100_000 {
        seen = p.speak()
        i += 1
    }
    assert_eq(seen, "hello")
}

#: The limit a method tail call has always had, pinned so that a later attempt
#: to give methods frame reuse is a deliberate change rather than a surprise.
class Sinker {
    pub fn down(self, n: int) -> int {
        if n <= 0 { return 0 }
        return self.down(n - 1)
    }
}

fn test_a_method_tail_call_still_bottoms_out() -> void {
    var raised = false
    try {
        Sinker().down(200_000)
    } catch e: Error {
        raised = true
    }
    assert_true(raised, "a method tail call reused its frame, which it never did")
}
