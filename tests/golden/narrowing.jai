#: Flow-sensitive narrowing of optionals (spec §2.1, §4.2).
#:
#: `T?` is `T | null`. A guard that rules `null` out must leave the binding
#: usable as `T` for the rest of the guarded region, whichever way the guard is
#: spelled, and a store inside that region is checked against the declaration
#: rather than against the narrowed view.
fn double(n: int) -> int { return n * 2 }

# --- the guard on the true edge ---------------------------------------------
fn ne_form(value: int?) -> int {
    if value != null { return double(value) }
    return -1
}

fn is_not_form(value: int?) -> int {
    if value is not null { return double(value) }
    return -1
}

fn negated_form(value: int?) -> int {
    if not value is null { return double(value) }
    return -1
}

print(ne_form(21))
print(is_not_form(21))
print(negated_form(21))
print(is_not_form(null))

# --- the guard as an early return -------------------------------------------
fn eq_early(value: int?) -> int {
    if value == null { return -1 }
    return double(value)
}

fn is_early(value: int?) -> int {
    if value is null { return -1 }
    return double(value)
}

print(eq_early(3))
print(is_early(3))
print(is_early(null))

# --- a conjunction narrows both operands ------------------------------------
fn sum_both(a: int?, b: int?) -> int {
    if a is not null and b is not null { return a + b }
    return -1
}

print(sum_both(2, 5))
print(sum_both(2, null))

# --- storing into a narrowed binding ----------------------------------------
class Countdown {
    var n: int

    fn init(self, n: int) { self.n = n }

    #: Count down to zero, then report exhaustion the way an iterator does.
    pub fn next(self) -> int? {
        if self.n <= 0 { return null }
        self.n -= 1
        return self.n
    }
}

fn drain(source: Countdown) -> int {
    var total = 0
    var value = source.next()
    while value is not null {
        total += double(value)
        value = source.next()
    }
    return total
}

fn first_two(source: Countdown) -> int {
    var value = source.next()
    if value is null { return -1 }
    var total = value
    value = source.next()
    if value is null { return total }
    return total + value
}

print(drain(Countdown(4)))
print(first_two(Countdown(4)))
print(first_two(Countdown(1)))
print(first_two(Countdown(0)))

# --- the same forms with a generic inner type -------------------------------
class Slot[T] {
    var value: T?
    var last: T?
    var refill: T?

    fn init(self, value: T?, refill: T?) {
        self.value = value
        self.last = null
        self.refill = refill
    }

    #: Takes a bare `T`, so the guards below only reach it once narrowing has
    #: taken `null` out of the union.
    fn record(self, item: T) -> void { self.last = item }

    pub fn or_else(self, fallback: T) -> T {
        let held = self.value
        if held is not null { return held }
        return fallback
    }

    pub fn or_else_early(self, fallback: T) -> T {
        let held = self.value
        if held is null { return fallback }
        return held
    }

    pub fn or_else_ne(self, fallback: T) -> T {
        let held = self.value
        if held != null { return held }
        return fallback
    }

    #: The loop every iterator adapter is built from, generically: the guard
    #: narrows to `T`, and the store at the end of the body puts a fresh `T?`
    #: back into the same binding.
    pub fn drain(self, limit: int) -> int {
        var total = 0
        var value = self.value
        while value is not null {
            self.record(value)
            total += 1
            if total >= limit { return total }
            value = self.refill
        }
        return total
    }
}
