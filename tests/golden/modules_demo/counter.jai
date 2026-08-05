#: Module-level state initialises exactly once, on first import.
var _count = 0

pub fn bump() -> int {
    _count += 1
    return _count
}

pub fn current() -> int {
    return _count
}

print("counter module body ran")
