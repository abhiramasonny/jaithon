#: A module that binds a builtin type's name to a class of its own. A
#: `-> dict` boundary there means that class, so a builtin dict fails it: the
#: interpreter resolves a guard's name through the module first
#: (valueMatchesType), and compiled code, which answers the common guards
#: from the name alone, has to decline rather than wave the value through.
#:
#: Each helper is called more than JAI_JIT_THRESHOLD times, so before the
#: fix the whole-function tier compiled the guard and let the builtin pass on
#: every call after the 64th; `JAITHON_NO_JIT=1` runs the same file.
from std.test import assert_eq

let CALLS = 100

class str {
    pub var x: int

    fn init(self) { self.x = 1 }
}

class dict {
    pub var x: int

    fn init(self) { self.x = 2 }
}

class list {
    pub var x: int

    fn init(self) { self.x = 3 }
}

let WORDS = ["a", "b", "c"]

fn any_str(n: int) -> any {
    return WORDS[n % 3]
}

fn any_dict(n: int) -> any {
    var d = {}
    d[n] = 1
    return d
}

fn any_list(n: int) -> any {
    return [n]
}

fn as_str(n: int) -> str {
    let v = any_str(n)
    return v
}

fn as_dict(n: int) -> dict {
    let v = any_dict(n)
    return v
}

fn as_list(n: int) -> list {
    let v = any_list(n)
    return v
}

fn own_dict(v: any) -> dict {
    return v
}

pub fn test_a_builtin_str_fails_a_shadowed_str_guard() -> void {
    var raised = 0
    var call = 0
    while call < CALLS {
        try {
            as_str(call)
        } catch _e: Error {
            raised += 1
        }
        call += 1
    }
    assert_eq(raised, CALLS)
}

pub fn test_a_builtin_dict_fails_a_shadowed_dict_guard() -> void {
    var raised = 0
    var call = 0
    while call < CALLS {
        try {
            as_dict(call)
        } catch _e: Error {
            raised += 1
        }
        call += 1
    }
    assert_eq(raised, CALLS)
}

pub fn test_a_builtin_list_fails_a_shadowed_list_guard() -> void {
    var raised = 0
    var call = 0
    while call < CALLS {
        try {
            as_list(call)
        } catch _e: Error {
            raised += 1
        }
        call += 1
    }
    assert_eq(raised, CALLS)
}

pub fn test_the_module_class_passes_its_own_guard() -> void {
    var passed = 0
    var call = 0
    while call < CALLS {
        let d = own_dict(dict())
        passed += d.x - 1
        call += 1
    }
    assert_eq(passed, CALLS)
}
