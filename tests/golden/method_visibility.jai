#: Runtime enforcement of method visibility (spec §7.1). The checker rejects
#: what it can prove (E0701); everything reached here goes through an
#: `any`-typed parameter, which is exactly the case it cannot prove.
class Vault {
    var code: int

    fn init(self, code: int) { self.code = code }

    fn unlock(self) -> int {
        return self.code
    }

    prot fn hint(self) -> str { return "starts with a digit" }

    static fn issued() -> int {
        return 1
    }

    pub fn open(self) -> int {
        return self.unlock()
    }

    pub fn clue(self) -> str { return self.hint() }
}

class Deposit extends Vault {
    fn init(self) { super(99) }

    pub fn nudge(self) -> str {
        return self.hint()
    }
}

fn call_unlock(target: any) { return target.unlock() }

fn call_hint(target: any) { return target.hint() }

fn call_issued(target: any) { return target.issued() }

fn read_unlock(target: any) { return target.unlock }

# --- from inside the declaring class ---------------------------------------
let v = Vault(4)
print(v.open())
print(v.clue())
print(Deposit().open())
print(Deposit().nudge())

# --- from outside ----------------------------------------------------------
try { print(call_unlock(v)) } catch e: AttributeError { print(e.message) }
try { print(call_hint(v)) } catch e: AttributeError { print(e.message) }
try { print(call_issued(Vault)) } catch e: AttributeError { print(e.message) }
try { print(call_unlock(Deposit())) } catch e: AttributeError { print(e.message) }

try { print(read_unlock(v)) } catch e: AttributeError { print(e.message) }

# --- an override that adds `pub` widens ------------------------------------
class Sealed {
    fn note(self) -> str { return "sealed" }
}

class Opened extends Sealed {
    pub fn note(self) -> str { return "opened" }
}

fn call_note(target: any) { return target.note() }

print(call_note(Opened()))
try { print(call_note(Sealed())) } catch e: AttributeError { print(e.message) }

# --- a property is a method, and obeys the same rule ------------------------
class Temp {
    var c: float

    fn init(self) { self.c = 0.0 }

    pub get f() -> float { return self.c * 1.8 + 32.0 }

    get raw() -> float { return self.c }

    set raw(v: float) { self.c = v }

    pub fn warm(self) -> float {
        self.raw = 10.0
        return self.raw
    }
}

fn read_raw(target: any) { return target.raw }

fn write_raw(target: any) -> void { target.raw = 1.0 }

let temp = Temp()
print(temp.warm())
print(temp.f)
try { print(read_raw(temp)) } catch e: AttributeError { print(e.message) }
try { write_raw(temp) } catch e: AttributeError { print(e.message) }
