#: Value identity, block results, and who may see a private field.
#:
#: Five separate rules that were each broken once, kept together because they
#: are all the same kind of promise: an expression means one thing, and it means
#: it wherever it appears.
#:
#:   - A block read for its value yields a trailing `if` or `match` (spec §4.3),
#:     not null.
#:   - A payload-less enum variant is one value, so `is` — identity, spec §4.2 —
#:     holds between two mentions of it.
#:   - A variant that takes a payload is a function of it, and a function is a
#:     value: it survives being bound, passed, and tail-called.
#:   - `static fn` is a member of its class and sees the class's private fields
#:     (spec §7.1).
#:   - `<<` and `>>` are bit-pattern operations: they discard, they do not
#:     overflow (spec §1.6).

# --- a block yields its trailing if / match ---------------------------------
enum Step { Halt, Go(distance: int) }

fn advance(step: Step, moving: bool) -> int? {
    return match step {
        Step.Halt => 0,
        Step.Go(distance) => {
            let scaled = distance * 2
            if moving {
                scaled
            } else {
                null
            }
        }
    }
}

fn nested(step: Step, moving: bool, doubled: bool) -> int? {
    return match step {
        Step.Halt => 0,
        Step.Go(distance) => {
            if not moving {
                null
            } elif doubled {
                distance * 2
            } else {
                distance
            }
        }
    }
}

#: A trailing `match` is a value in the same way a trailing `if` is.
#:
#: `{ ... }` in expression position is a set literal, so a block is only read
#: for its value where one is already expected: a match arm, or an `if` branch.
fn classify(value: int, loud: bool) -> str {
    let word = if loud {
        let shout = match value {
            0 => "ZERO",
            1 => "ONE",
            _ => "MANY",
        }
        shout
    } else {
        match value {
            0 => "zero",
            1 => "one",
            _ => "many",
        }
    }
    return word
}

print(advance(Step.Go(21), true))
print(advance(Step.Go(21), false))
print(advance(Step.Halt, true))
print(nested(Step.Go(7), true, true))
print(nested(Step.Go(7), true, false))
print(nested(Step.Go(7), false, false))
print(classify(0, false))
print(classify(1, true))
print(classify(9, false))

# --- a payload-less variant is a single value -------------------------------
enum Token { Number(value: int), Plus, LParen, RParen }

let here = Token.LParen
let again = Token.LParen

print(here is again)
print(here is Token.LParen)
print(here == Token.LParen)
print(here is Token.Plus)
print(Token.RParen is Token.RParen)

#: The parser idiom the identity rule exists for.
fn is_open(t: Token) -> bool { return t is Token.LParen }

print(is_open(Token.LParen))
print(is_open(Token.Plus))

# --- a payload variant is a first-class function ----------------------------
fn describe(t: Token) -> str {
    return match t {
        Token.Number(v) => f"Number({v})",
        Token.Plus => "Plus",
        Token.LParen => "LParen",
        Token.RParen => "RParen",
    }
}

let make_number = Token.Number
print(describe(make_number(5)))
print(Token.Number is Token.Number)

#: A tail call through a variant: the callee is a value by the time it is
#: called, so there is nothing left for a call site to special-case.
fn wrap(value: int) -> Token {
    return Token.Number(value)
}

print(describe(wrap(11)))

fn apply(f: fn(int) -> Token, value: int) -> Token { return f(value) }

print(describe(apply(Token.Number, 3)))

# --- a static method sees its class's private fields ------------------------
class Counter {
    var ticks: int

    fn init(self) { self.ticks = 0 }

    #: A factory reaches the field it exists to fill in.
    pub static fn starting_at(value: int) -> Counter {
        let c = Counter()
        c.ticks = value
        return c
    }

    pub fn read(self) -> int { return self.ticks }
}

print(Counter.starting_at(41).read())
print(Counter().read())

# --- shifts discard rather than overflow ------------------------------------
let SPLIT_MIX_GAMMA = 0x9E3779B9 << 32 | 0x7F4A7C15

print(SPLIT_MIX_GAMMA)
print(1 << 62)
print(1 << 63)
print(1 << 63 < 0)
print(1 << 64)
print(-8 >> 1)
print(-1 >> 63)
print(255 >> 4)

fn shift_left(value: int, count: int) -> int { return value << count }

print(shift_left(0x9E3779B9, 32) | 0x7F4A7C15)
print(shift_left(1, 63) == SPLIT_MIX_GAMMA - SPLIT_MIX_GAMMA + (1 << 63))
