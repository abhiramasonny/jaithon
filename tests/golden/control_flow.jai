#: if/elif/else, loops, labels, match, and comprehensions (spec §5).

# --- if ---------------------------------------------------------------------
let n = 7
if n > 10 {
    print("big")
} elif n > 5 {
    print("medium")
} else {
    print("small")
}

# `if` is an expression when both arms produce a value.
let label = if n % 2 == 0 { "even" } else { "odd" }
print(label)

# --- while ------------------------------------------------------------------
var i = 0
var total = 0
while i < 5 {
    total += i
    i += 1
}
print(total)

# --- loop + break -----------------------------------------------------------
var j = 0
loop {
    j += 1
    if j >= 3 { break }
}
print(j)

# --- for over ranges --------------------------------------------------------
var acc = []
for k in 0..5 { acc.push(k) }
print(acc)

acc = []
for k in 0..=5 { acc.push(k) }
print(acc)

# --- for over containers ----------------------------------------------------
for item in ["a", "b"] { print(item) }

for (index, value) in ["x", "y"].enumerate() {
    print(f"{index}:{value}")
}

let scores = {"ada": 1, "alan": 2}
var names = []
for (key, _value) in scores.items() { names.push(key) }
names.sort()
print(names)

# --- continue ---------------------------------------------------------------
var odds = []
for k in 0..10 {
    if k % 2 == 0 { continue }
    odds.push(k)
}
print(odds)

# --- labelled break and continue --------------------------------------------
var found = null
'outer: for row in 0..5 {
    for col in 0..5 {
        if row * col == 6 {
            found = [row, col]
            break 'outer
        }
    }
}
print(found)

var kept = []
'rows: for row in 0..3 {
    for col in 0..3 {
        if col == 1 { continue 'rows }
        kept.push([row, col])
    }
}
print(kept)

# --- match ------------------------------------------------------------------
fn describe(code: int) -> str {
    return match code {
        200 => "ok",
        301 | 302 => "redirect",
        400..=499 => "client error",
        c if c >= 500 => "server error",
        _ => "unknown",
    }
}
print(describe(200))
print(describe(302))
print(describe(404))
print(describe(503))
print(describe(100))

# match over a destructured sequence
fn head_tail(xs: list[int]) -> str {
    return match xs {
        [] => "empty",
        [only] => f"one: {only}",
        [first, ...rest] => f"{first} then {rest.len()} more",
    }
}
print(head_tail([]))
print(head_tail([9]))
print(head_tail([1, 2, 3]))

# --- comprehensions ---------------------------------------------------------
print([x * 2 for x in 0..5])
print([x for x in 0..10 if x % 3 == 0])
print([[r, c] for r in 0..2 for c in 0..2])
print({x: x * x for x in 1..4})
let uniq = {x % 3 for x in 0..10}
print(uniq.to_list().sorted())
