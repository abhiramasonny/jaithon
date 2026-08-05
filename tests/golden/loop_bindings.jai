#: A loop binding is fresh on every iteration (spec §5.2, §6).
#:
#: Two closures made in two different iterations must never share a cell. For a
#: `let` binding — which every `for` pattern name is — spec §6's by-value capture
#: settles it when the closure is built. A `var` declared in the body is captured
#: by reference, so the emitter closes that iteration's upvalues at every edge
#: that ends an iteration: the bottom of the body, `continue`, and `break`,
#: labelled or not. Without the close, all the closures of one loop would share
#: one slot, and one that escaped the loop would go on aliasing whatever the
#: enclosing scope declares next.

# --- an uncaptured loop variable stays a plain slot -------------------------
var total = 0
for i in 0..5 { total = total + i }
print(total)

# --- a captured `let` binding: one snapshot per iteration -------------------
var byvalue = []
for i in 0..3 { byvalue.push(|| i) }
print(byvalue.map(|f| f()))

# --- a captured `var` in the body: one cell per iteration -------------------
var readers = []
var writers = []
for k in 0..3 {
    var cell = k * 10
    readers.push(|| cell)
    writers.push(fn() { cell = -1 })
}
writers[1]()  # reaches iteration 1's cell and no other
print(readers.map(|f| f()))

# --- a destructuring pattern binds several names, all per-iteration ---------
var pairs = [(1, "a"), (2, "b"), (3, "c")]
var labels = []
var bumps = []
for (n, s) in pairs {
    var seen = 0
    labels.push(|| f"{n}{s}{seen}")
    bumps.push(fn() { seen = seen + 1 })
}
bumps[0]()
bumps[0]()
print(labels.map(|f| f()))

# --- `while` bodies have the same rule --------------------------------------
var whiles = []
var w = 0
while w < 3 {
    var v = w
    whiles.push(|| v)
    w = w + 1
}
print(whiles.map(|f| f()))

# --- `continue` ends an iteration too ---------------------------------------
var kept = []
for m in 0..4 {
    var v = m
    kept.push(|| v)
    if m % 2 == 0 { continue }
    v = v + 100
}
print(kept.map(|f| f()))

# --- a nested loop and a labelled break close every level they leave --------
var grabbed = []
'outer: for row in 0..3 {
    var tag = row * 100
    for col in 0..3 {
        var mark = col
        grabbed.push(|| tag + mark)
        if row == 1 and col == 1 { break 'outer }
    }
}
print(grabbed.map(|f| f()))

# --- a closure that escapes via `break` keeps its own cell ------------------
#: Inside a function the loop's slots are handed straight back to the next
#: declaration, so an upvalue left open on the way out would read `filler_a`.
fn escape_via_break() -> fn() -> int {
    var escaped = || 0
    var n = 0
    loop {
        var v = n * 7
        escaped = || v
        n = n + 1
        if n == 3 { break }
    }
    let filler_a = 111
    let filler_b = 222
    print(filler_a + filler_b)
    return escaped
}
print(escape_via_break()())
