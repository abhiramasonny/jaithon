# Mutating a container while a `for` is walking it is a RuntimeError, not a
# silently wrong traversal. Two distinct faults: a resize, which invalidates
# the iterator's bounds, and an in-place store, which does not — the second is
# only visible because every container carries a version counter.

# --- lists: resize --------------------------------------------------------
let grow = [1, 2, 3]
try {
    for x in grow { grow.push(x) }
} catch e: RuntimeError {
    print(f"grow: {e.message}")
}

let shrink = [1, 2, 3]
try {
    for _x in shrink { shrink.pop() }
} catch e: RuntimeError {
    print(f"shrink: {e.message}")
}

# A pop paired with a push leaves the count where it started, so the count
# alone cannot witness it.
let churn = [1, 2, 3]
try {
    for _x in churn {
        churn.pop()
        churn.push(9)
    }
} catch e: RuntimeError {
    print(f"churn: {e.message}")
}

# --- lists: in-place ------------------------------------------------------
let store = [1, 2, 3]
try {
    for _x in store { store[0] = 99 }
} catch e: RuntimeError {
    print(f"store: {e.message}")
}

let sliced = [1, 2, 3, 4]
try {
    for _x in sliced { sliced[0:2] = [8, 9] }
} catch e: RuntimeError {
    print(f"slice: {e.message}")
}

let sorting = [3, 1, 2]
try {
    for _x in sorting { sorting.sort() }
} catch e: RuntimeError {
    print(f"sort: {e.message}")
}

let flipping = [1, 2, 3]
try {
    for _x in flipping { flipping.reverse() }
} catch e: RuntimeError {
    print(f"reverse: {e.message}")
}

let wiping = [1, 2, 3]
try {
    for _x in wiping { wiping.clear() }
} catch e: RuntimeError {
    print(f"clear: {e.message}")
}

# --- dicts and sets -------------------------------------------------------
let d = {"a": 1, "b": 2}
try {
    for k in d { d[k + "!"] = 1 }
} catch e: RuntimeError {
    print(f"dict grow: {e.message}")
}

let d2 = {"a": 1, "b": 2}
try {
    for k in d2 { d2[k] = 7 }
} catch e: RuntimeError {
    print(f"dict store: {e.message}")
}

let s = {1, 2, 3}
try {
    for v in s { s.add(v + 10) }
} catch e: RuntimeError {
    print(f"set grow: {e.message}")
}

# --- what stays legal -----------------------------------------------------
# Reading is free, and so is mutating a *different* container.
let source = [1, 2, 3]
let sink: list[int] = []
for x in source { sink.push(x * 2) }
print(f"copied: {sink}")

# A snapshot is the supported way to edit while walking.
let live = [1, 2, 3, 4]
for x in live.copy() {
    if x % 2 == 0 { live.remove(x) }
}
print(f"filtered: {live}")

# Mutating after the loop has finished is fine.
for _x in live { }
live.push(5)
print(f"after: {live}")

# Immutable sources have nothing to witness and never raise.
let t = (1, 2, 3)
var total = 0
for x in t { total = total + x }
for _c in "abc" { total = total + 1 }
for i in 0..3 { total = total + i }
print(f"immutable: {total}")
