#: A declared container element type must be enforced at run time.
#:
#: `list[int]` is otherwise a compile-time fact only: a container reaching a
#: function through `any` can be mutated with no diagnostic at all, because with
#: an `any` receiver the checker does not know what the container was declared
#: to hold. OP_ELEM_KIND stamps the promise onto the container so the mutation
#: sites can check it.
#:
#: The guard lives in jaiListPush / jaiListInsert / jaiDictSet -- the lowest
#: point every writer funnels through. Guarding the `list.push` builtin instead
#: caught a typed receiver and MISSED an `any` one, which is the case that
#: matters.
import std.io

let xs: list[int] = []
let d: dict[str, int] = {}

fn push_any(l: any, v: any) -> void { l.push(v) }

fn set_any(l: any, i: any, v: any) -> void { l[i] = v }

fn put_any(m: any, k: any, v: any) -> void { m[k] = v }

# Correct values pass, through `any` and directly.
push_any(xs, 1)
xs.push(2)
xs.insert(0, 0)
put_any(d, "a", 1)
d["b"] = 2
print(xs)
print(d["a"] + d["b"])

# Wrong ones are caught, on every path.
try { push_any(xs, "no") } catch e: TypeError { print(f"push: {e.message}") }
try { xs.insert(0, "no") } catch e: TypeError { print(f"insert: {e.message}") }
try { set_any(xs, 0, "no") } catch e: TypeError { print(f"setindex: {e.message}") }
try { put_any(d, "c", "no") } catch e: TypeError { print(f"dictval: {e.message}") }
try { put_any(d, 9, 1) } catch e: TypeError { print(f"dictkey: {e.message}") }

# Nothing was mutated by the rejected writes.
print(xs)
print(d.len())

# An UNDECLARED container promises nothing and still accepts anything.
var loose = []
loose.push(1)
loose.push("mixed")
print(loose)

# A float element accepts an int (LANGUAGE.md §2.5).
let fs: list[float] = []
push_any(fs, 3)
print(fs)
