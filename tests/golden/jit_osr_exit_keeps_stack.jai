#: A branch out of a compiled loop must hand the interpreter its operand stack.
#:
#: The OSR tier compiles a loop from its head to its LAST back edge, and when
#: that fails it falls back to the prefix up to the FIRST one -- here a
#: labelled `continue` out of a nested `for x in xs`. Whatever lies past the
#: prefix is reached by branching out of the region, and the stub for such a
#: branch handed back the locals and nothing else. That is right only while
#: the model's operand stack is empty, and inside `for x in xs` it is not: the
#: loop's iterator lives there. So the branch past the `continue` dropped it,
#: the interpreter ran the rest of the body one entry short, and the inner
#: loop's next OP_FOR_ITER_BIND peeked whatever lay below:
#:
#:     TypeError: for-loop expected an iterator, not 'null'
#:
#: for_range_bind.jai's `mixed` hit it about one run in thirty under
#: --gc-stress, which mattered only because a collection sends every back edge
#: through the safepoint that compiles the outer head. Nothing about it needs
#: a collection. What it needs is the outer head compiled, and here the
#: program's warm-up tick (JAITHON_JIT_MAIN_EDGES, a count, not a timer) does
#: that: it lands on the nested range loop, which compiles and asks for the
#: loop around it. Every run of this file raised before the fix.
#:
#: First in the file on purpose: the warm-up counts back edges from the start
#: of the module body.

#: The `continue` is the first back edge to the outer head, so the prefix ends
#: right after it and `total +=` is reachable only by branching out.
fn skip_rest(n: int) -> int {
    var total = 0
    'outer: for i in 0..n {
        var s = 0
        for k in 0..16 { s += k }
        for x in [1, 2, 3] {
            if x == 3 { continue 'outer }
            total += i + x + s
        }
    }
    return total
}
print(skip_rest(20000))

#: The reduced golden: a `break` out of the same nest as well, whose OP_POP
#: after the `continue` is what stops the whole-body compile.
fn skip_or_stop(n: int) -> list[int] {
    var acc: list[int] = []
    var s = 0
    'other: for i in 0..n {
        for k in 0..16 { s += k }
        for name in [7, 8, 9] {
            if name == 8 { continue 'other }
            if i == n - 1 { break 'other }
            acc.push(i * 10 + name)
        }
    }
    return [acc.len(), acc[0], acc[acc.len() - 1], s]
}
print(skip_or_stop(20000))

#: Two iterators: the outer loop's own is below the model, the inner one is on
#: it, and only the inner one may come back from the branch out.
fn nested_lists(rows: list[int]) -> int {
    var total = 0
    'row: for r in rows {
        var s = 0
        for k in 0..16 { s += k }
        for c in [1, 2, 3, 4] {
            if c == 4 { continue 'row }
            total += r * c + s
        }
    }
    return total
}
var rows: list[int] = []
for r in 0..20000 { rows.push(r % 13) }
print(nested_lists(rows))
