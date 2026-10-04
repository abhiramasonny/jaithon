#: The collector keeps running in a process that compiles again and again.
#:
#: Loading the self-hosted front end pauses collection and then credits the
#: bytes it built to the collector. Every eval, REPL line, test case and
#: `check` file passes through that code again with the front end already
#: loaded. When the credit was "4x the heap as it stands", garbage included,
#: each of those calls pushed the next collection further out and the
#: collector never ran: `check lib` went from 15 collections to 0 and from
#: 0.9GB to 3.3GB. This makes garbage past the budget the collector reported,
#: evaluating as it goes, and expects a collection before three budgets.
from std.test import assert_true

fn test_eval_does_not_postpone_the_next_collection() -> void {
    let _ = __prim__.eval("1 + 1")
    let start = __prim__.gc_stats()
    let startCollections = start["collections"]
    let budget = start["next_gc"] - start["bytes_allocated"]
    let limit = budget * 3 + 8 * 1024 * 1024
    var made = 0
    var collected = false
    var round = 0
    while not collected and made < limit {
        let before = __prim__.gc_stats()["bytes_allocated"]
        var junk = []
        for i in 0..64 { junk.push(str(round * 64 + i) * 400) }
        let after = __prim__.gc_stats()["bytes_allocated"]
        if after > before { made = made + (after - before) }
        let _ = __prim__.eval("1 + 1")
        collected = __prim__.gc_stats()["collections"] > startCollections
        round = round + 1
    }
    assert_true(
        collected,
        f"no collection after {made} bytes of garbage against a budget of {budget}"
    )
}
