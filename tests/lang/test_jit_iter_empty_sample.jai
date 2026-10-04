#: A loop whose container had nothing in it when the function compiled.
#:
#: The function tier samples an element to decide the loop variable's kind.
#: The call that crosses the threshold decides what is sampled, and for a tree
#: walk or a "for each generic parameter" loop that is usually an EMPTY list --
#: the whole function used to decline for it. Now (JAITHON_JIT_ITER_SOFT) the
#: rest of the body compiles and the loop is left to the interpreter; and
#: (JAITHON_JIT_ITER_EMPTY_SKIP) a list that is still empty when the compiled
#: code reaches it skips straight to the loop's exit without deoptimising.
#:
#: Each function below is warmed on empty containers first, so it compiles
#: against no sample, and then called on full ones: the loop must run, run
#: exactly once per element, and leave the right totals behind.
from std.test import assert_eq

let WARM = 300

class Tree {
    pub var kind: int
    pub var value: int
    pub var kids: list[Tree]
    pub var named: dict[str, Tree]

    fn init(self, kind: int, value: int) {
        self.kind = kind
        self.value = value
        self.kids = []
        self.named = {}
    }
}

fn total(node: Tree) -> int {
    if node.kind == 0 { return node.value }
    if node.kind == 1 {
        var sum = node.value
        for kid in node.kids { sum += total(kid) }
        return sum
    }
    var sum = 0
    for (name, kid) in node.named.items() { sum += name.len() + total(kid) }
    return sum
}

fn count_after(xs: list[int], tail: int) -> int {
    var n = 0
    for x in xs { n += x }
    return n * 10 + tail
}

fn nested(rows: list[list[int]]) -> int {
    var n = 0
    for row in rows {
        for x in row { n += x }
        n += 1000
    }
    return n
}

pub fn test_a_tree_walk_warmed_on_leaves() -> void {
    let leaf = Tree(0, 7)
    var warm = 0
    for _w in 0..WARM { warm += total(leaf) }
    assert_eq(warm, 7 * WARM, "leaves")

    var root = Tree(1, 1)
    for i in 0..10 {
        var mid = Tree(1, i)
        for j in 0..3 { mid.kids.push(Tree(0, j)) }
        root.kids.push(mid)
    }
    var named = Tree(2, 0)
    named.named["alpha"] = Tree(0, 5)
    named.named["be"] = root
    root.kids.push(Tree(1, 100))
    # root: 1 + sum(i + 0+1+2 for i in 0..10) + 100 = 1 + 45 + 30 + 100
    assert_eq(total(root), 176, "the list branch")
    assert_eq(total(named), 5 + 5 + 2 + 176, "the dict branch")
}

pub fn test_empty_then_full() -> void {
    var warm = 0
    for _w in 0..WARM { warm += count_after([], 3) }
    assert_eq(warm, 3 * WARM, "empty")
    assert_eq(count_after([1, 2, 3], 4), 64, "full")
    assert_eq(count_after([], 5), 5, "empty again")
    assert_eq(count_after([9], 0), 90, "one element")
}

pub fn test_nested_loops_warmed_empty() -> void {
    var warm = 0
    for _w in 0..WARM { warm += nested([]) }
    assert_eq(warm, 0, "nothing")
    assert_eq(nested([[], [1, 2], []]), 3 + 3000, "inner empty and full")
    assert_eq(nested([[5]]), 1005, "one row")
}
