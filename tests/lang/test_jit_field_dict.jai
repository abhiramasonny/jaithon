#: A declared `dict` field read off an instance the body built itself.
#:
#: The tier reads a field's kind off the live receiver, and a receiver made in
#: the same body has none at compile time; only the declaration is left. A
#: `dict[K, V]` field used to decline the whole function there, which is every
#: hand-written parser's object case: `node.fields[key] = self.value()`. It is
#: now predicted (JAITHON_JIT_FIELD_DICT): tag and OBJ_DICT guarded, sampled by
#: an empty exemplar dict that nothing compiled ever holds.
#:
#: Checked here: the writes land in the right dict, the dict methods answer
#: through the predicted entry, a collection in the middle does not lose the
#: exemplar, and the results match a version with no field in between.
# jit-compiles: build, fill_from, sizes
from std.test import assert_eq, assert_true

let CALLS = 300

class Node {
    pub var kind: int
    pub var fields: dict[str, int]
    pub var kids: dict[str, Node]

    fn init(self, kind: int) {
        self.kind = kind
        self.fields = {}
        self.kids = {}
    }
}

fn build(n: int) -> Node {
    var node = Node(1)
    var i = 0
    while i < n {
        node.fields[f"k{i}"] = i * 2
        i += 1
    }
    return node
}

fn fill_from(keys: list[str]) -> Node {
    var node = Node(2)
    for key in keys {
        node.fields[key] = key.len()
        var kid = Node(3)
        kid.fields["depth"] = 1
        node.kids[key] = kid
    }
    return node
}

fn sizes(node: Node) -> int {
    var fresh = Node(4)
    fresh.fields["a"] = 1
    fresh.fields["b"] = 2
    var total = fresh.fields.len()
    if "a" in fresh.fields { total += 10 }
    total += fresh.fields.get("b", 0)
    total += fresh.fields.get("zz", 100)
    return total + node.fields.len()
}

pub fn test_writes_land_in_the_new_nodes_dict() -> void {
    var last = Node(0)
    for _c in 0..CALLS { last = build(5) }
    assert_eq(last.fields.len(), 5, "five keys")
    assert_eq(last.fields["k4"], 8, "the last value")
    assert_eq(last.kind, 1, "the kind")
}

pub fn test_nested_nodes() -> void {
    var node = Node(0)
    let keys = ["alpha", "be", "c"]
    for _c in 0..CALLS { node = fill_from(keys) }
    assert_eq(node.fields["alpha"], 5, "a length")
    assert_eq(node.kids.len(), 3, "three kids")
    assert_eq(node.kids["be"].fields["depth"], 1, "a kid's field")
}

pub fn test_dict_methods_through_the_prediction() -> void {
    var total = 0
    let base = build(3)
    for _c in 0..CALLS { total = sizes(base) }
    assert_eq(total, 2 + 10 + 2 + 100 + 3, "len, in, get, get-default")
}

pub fn test_a_collection_between_calls() -> void {
    var keep: list[Node] = []
    for c in 0..CALLS {
        keep.push(build(c % 7))
        # Garbage between calls, so collections land while compiled code runs.
        var junk: list[str] = []
        for j in 0..50 { junk.push(f"junk{j}") }
    }
    var total = 0
    for n in keep { total += n.fields.len() }
    var expect = 0
    for c in 0..CALLS { expect += c % 7 }
    assert_eq(total, expect, "every dict kept its keys")
}
