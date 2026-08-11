#: A binary tree built, walked, and dropped.
#:
#: `alloc_churn` allocates objects that point at nothing, so its live set never
#: grows and there is never anything to trace. This holds a whole tree alive at
#: once and then chases the pointers back down it, which is the other half of
#: what a collector costs: the walk is a load whose address the previous load
#: produced, and nothing about it prefetches.
from std.os import env_or

let LEVEL = env_or("BENCH_LEVEL", "hard")
# Work is exponential in the depth, so the level subtracts from it rather than
# dividing: the repetition count stays at 8 so the shape is unchanged.
let DEPTH_DROP = if LEVEL == "easy" { 4 } elif LEVEL == "medium" { 2 } else { 0 }
let DEPTH = 18 - DEPTH_DROP

class Node {
    pub var value: int
    pub var left: Node?
    pub var right: Node?

    fn init(self, value: int, left: Node?, right: Node?) {
        self.value = value
        self.left = left
        self.right = right
    }
}

fn build(depth: int, value: int) -> Node? {
    if depth == 0 { return null }
    return Node(value, build(depth - 1, value * 2), build(depth - 1, value * 2 + 1))
}

fn walk(node: Node?) -> int {
    if node == null { return 0 }
    return node.value + walk(node.left) + walk(node.right)
}

fn main() -> int {
    let depth = DEPTH
    var total = 0
    for _rep in 0..8 {
        let tree = build(depth, 1)
        total = (total + walk(tree)) % 1000000007
    }
    print(depth)
    print(total)
    return 0
}
