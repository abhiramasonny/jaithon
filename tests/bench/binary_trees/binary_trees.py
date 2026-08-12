import os

_LEVEL = os.environ.get("BENCH_LEVEL", "hard")
# Work is exponential in the depth, so the level subtracts from it instead of
# dividing it: -2 is 4x less tree, -4 is 16x less.
DEPTH_DROP = 4 if _LEVEL == "easy" else 2 if _LEVEL == "medium" else 0
DEPTH = 18 - DEPTH_DROP


class Node:
    __slots__ = ("value", "left", "right")

    def __init__(self, value, left, right):
        self.value = value
        self.left = left
        self.right = right


def build(depth, value):
    if depth == 0:
        return None
    return Node(value, build(depth - 1, value * 2), build(depth - 1, value * 2 + 1))


def walk(node):
    if node is None:
        return 0
    return node.value + walk(node.left) + walk(node.right)


depth = DEPTH
total = 0
for _rep in range(8):
    tree = build(depth, 1)
    total = (total + walk(tree)) % 1000000007
print(depth)
print(total)
