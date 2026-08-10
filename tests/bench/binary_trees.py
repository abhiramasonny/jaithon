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


depth = 16
total = 0
for _rep in range(8):
    tree = build(depth, 1)
    total = (total + walk(tree)) % 1000000007
print(depth)
print(total)
