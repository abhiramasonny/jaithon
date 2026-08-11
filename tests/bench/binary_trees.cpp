// Same program as binary_trees.jai. See tests/bench/README.md for why the C++
// and Java rows are here and what they are not. The children are unique_ptr so
// that dropping the root frees the tree, which is what the .jai version does.
#include <cstdint>
#include <cstdio>
#include <memory>
#include <utility>

struct Node {
    int64_t value;
    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;
    Node(int64_t value, std::unique_ptr<Node> left, std::unique_ptr<Node> right)
        : value(value), left(std::move(left)), right(std::move(right)) {}
};

static std::unique_ptr<Node> build(int64_t depth, int64_t value) {
    if (depth == 0) return nullptr;
    std::unique_ptr<Node> left = build(depth - 1, value * 2);
    std::unique_ptr<Node> right = build(depth - 1, value * 2 + 1);
    return std::make_unique<Node>(value, std::move(left), std::move(right));
}

static int64_t walk(const Node *node) {
    if (node == nullptr) return 0;
    return node->value + walk(node->left.get()) + walk(node->right.get());
}

int main() {
    const int64_t depth = 18;
    int64_t total = 0;
    for (int rep = 0; rep < 8; rep++) {
        std::unique_ptr<Node> tree = build(depth, 1);
        total = (total + walk(tree.get())) % 1000000007;
    }
    std::printf("%lld\n", (long long)depth);
    std::printf("%lld\n", (long long)total);
    return 0;
}
