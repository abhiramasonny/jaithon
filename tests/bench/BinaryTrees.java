// Same program as binary_trees.jai.
public class BinaryTrees {
    static final class Node {
        final long value;
        final Node left, right;
        Node(long value, Node left, Node right) {
            this.value = value;
            this.left = left;
            this.right = right;
        }
    }

    static Node build(long depth, long value) {
        if (depth == 0) return null;
        return new Node(value, build(depth - 1, value * 2), build(depth - 1, value * 2 + 1));
    }

    static long walk(Node node) {
        if (node == null) return 0;
        return node.value + walk(node.left) + walk(node.right);
    }

    public static void main(String[] args) {
        final int depth = 16;
        long total = 0;
        for (int rep = 0; rep < 8; rep++) {
            Node tree = build(depth, 1);
            total = (total + walk(tree)) % 1000000007L;
        }
        System.out.println(depth);
        System.out.println(total);
    }
}
