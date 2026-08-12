// Same program as binary_trees.jai.
public class BinaryTrees {
    // BENCH_LEVEL picks how much work this benchmark does: easy a sixteenth of it,
    // medium a quarter, hard (the default, and anything unrecognised) all of it.
    static long scale() {
        String l = System.getenv("BENCH_LEVEL");
        if ("easy".equals(l)) return 16;
        if ("medium".equals(l)) return 4;
        return 1;
    }

    static final long SCALE = scale();
    // Work is exponential in the depth, so subtract rather than divide.
    static final int DEPTH = (int) (18L - (SCALE == 16 ? 4L : SCALE == 4 ? 2L : 0L));

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
        final int depth = DEPTH;
        long total = 0;
        for (int rep = 0; rep < 8; rep++) {
            Node tree = build(depth, 1);
            total = (total + walk(tree)) % 1000000007L;
        }
        System.out.println(depth);
        System.out.println(total);
    }
}
