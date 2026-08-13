// Same program as graph_bfs.jai.
import java.util.ArrayList;
import java.util.List;

public class GraphBfs {
    static long scale() {
        String l = System.getenv("BENCH_LEVEL");
        if ("easy".equals(l)) return 16;
        if ("medium".equals(l)) return 4;
        return 1;
    }

    static final long SCALE = scale();
    static final int NODES = 80_000;
    static final int DEGREE = 6;
    static final int REPS = (int) Math.max(6 / SCALE, 1);

    static List<List<Long>> build(int n, int deg) {
        List<List<Long>> g = new ArrayList<>();
        long s = 7;
        for (int i = 0; i < n; i++) {
            List<Long> adj = new ArrayList<>();
            for (int d = 0; d < deg; d++) {
                s = (s * 1103515245L + 12345L) % 2147483648L;
                adj.add(s % n);
            }
            g.add(adj);
        }
        return g;
    }

    static long bfs(List<List<Long>> g, int n, int start) {
        List<Long> dist = new ArrayList<>();
        for (int i = 0; i < n; i++) dist.add(-1L);
        List<Long> queue = new ArrayList<>();
        queue.add((long) start);
        dist.set(start, 0L);
        int head = 0;
        long total = 0;
        while (head < queue.size()) {
            int node = (int) (long) queue.get(head);
            head += 1;
            long d = dist.get(node);
            total += d;
            for (long nb : g.get(node)) {
                if (dist.get((int) nb) < 0) {
                    dist.set((int) nb, d + 1);
                    queue.add(nb);
                }
            }
        }
        return total;
    }

    public static void main(String[] args) {
        final int n = NODES;
        List<List<Long>> g = build(n, DEGREE);
        long total = 0;
        for (int r = 0; r < REPS; r++) total = bfs(g, n, 0);
        System.out.println(n);
        System.out.println(total);
    }
}
