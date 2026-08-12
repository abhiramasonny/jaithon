// Same program as alloc_churn.jai.
public class AllocChurn {
    // BENCH_LEVEL picks how much work this benchmark does: easy a sixteenth of it,
    // medium a quarter, hard (the default, and anything unrecognised) all of it.
    static long scale() {
        String l = System.getenv("BENCH_LEVEL");
        if ("easy".equals(l)) return 16;
        if ("medium".equals(l)) return 4;
        return 1;
    }

    static final long SCALE = scale();
    static final long ITERATIONS = 4000000L / SCALE;

    static final class Point {
        final long x, y;
        Point(long x, long y) { this.x = x; this.y = y; }
        long dot(Point o) { return x * o.x + y * o.y; }
    }

    public static void main(String[] args) {
        long total = 0;
        for (long i = 0; i < ITERATIONS; i++) {
            Point a = new Point(i % 100, i % 37);
            Point b = new Point(i % 53, i % 11);
            total = (total + a.dot(b)) % 1000000007L;
        }
        System.out.println(total);
    }
}
