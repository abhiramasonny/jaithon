// Same program as alloc_churn.jai.
public class AllocChurn {
    static final class Point {
        final long x, y;
        Point(long x, long y) { this.x = x; this.y = y; }
        long dot(Point o) { return x * o.x + y * o.y; }
    }

    public static void main(String[] args) {
        long total = 0;
        for (long i = 0; i < 4000000L; i++) {
            Point a = new Point(i % 100, i % 37);
            Point b = new Point(i % 53, i % 11);
            total = (total + a.dot(b)) % 1000000007L;
        }
        System.out.println(total);
    }
}
