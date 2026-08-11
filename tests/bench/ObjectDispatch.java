// Same program as object_dispatch.jai.
import java.util.Locale;

public class ObjectDispatch {
    // BENCH_LEVEL picks how much work this benchmark does: easy a sixteenth of it,
    // medium a quarter, hard (the default, and anything unrecognised) all of it.
    static long scale() {
        String l = System.getenv("BENCH_LEVEL");
        if ("easy".equals(l)) return 16;
        if ("medium".equals(l)) return 4;
        return 1;
    }

    static final long SCALE = scale();
    static final long ITERATIONS = 10000000L / SCALE;

    static final class Vec2 {
        final double x, y;

        Vec2(double x, double y) {
            this.x = x;
            this.y = y;
        }

        Vec2 add(Vec2 other) { return new Vec2(x + other.x, y + other.y); }

        double dot(Vec2 other) { return x * other.x + y * other.y; }
    }

    public static void main(String[] args) {
        Vec2 acc = new Vec2(0.0, 0.0);
        Vec2 step = new Vec2(1.0, 2.0);
        double total = 0.0;
        for (long i = 0; i < ITERATIONS; i++) {
            acc = acc.add(step);
            total += acc.dot(step);
        }
        System.out.printf(Locale.ROOT, "%.1f\n", total);
    }
}
