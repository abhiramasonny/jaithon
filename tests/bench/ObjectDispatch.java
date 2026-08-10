// Same program as object_dispatch.jai.
import java.util.Locale;

public class ObjectDispatch {
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
        for (int i = 0; i < 1000000; i++) {
            acc = acc.add(step);
            total += acc.dot(step);
        }
        System.out.printf(Locale.ROOT, "%.1f\n", total);
    }
}
