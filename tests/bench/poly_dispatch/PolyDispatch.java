// Same program as poly_dispatch.jai.
import java.util.ArrayList;
import java.util.List;

public class PolyDispatch {
    static long scale() {
        String l = System.getenv("BENCH_LEVEL");
        if ("easy".equals(l)) return 16;
        if ("medium".equals(l)) return 4;
        return 1;
    }

    static final long SCALE = scale();
    static final int OPS = 512;
    static final int REPS = (int) Math.max(8000 / SCALE, 1);
    static final long M = 1000003L;

    interface Op {
        long apply(long x);
    }

    static final class AddK implements Op {
        final long k;
        AddK(long k) { this.k = k; }
        public long apply(long x) { return (x + k) % M; }
    }
    static final class MulK implements Op {
        final long k;
        MulK(long k) { this.k = k; }
        public long apply(long x) { return (x * k) % M; }
    }
    static final class SquareK implements Op {
        final long k;
        SquareK(long k) { this.k = k; }
        public long apply(long x) { return (x * x + k) % M; }
    }
    static final class DoubleK implements Op {
        final long k;
        DoubleK(long k) { this.k = k; }
        public long apply(long x) { return (x * 2 + k) % M; }
    }
    static final class DivK implements Op {
        final long k;
        DivK(long k) { this.k = k; }
        public long apply(long x) { return (x + x / k) % M; }
    }
    static final class ModK implements Op {
        final long k;
        ModK(long k) { this.k = k; }
        public long apply(long x) { return (x + x % k) % M; }
    }
    static final class FlipK implements Op {
        final long k;
        FlipK(long k) { this.k = k; }
        public long apply(long x) { return (M - 1 - x + k) % M; }
    }
    static final class MixK implements Op {
        final long k;
        MixK(long k) { this.k = k; }
        public long apply(long x) { return (x + x / 3 + k) % M; }
    }

    static List<Op> build(int n) {
        List<Op> ops = new ArrayList<>();
        long s = 7;
        for (int i = 0; i < n; i++) {
            s = (s * 1103515245L + 12345L) % 2147483648L;
            long which = (s / 65536) % 8;
            long k = s % 97 + 2;
            if (which == 0) ops.add(new AddK(k));
            else if (which == 1) ops.add(new MulK(k));
            else if (which == 2) ops.add(new SquareK(k));
            else if (which == 3) ops.add(new DoubleK(k));
            else if (which == 4) ops.add(new DivK(k));
            else if (which == 5) ops.add(new ModK(k));
            else if (which == 6) ops.add(new FlipK(k));
            else ops.add(new MixK(k));
        }
        return ops;
    }

    public static void main(String[] args) {
        List<Op> ops = build(OPS);
        long acc = 1;
        long check = 0;
        for (int r = 0; r < REPS; r++) {
            acc = (acc + r) % M;
            for (Op op : ops) acc = op.apply(acc);
            check = (check + acc) % M;
        }
        System.out.println(ops.size());
        System.out.println(acc);
        System.out.println(check);
    }
}
