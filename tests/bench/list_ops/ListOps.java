// Same program as list_ops.jai.
import java.util.ArrayList;
import java.util.List;

public class ListOps {
    // BENCH_LEVEL picks how much work this benchmark does: easy a sixteenth of it,
    // medium a quarter, hard (the default, and anything unrecognised) all of it.
    static long scale() {
        String l = System.getenv("BENCH_LEVEL");
        if ("easy".equals(l)) return 16;
        if ("medium".equals(l)) return 4;
        return 1;
    }

    static final long SCALE = scale();
    static final long N = 10000000L / SCALE;
    static final long PROBE = 999999L / SCALE;

    public static void main(String[] args) {
        List<Long> xs = new ArrayList<>();
        for (long i = 0; i < N; i++) {
            xs.add(i * 3);
        }

        long total = 0;
        for (long x : xs) {
            total += x % 11;
        }

        List<Long> doubled = new ArrayList<>(xs.size());
        for (long x : xs) {
            doubled.add(x * 2);
        }

        System.out.println(total);
        System.out.println(doubled.get((int) PROBE));
    }
}
