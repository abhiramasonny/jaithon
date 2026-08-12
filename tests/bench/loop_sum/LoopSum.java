// Same program as loop_sum.jai.
public class LoopSum {
    // BENCH_LEVEL picks how much work this benchmark does: easy a sixteenth of it,
    // medium a quarter, hard (the default, and anything unrecognised) all of it.
    static long scale() {
        String l = System.getenv("BENCH_LEVEL");
        if ("easy".equals(l)) return 16;
        if ("medium".equals(l)) return 4;
        return 1;
    }

    static final long SCALE = scale();
    static final long ITERATIONS = 50000000L / SCALE;

    public static void main(String[] args) {
        long total = 0;
        for (long i = 0; i < ITERATIONS; i++) {
            total += i % 7;
        }
        System.out.println(total);
    }
}
