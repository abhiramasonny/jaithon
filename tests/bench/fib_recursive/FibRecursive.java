// Same program as fib_recursive.jai.
public class FibRecursive {
    // BENCH_LEVEL picks how much work this benchmark does: easy a sixteenth of it,
    // medium a quarter, hard (the default, and anything unrecognised) all of it.
    static long scale() {
        String l = System.getenv("BENCH_LEVEL");
        if ("easy".equals(l)) return 16;
        if ("medium".equals(l)) return 4;
        return 1;
    }

    static final long SCALE = scale();
    // Work is exponential in n, so subtract rather than divide.
    static final long N = 38L - (SCALE == 16 ? 4L : SCALE == 4 ? 2L : 0L);

    static long fib(long n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }

    public static void main(String[] args) {
        System.out.println(fib(N));
    }
}
