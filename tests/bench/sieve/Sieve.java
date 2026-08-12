// Same program as sieve.jai.
public class Sieve {
    // BENCH_LEVEL picks how much work this benchmark does: easy a sixteenth of it,
    // medium a quarter, hard (the default, and anything unrecognised) all of it.
    static long scale() {
        String l = System.getenv("BENCH_LEVEL");
        if ("easy".equals(l)) return 16;
        if ("medium".equals(l)) return 4;
        return 1;
    }

    static final long SCALE = scale();
    static final int LIMIT = (int) (10000000L / SCALE);

    public static void main(String[] args) {
        final int n = LIMIT;
        boolean[] flags = new boolean[n + 1];
        for (int i = 0; i <= n; i++) flags[i] = true;

        long count = 0;
        long total = 0;
        for (long p = 2; p * p <= n; p++) {
            if (flags[(int) p]) {
                for (long q = p * p; q <= n; q += p) {
                    flags[(int) q] = false;
                }
            }
        }

        for (int i = 2; i <= n; i++) {
            if (flags[i]) {
                count += 1;
                total = (total + i) % 1000000007L;
            }
        }
        System.out.println(count);
        System.out.println(total);
    }
}
