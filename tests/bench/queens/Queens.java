// Same program as queens.jai.
public class Queens {
    // BENCH_LEVEL picks how much work this benchmark does: easy a sixteenth of it,
    // medium a quarter, hard (the default, and anything unrecognised) all of it.
    static long scale() {
        String l = System.getenv("BENCH_LEVEL");
        if ("easy".equals(l)) return 16;
        if ("medium".equals(l)) return 4;
        return 1;
    }

    static final long SCALE = scale();
    // Scale the repetitions, not the board: work explodes with n.
    static final long REPS = Math.max(1L, 3L / SCALE);

    static boolean safe(long[] cols, long row, long col) {
        for (long r = 0; r < row; r++) {
            long c = cols[(int) r];
            if (c == col) return false;
            if (c - r == col - row) return false;
            if (c + r == col + row) return false;
        }
        return true;
    }

    static long place(long[] cols, long row, long n) {
        if (row == n) return 1;
        long found = 0;
        for (long col = 0; col < n; col++) {
            if (safe(cols, row, col)) {
                cols[(int) row] = col;
                found += place(cols, row + 1, n);
            }
        }
        return found;
    }

    public static void main(String[] args) {
        final int n = 11;
        long[] cols = new long[n];
        long total = 0;
        for (long rep = 0; rep < REPS; rep++) total = place(cols, 0, n);
        System.out.println(n);
        System.out.println(total);
    }
}
