// Same program as spectral.jai.
public class Spectral {
    // BENCH_LEVEL picks how much work this benchmark does: easy a sixteenth of it,
    // medium a quarter, hard (the default, and anything unrecognised) all of it.
    static long scale() {
        String l = System.getenv("BENCH_LEVEL");
        if ("easy".equals(l)) return 16;
        if ("medium".equals(l)) return 4;
        return 1;
    }

    static final long SCALE = scale();
    // Scale the repetitions and leave the matrix size alone.
    static final long ROUNDS = Math.max(1L, 6L / SCALE);

    static double evalA(long i, long j) {
        long s = i + j;
        return 1.0 / (double) (s * (s + 1) / 2 + i + 1);
    }

    static double[] timesA(double[] v, int n) {
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) sum += evalA(i, j) * v[j];
            out[i] = sum;
        }
        return out;
    }

    static double[] timesAt(double[] v, int n) {
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) sum += evalA(j, i) * v[j];
            out[i] = sum;
        }
        return out;
    }

    public static void main(String[] args) {
        final int n = 550;
        double[] u = new double[n];
        for (int i = 0; i < n; i++) u[i] = 1.0;
        double[] v = new double[n];
        for (long r = 0; r < ROUNDS; r++) {
            v = timesAt(timesA(u, n), n);
            u = timesAt(timesA(v, n), n);
        }

        double vBv = 0.0;
        double vv = 0.0;
        for (int i = 0; i < n; i++) {
            vBv += u[i] * v[i];
            vv += v[i] * v[i];
        }
        System.out.println(n);
        System.out.printf("%.9f%n", Math.pow(vBv / vv, 0.5));
    }
}
