// Same program as spectral.jai.
public class Spectral {
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
        final int n = 150;
        double[] u = new double[n];
        for (int i = 0; i < n; i++) u[i] = 1.0;
        double[] v = new double[n];
        for (int r = 0; r < 6; r++) {
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
