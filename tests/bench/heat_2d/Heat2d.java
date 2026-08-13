// Same program as heat_2d.jai.
public class Heat2d {
    static long shrink() {
        String l = System.getenv("BENCH_LEVEL");
        if ("easy".equals(l)) return 4;
        if ("medium".equals(l)) return 2;
        return 1;
    }

    static final long SHRINK = shrink();
    static final int SIDE = (int) (300 / SHRINK);
    static final int STEPS = (int) Math.max(40 / SHRINK, 1);

    static double[][] plate(int n) {
        double[][] g = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                g[i][j] = i == 0 ? 100.0 : 0.0;
            }
        }
        return g;
    }

    static void relax(double[][] src, double[][] dst, int n) {
        for (int i = 1; i < n - 1; i++) {
            double[] up = src[i - 1];
            double[] mid = src[i];
            double[] down = src[i + 1];
            double[] out = dst[i];
            for (int j = 1; j < n - 1; j++) {
                out[j] = 0.25 * (up[j] + down[j] + mid[j - 1] + mid[j + 1]);
            }
        }
    }

    public static void main(String[] args) {
        final int n = SIDE;
        double[][] a = plate(n);
        double[][] b = plate(n);

        int s = 0;
        while (s < STEPS) {
            relax(a, b, n);
            relax(b, a, n);
            s += 2;
        }

        double total = 0.0;
        for (int i = 0; i < n; i++) {
            double[] row = a[i];
            for (int j = 0; j < n; j++) total += row[j];
        }
        System.out.println(n);
        System.out.printf("%.6f%n", total);
    }
}
