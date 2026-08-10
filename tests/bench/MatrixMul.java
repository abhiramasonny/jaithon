// Same program as matrix_mul.jai.
public class MatrixMul {
    static double[][] make(int n, long seed) {
        double[][] m = new double[n][n];
        long s = seed;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                s = (s * 1103515245L + 12345L) % 2147483648L;
                m[i][j] = (double) (s % 1000L) / 1000.0;
            }
        }
        return m;
    }

    public static void main(String[] args) {
        final int n = 120;
        double[][] a = make(n, 12345L);
        double[][] b = make(n, 67890L);

        double[][] c = new double[n][n];
        for (int i = 0; i < n; i++) {
            double[] ai = a[i];
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += ai[k] * b[k][j];
                }
                c[i][j] = sum;
            }
        }

        double trace = 0.0;
        for (int i = 0; i < n; i++) trace += c[i][i];
        System.out.println(n);
        System.out.printf("%.6f%n", trace);
    }
}
