// Same program as sieve.jai.
public class Sieve {
    public static void main(String[] args) {
        final int n = 10000000;
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
