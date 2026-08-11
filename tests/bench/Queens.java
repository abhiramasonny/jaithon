// Same program as queens.jai.
public class Queens {
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
        for (int rep = 0; rep < 3; rep++) total = place(cols, 0, n);
        System.out.println(n);
        System.out.println(total);
    }
}
