// Same program as life.jai.
public class Life {
    static long[] zeros(int n) {
        return new long[n];
    }

    static long[][] seedBoard(int h, int w, long seed) {
        long[][] board = new long[h + 2][];
        long s = seed;
        board[0] = zeros(w + 2);
        for (int r = 0; r < h; r++) {
            long[] row = zeros(w + 2);
            for (int c = 0; c < w; c++) {
                s = (s * 1103515245L + 12345L) % 2147483648L;
                long bits = (s / 65536L) % 100L;
                long alive = 0;
                if (bits < 35L) alive = 1;
                row[c + 1] = alive;
            }
            board[r + 1] = row;
        }
        board[h + 1] = zeros(w + 2);
        return board;
    }

    static long[][] step(long[][] board, int h, int w) {
        long[][] out = new long[h + 2][];
        out[0] = zeros(w + 2);
        for (int r = 1; r < h + 1; r++) {
            long[] up = board[r - 1];
            long[] mid = board[r];
            long[] down = board[r + 1];
            long[] row = zeros(w + 2);
            for (int c = 1; c < w + 1; c++) {
                long n = up[c - 1] + up[c] + up[c + 1];
                n += mid[c - 1] + mid[c + 1];
                n += down[c - 1] + down[c] + down[c + 1];
                long alive = 0;
                if (n == 3) {
                    alive = 1;
                } else if (n == 2) {
                    alive = mid[c];
                }
                row[c] = alive;
            }
            out[r] = row;
        }
        out[h + 1] = zeros(w + 2);
        return out;
    }

    public static void main(String[] args) {
        final int h = 100;
        final int w = 100;
        final int generations = 50;

        long[][] board = seedBoard(h, w, 7L);
        for (int g = 0; g < generations; g++) board = step(board, h, w);

        long live = 0;
        for (int r = 1; r < h + 1; r++) {
            long[] row = board[r];
            for (int c = 1; c < w + 1; c++) live += row[c];
        }
        System.out.println(generations);
        System.out.println(live);
    }
}
