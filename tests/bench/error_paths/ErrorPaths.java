// Same program as error_paths.jai.
public class ErrorPaths {
    static long scale() {
        String l = System.getenv("BENCH_LEVEL");
        if ("easy".equals(l)) return 16;
        if ("medium".equals(l)) return 4;
        return 1;
    }

    static final long SCALE = scale();
    static final long CALM = Math.max(8_000_000L / SCALE, 1);
    static final long STORM = Math.max(160_000L / SCALE, 1);

    static class BadValue extends RuntimeException {
        BadValue(String msg) { super(msg); }
    }

    static long checked(long x) {
        if (x % 8 == 3) throw new BadValue("bad value " + x);
        return x + 1;
    }

    static long calm(long n) {
        long total = 0;
        long i = 0;
        while (i < n) {
            try {
                total += i % 13;
            } catch (BadValue e) {
                total -= 1;
            }
            i += 1;
        }
        return total;
    }

    static long storm(long n) {
        long ok = 0;
        long bad = 0;
        long i = 0;
        while (i < n) {
            try {
                ok += checked(i);
            } catch (BadValue e) {
                bad += 1;
            }
            i += 1;
        }
        return ok - bad;
    }

    public static void main(String[] args) {
        System.out.println(calm(CALM));
        System.out.println(storm(STORM));
    }
}
