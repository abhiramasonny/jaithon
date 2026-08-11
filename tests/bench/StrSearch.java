// Same program as str_search.jai. The search is written out by hand on purpose:
// indexOf would measure the standard library rather than the per-character
// indexing this benchmark is about.
public class StrSearch {
    // BENCH_LEVEL picks how much work this benchmark does: easy a sixteenth of it,
    // medium a quarter, hard (the default, and anything unrecognised) all of it.
    static long scale() {
        String l = System.getenv("BENCH_LEVEL");
        if ("easy".equals(l)) return 16;
        if ("medium".equals(l)) return 4;
        return 1;
    }

    static final long SCALE = scale();
    // Scale the repetitions and leave the text size alone.
    static final long REPS = Math.max(1L, 5L / SCALE);

    static long countOccurrences(String text, String needle) {
        long n = text.length();
        long m = needle.length();
        char first = needle.charAt(0);
        long hits = 0;
        long i = 0;
        while (i + m <= n) {
            if (text.charAt((int) i) == first) {
                long k = 1;
                while (k < m && text.charAt((int) (i + k)) == needle.charAt((int) k)) k += 1;
                if (k == m) hits += 1;
            }
            i += 1;
        }
        return hits;
    }

    public static void main(String[] args) {
        String[] chunks = {"abcdbadc", "bcadcbda", "cdabacbd", "dacbdabc",
                           "abdcadbc", "bdacbcad", "cabdbdca", "dbcaabcd"};
        StringBuilder builder = new StringBuilder();
        long seed = 7;
        for (int i = 0; i < 250000; i++) {
            seed = (seed * 1103515245L + 12345L) % 2147483648L;
            builder.append(chunks[(int) ((seed / 65536L) % 8L)]);
        }
        String text = builder.toString();

        long hits = 0;
        for (long rep = 0; rep < REPS; rep++) hits = countOccurrences(text, "abcd");
        System.out.println(text.length());
        System.out.println(hits);
    }
}
