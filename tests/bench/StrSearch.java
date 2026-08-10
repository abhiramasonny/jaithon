// Same program as str_search.jai. The search is written out by hand on purpose:
// indexOf would measure the standard library rather than the per-character
// indexing this benchmark is about.
public class StrSearch {
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
        for (int i = 0; i < 25000; i++) {
            seed = (seed * 1103515245L + 12345L) % 2147483648L;
            builder.append(chunks[(int) ((seed / 65536L) % 8L)]);
        }
        String text = builder.toString();

        long hits = 0;
        for (int rep = 0; rep < 5; rep++) hits = countOccurrences(text, "abcd");
        System.out.println(text.length());
        System.out.println(hits);
    }
}
