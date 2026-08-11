// Same program as sort_merge.jai. Hand-written merge sort, not
// Collections.sort, for the same reason the original avoids `list.sort()`.
import java.util.ArrayList;
import java.util.List;

public class SortMerge {
    // BENCH_LEVEL picks how much work this benchmark does: easy a sixteenth of it,
    // medium a quarter, hard (the default, and anything unrecognised) all of it.
    static long scale() {
        String l = System.getenv("BENCH_LEVEL");
        if ("easy".equals(l)) return 16;
        if ("medium".equals(l)) return 4;
        return 1;
    }

    static final long SCALE = scale();
    static final long N = 1000000L / SCALE;

    static List<Long> merge(List<Long> left, List<Long> right) {
        List<Long> out = new ArrayList<>();
        int i = 0;
        int j = 0;
        while (i < left.size() && j < right.size()) {
            if (left.get(i) <= right.get(j)) {
                out.add(left.get(i));
                i += 1;
            } else {
                out.add(right.get(j));
                j += 1;
            }
        }
        while (i < left.size()) {
            out.add(left.get(i));
            i += 1;
        }
        while (j < right.size()) {
            out.add(right.get(j));
            j += 1;
        }
        return out;
    }

    static List<Long> sort(List<Long> values) {
        if (values.size() <= 1) return values;
        int mid = values.size() / 2;
        List<Long> left = new ArrayList<>(values.subList(0, mid));
        List<Long> right = new ArrayList<>(values.subList(mid, values.size()));
        return merge(sort(left), sort(right));
    }

    public static void main(String[] args) {
        List<Long> data = new ArrayList<>();
        long seed = 12345;
        for (long i = 0; i < N; i++) {
            seed = (seed * 1103515245L + 12345L) % 2147483648L;
            data.add(seed % 100000L);
        }
        List<Long> sorted = sort(data);
        long checksum = 0;
        int at = 0;
        while (at < sorted.size()) {
            checksum = (checksum + sorted.get(at) * (at % 7 + 1)) % 1000000007L;
            at += 1;
        }
        System.out.println(sorted.size());
        System.out.println(checksum);
    }
}
