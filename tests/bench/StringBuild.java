// Same program as string_build.jai.
import java.util.ArrayList;
import java.util.List;

public class StringBuild {
    // BENCH_LEVEL picks how much work this benchmark does: easy a sixteenth of it,
    // medium a quarter, hard (the default, and anything unrecognised) all of it.
    static long scale() {
        String l = System.getenv("BENCH_LEVEL");
        if ("easy".equals(l)) return 16;
        if ("medium".equals(l)) return 4;
        return 1;
    }

    static final long SCALE = scale();
    static final long N = 2000000L / SCALE;

    public static void main(String[] args) {
        List<String> parts = new ArrayList<>();
        for (long i = 0; i < N; i++) {
            parts.add("item-" + i);
        }
        String joined = String.join(",", parts);
        System.out.println(joined.length());
        System.out.println(joined.split(",", -1).length);
    }
}
