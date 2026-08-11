// Same program as list_ops.jai.
import java.util.ArrayList;
import java.util.List;

public class ListOps {
    public static void main(String[] args) {
        List<Long> xs = new ArrayList<>();
        for (long i = 0; i < 10000000L; i++) {
            xs.add(i * 3);
        }

        long total = 0;
        for (long x : xs) {
            total += x % 11;
        }

        List<Long> doubled = new ArrayList<>(xs.size());
        for (long x : xs) {
            doubled.add(x * 2);
        }

        System.out.println(total);
        System.out.println(doubled.get(999999));
    }
}
