// Same program as loop_sum.jai.
public class LoopSum {
    public static void main(String[] args) {
        long total = 0;
        for (long i = 0; i < 50000000L; i++) {
            total += i % 7;
        }
        System.out.println(total);
    }
}
