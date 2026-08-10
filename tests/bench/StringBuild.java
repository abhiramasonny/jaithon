// Same program as string_build.jai.
import java.util.ArrayList;
import java.util.List;

public class StringBuild {
    public static void main(String[] args) {
        List<String> parts = new ArrayList<>();
        for (int i = 0; i < 200000; i++) {
            parts.add("item-" + i);
        }
        String joined = String.join(",", parts);
        System.out.println(joined.length());
        System.out.println(joined.split(",", -1).length);
    }
}
