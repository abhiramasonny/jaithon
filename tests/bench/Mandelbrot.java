// Same program as mandelbrot.jai.
public class Mandelbrot {
    public static void main(String[] args) {
        final int width = 3200;
        final int height = 240;
        final int limit = 100;

        long inside = 0;
        for (int py = 0; py < height; py++) {
            double y0 = (double) py * 2.0 / (double) height - 1.0;
            for (int px = 0; px < width; px++) {
                double x0 = (double) px * 3.0 / (double) width - 2.0;
                double x = 0.0;
                double y = 0.0;
                int i = 0;
                while (i < limit) {
                    double x2 = x * x;
                    double y2 = y * y;
                    if (x2 + y2 > 4.0) break;
                    double xy = x * y;
                    y = 2.0 * xy + y0;
                    x = x2 - y2 + x0;
                    i++;
                }
                if (i == limit) inside++;
            }
        }
        System.out.println(limit);
        System.out.println(inside);
    }
}
