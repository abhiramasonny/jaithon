// Same program as mandelbrot.jai.
public class Mandelbrot {
    // BENCH_LEVEL picks how much work this benchmark does: easy a sixteenth of it,
    // medium a quarter, hard (the default, and anything unrecognised) all of it.
    static long scale() {
        String l = System.getenv("BENCH_LEVEL");
        if ("easy".equals(l)) return 16;
        if ("medium".equals(l)) return 4;
        return 1;
    }

    static final long SCALE = scale();
    // Work is the pixel area, so halve each side per level rather than divide.
    static final long SIDE_DIV = SCALE == 16 ? 4L : SCALE == 4 ? 2L : 1L;
    static final int WIDTH = (int) (3200L / SIDE_DIV);
    static final int HEIGHT = (int) (240L / SIDE_DIV);

    public static void main(String[] args) {
        final int width = WIDTH;
        final int height = HEIGHT;
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
