// Same program as nbody.jai. Math.pow(d2, 0.5) rather than Math.sqrt(d2), and
// the same association of every product, so the bits agree to nine places.
import java.util.Locale;

public class Nbody {
    // BENCH_LEVEL picks how much work this benchmark does: easy a sixteenth of it,
    // medium a quarter, hard (the default, and anything unrecognised) all of it.
    static long scale() {
        String l = System.getenv("BENCH_LEVEL");
        if ("easy".equals(l)) return 16;
        if ("medium".equals(l)) return 4;
        return 1;
    }

    static final long SCALE = scale();
    static final long STEPS = 500000L / SCALE;

    static final double SOLAR_MASS = 39.47841760435743;
    static final double DAYS_PER_YEAR = 365.24;

    static final class Body {
        double x, y, z, vx, vy, vz, mass;

        Body(double x, double y, double z, double vx, double vy, double vz, double mass) {
            this.x = x;
            this.y = y;
            this.z = z;
            this.vx = vx;
            this.vy = vy;
            this.vz = vz;
            this.mass = mass;
        }
    }

    static void advance(Body[] bodies, double dt) {
        int n = bodies.length;
        for (int i = 0; i < n; i++) {
            Body bi = bodies[i];
            for (int j = i + 1; j < n; j++) {
                Body bj = bodies[j];
                double dx = bi.x - bj.x;
                double dy = bi.y - bj.y;
                double dz = bi.z - bj.z;
                double d2 = dx * dx + dy * dy + dz * dz;
                double mag = dt / (d2 * Math.pow(d2, 0.5));
                bi.vx -= dx * bj.mass * mag;
                bi.vy -= dy * bj.mass * mag;
                bi.vz -= dz * bj.mass * mag;
                bj.vx += dx * bi.mass * mag;
                bj.vy += dy * bi.mass * mag;
                bj.vz += dz * bi.mass * mag;
            }
        }
        for (Body b : bodies) {
            b.x += dt * b.vx;
            b.y += dt * b.vy;
            b.z += dt * b.vz;
        }
    }

    static double energy(Body[] bodies) {
        double e = 0.0;
        int n = bodies.length;
        for (int i = 0; i < n; i++) {
            Body bi = bodies[i];
            e += 0.5 * bi.mass * (bi.vx * bi.vx + bi.vy * bi.vy + bi.vz * bi.vz);
            for (int j = i + 1; j < n; j++) {
                Body bj = bodies[j];
                double dx = bi.x - bj.x;
                double dy = bi.y - bj.y;
                double dz = bi.z - bj.z;
                e -= bi.mass * bj.mass / Math.pow(dx * dx + dy * dy + dz * dz, 0.5);
            }
        }
        return e;
    }

    public static void main(String[] args) {
        Body[] bodies = {
            new Body(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, SOLAR_MASS),
            new Body(
                4.84143144246472090,
                -1.16032004402742839,
                -0.103622044471123109,
                0.00166007664274403694 * DAYS_PER_YEAR,
                0.00769901118419740425 * DAYS_PER_YEAR,
                -0.0000690460016972063023 * DAYS_PER_YEAR,
                0.000954791938424326609 * SOLAR_MASS
            ),
            new Body(
                8.34336671824457987,
                4.12479856412430479,
                -0.403523417114321381,
                -0.00276742510726862411 * DAYS_PER_YEAR,
                0.00499852801234917238 * DAYS_PER_YEAR,
                0.0000230417297573763929 * DAYS_PER_YEAR,
                0.000285885980666130812 * SOLAR_MASS
            ),
        };
        System.out.printf(Locale.ROOT, "%.9f\n", energy(bodies));
        for (long step = 0; step < STEPS; step++) {
            advance(bodies, 0.01);
        }
        System.out.printf(Locale.ROOT, "%.9f\n", energy(bodies));
    }
}
