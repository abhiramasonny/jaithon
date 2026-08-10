// Same program as nbody.jai. See tests/bench/README.md for why the C++ and Java
// rows are here and what they are not. `pow(d2, 0.5)` rather than `sqrt(d2)`,
// and the same association of every product, so the bits agree to nine places.
#include <cmath>
#include <cstdio>
#include <vector>

static const double SOLAR_MASS = 39.47841760435743;
static const double DAYS_PER_YEAR = 365.24;

struct Body {
    double x, y, z, vx, vy, vz, mass;
    Body(double x, double y, double z, double vx, double vy, double vz, double mass)
        : x(x), y(y), z(z), vx(vx), vy(vy), vz(vz), mass(mass) {}
};

static void advance(std::vector<Body> &bodies, double dt) {
    size_t n = bodies.size();
    for (size_t i = 0; i < n; i++) {
        Body &bi = bodies[i];
        for (size_t j = i + 1; j < n; j++) {
            Body &bj = bodies[j];
            double dx = bi.x - bj.x;
            double dy = bi.y - bj.y;
            double dz = bi.z - bj.z;
            double d2 = dx * dx + dy * dy + dz * dz;
            double mag = dt / (d2 * std::pow(d2, 0.5));
            bi.vx -= dx * bj.mass * mag;
            bi.vy -= dy * bj.mass * mag;
            bi.vz -= dz * bj.mass * mag;
            bj.vx += dx * bi.mass * mag;
            bj.vy += dy * bi.mass * mag;
            bj.vz += dz * bi.mass * mag;
        }
    }
    for (Body &b : bodies) {
        b.x += dt * b.vx;
        b.y += dt * b.vy;
        b.z += dt * b.vz;
    }
}

static double energy(const std::vector<Body> &bodies) {
    double e = 0.0;
    size_t n = bodies.size();
    for (size_t i = 0; i < n; i++) {
        const Body &bi = bodies[i];
        e += 0.5 * bi.mass * (bi.vx * bi.vx + bi.vy * bi.vy + bi.vz * bi.vz);
        for (size_t j = i + 1; j < n; j++) {
            const Body &bj = bodies[j];
            double dx = bi.x - bj.x;
            double dy = bi.y - bj.y;
            double dz = bi.z - bj.z;
            e -= bi.mass * bj.mass / std::pow(dx * dx + dy * dy + dz * dz, 0.5);
        }
    }
    return e;
}

int main() {
    std::vector<Body> bodies = {
        Body(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, SOLAR_MASS),
        Body(
            4.84143144246472090,
            -1.16032004402742839,
            -0.103622044471123109,
            0.00166007664274403694 * DAYS_PER_YEAR,
            0.00769901118419740425 * DAYS_PER_YEAR,
            -0.0000690460016972063023 * DAYS_PER_YEAR,
            0.000954791938424326609 * SOLAR_MASS
        ),
        Body(
            8.34336671824457987,
            4.12479856412430479,
            -0.403523417114321381,
            -0.00276742510726862411 * DAYS_PER_YEAR,
            0.00499852801234917238 * DAYS_PER_YEAR,
            0.0000230417297573763929 * DAYS_PER_YEAR,
            0.000285885980666130812 * SOLAR_MASS
        ),
    };
    std::printf("%.9f\n", energy(bodies));
    for (int step = 0; step < 100000; step++) {
        advance(bodies, 0.01);
    }
    std::printf("%.9f\n", energy(bodies));
    return 0;
}
