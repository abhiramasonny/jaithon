#: Floating-point heavy simulation — the classic n-body benchmark.
const SOLAR_MASS = 39.47841760435743
const DAYS_PER_YEAR = 365.24

class Body {
    pub var x: float
    pub var y: float
    pub var z: float
    pub var vx: float
    pub var vy: float
    pub var vz: float
    pub var mass: float

    fn init(self, x: float, y: float, z: float, vx: float, vy: float, vz: float, mass: float) {
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.mass = mass
    }
}

fn advance(bodies: list[Body], dt: float) -> void {
    let n = bodies.len()
    for i in 0..n {
        let bi = bodies[i]
        for j in i + 1..n {
            let bj = bodies[j]
            let dx = bi.x - bj.x
            let dy = bi.y - bj.y
            let dz = bi.z - bj.z
            let d2 = dx * dx + dy * dy + dz * dz
            let mag = dt / (d2 * d2 ** 0.5)
            bi.vx -= dx * bj.mass * mag
            bi.vy -= dy * bj.mass * mag
            bi.vz -= dz * bj.mass * mag
            bj.vx += dx * bi.mass * mag
            bj.vy += dy * bi.mass * mag
            bj.vz += dz * bi.mass * mag
        }
    }
    for b in bodies {
        b.x += dt * b.vx
        b.y += dt * b.vy
        b.z += dt * b.vz
    }
}

fn energy(bodies: list[Body]) -> float {
    var e = 0.0
    let n = bodies.len()
    for i in 0..n {
        let bi = bodies[i]
        e += 0.5 * bi.mass * (bi.vx * bi.vx + bi.vy * bi.vy + bi.vz * bi.vz)
        for j in i + 1..n {
            let bj = bodies[j]
            let dx = bi.x - bj.x
            let dy = bi.y - bj.y
            let dz = bi.z - bj.z
            e -= bi.mass * bj.mass / (dx * dx + dy * dy + dz * dz) ** 0.5
        }
    }
    return e
}

fn main() -> int {
    let bodies = [
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
    ]
    print(f"{energy(bodies):.9f}")
    for _step in 0..100_000 { advance(bodies, 0.01) }
    print(f"{energy(bodies):.9f}")
    return 0
}
