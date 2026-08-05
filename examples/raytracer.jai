#: A path-traced sphere scene rendered to a PPM file.
#:
#: Floating-point heavy, class heavy, and entirely standard library — a good
#: measure of how the VM handles real numeric work.
#:
#:     jaithon run examples/raytracer.jai out.ppm 400 300 32
import std.math
import std.io
from std.random import Random

class Vec3 {
    pub var x: float
    pub var y: float
    pub var z: float

    fn init(self, x: float, y: float, z: float) {
        self.x = x
        self.y = y
        self.z = z
    }

    fn __add__(self, o: Vec3) -> Vec3 { return Vec3(self.x + o.x, self.y + o.y, self.z + o.z) }

    fn __sub__(self, o: Vec3) -> Vec3 { return Vec3(self.x - o.x, self.y - o.y, self.z - o.z) }

    fn __mul__(self, o: Vec3) -> Vec3 { return Vec3(self.x * o.x, self.y * o.y, self.z * o.z) }

    fn __neg__(self) -> Vec3 { return Vec3(-self.x, -self.y, -self.z) }

    pub fn scale(self, k: float) -> Vec3 { return Vec3(self.x * k, self.y * k, self.z * k) }

    pub fn dot(self, o: Vec3) -> float { return self.x * o.x + self.y * o.y + self.z * o.z }

    pub fn length(self) -> float { return math.sqrt(self.dot(self)) }

    pub fn length_squared(self) -> float { return self.dot(self) }

    pub fn cross(self, o: Vec3) -> Vec3 {
        return Vec3(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x
        )
    }

    pub fn normalized(self) -> Vec3 {
        let n = self.length()
        if n == 0.0 { throw ValueError("cannot normalise a zero vector") }
        return self.scale(1.0 / n)
    }

    pub fn near_zero(self) -> bool {
        let e = 1e-8
        return math.fabs(self.x) < e and math.fabs(self.y) < e and math.fabs(self.z) < e
    }
}

fn reflect(v: Vec3, n: Vec3) -> Vec3 {
    return v - n.scale(2.0 * v.dot(n))
}

class Ray {
    pub var origin: Vec3
    pub var direction: Vec3

    fn init(self, origin: Vec3, direction: Vec3) {
        self.origin = origin
        self.direction = direction
    }

    pub fn at(self, t: float) -> Vec3 { return self.origin + self.direction.scale(t) }
}

enum Material { Lambertian(albedo: Vec3), Metal(albedo: Vec3, fuzz: float) }

class Hit {
    pub var point: Vec3
    pub var normal: Vec3
    pub var t: float
    pub var front_face: bool
    pub var material: Material

    fn init(self, point: Vec3, normal: Vec3, t: float, front_face: bool, material: Material) {
        self.point = point
        self.normal = normal
        self.t = t
        self.front_face = front_face
        self.material = material
    }
}

class Sphere {
    pub var center: Vec3
    pub var radius: float
    pub var material: Material

    fn init(self, center: Vec3, radius: float, material: Material) {
        self.center = center
        self.radius = radius
        self.material = material
    }

    pub fn hit(self, ray: Ray, t_min: float, t_max: float) -> Hit? {
        let oc = ray.origin - self.center
        let a = ray.direction.length_squared()
        let half_b = oc.dot(ray.direction)
        let c = oc.length_squared() - self.radius * self.radius
        let discriminant = half_b * half_b - a * c
        if discriminant < 0.0 { return null }

        let root = math.sqrt(discriminant)
        var t = (-half_b - root) / a
        if t < t_min or t > t_max {
            t = (-half_b + root) / a
            if t < t_min or t > t_max { return null }
        }

        let point = ray.at(t)
        let outward = (point - self.center).scale(1.0 / self.radius)
        let front = ray.direction.dot(outward) < 0.0
        let normal = if front { outward } else { -outward }
        return Hit(point, normal, t, front, self.material)
    }
}

fn random_unit_vector(rng: Random) -> Vec3 {
    # Rejection sampling inside the unit sphere, then normalise. Sampling the
    # cube and normalising directly would bias toward the corners.
    loop {
        let p = Vec3(
            rng.float_range(-1.0, 1.0),
            rng.float_range(-1.0, 1.0),
            rng.float_range(-1.0, 1.0)
        )
        if p.length_squared() < 1.0 and p.length_squared() > 1e-12 {
            return p.normalized()
        }
    }
}

fn scatter(material: Material, ray: Ray, hit: Hit, rng: Random) -> Ray? {
    return match material {
        Material.Lambertian(_albedo) => {
            var direction = hit.normal + random_unit_vector(rng)
            if direction.near_zero() {
                direction = hit.normal
            }
            Ray(hit.point, direction)
        }
        Material.Metal(_albedo, fuzz) => {
            let reflected = reflect(ray.direction.normalized(), hit.normal)
            let scattered = Ray(hit.point, reflected + random_unit_vector(rng).scale(fuzz))
            if scattered.direction.dot(hit.normal) > 0.0 {
                scattered
            } else {
                null
            }
        }
    }
}

fn albedo_of(material: Material) -> Vec3 {
    return match material {
        Material.Lambertian(a) => a,
        Material.Metal(a, _f) => a,
    }
}

fn closest_hit(world: list[Sphere], ray: Ray, t_min: float, t_max: float) -> Hit? {
    var best: Hit? = null
    var closest = t_max
    for sphere in world {
        let hit = sphere.hit(ray, t_min, closest)
        if hit != null {
            closest = hit.t
            best = hit
        }
    }
    return best
}

fn ray_color(ray: Ray, world: list[Sphere], depth: int, rng: Random) -> Vec3 {
    if depth <= 0 { return Vec3(0.0, 0.0, 0.0) }

    let hit = closest_hit(world, ray, 0.001, 1e30)
    if hit == null {
        let unit = ray.direction.normalized()
        let t = 0.5 * (unit.y + 1.0)
        return Vec3(1.0, 1.0, 1.0).scale(1.0 - t) + Vec3(0.5, 0.7, 1.0).scale(t)
    }

    let bounced = scatter(hit.material, ray, hit, rng)
    if bounced == null { return Vec3(0.0, 0.0, 0.0) }
    return albedo_of(hit.material) * ray_color(bounced, world, depth - 1, rng)
}

fn to_byte(component: float, samples: int) -> int {
    # Average, then gamma-correct with gamma 2.
    let scaled = math.sqrt(component / samples)
    return int(256.0 * math.clamp(scaled, 0.0, 0.999))
}

fn main(args: list[str]) -> int {
    let path = if args.len() > 1 { args[1] } else { "render.ppm" }
    let width = if args.len() > 2 { args[2].to_int() ?? 400 } else { 400 }
    let height = if args.len() > 3 { args[3].to_int() ?? 225 } else { 225 }
    let samples = if args.len() > 4 { args[4].to_int() ?? 16 } else { 16 }
    let max_depth = 12

    let world = [
        Sphere(Vec3(0.0, -100.5, -1.0), 100.0, Material.Lambertian(Vec3(0.8, 0.8, 0.0))),
        Sphere(Vec3(0.0, 0.0, -1.2), 0.5, Material.Lambertian(Vec3(0.1, 0.2, 0.5))),
        Sphere(Vec3(-1.0, 0.0, -1.0), 0.5, Material.Metal(Vec3(0.8, 0.8, 0.8), 0.05)),
        Sphere(Vec3(1.0, 0.0, -1.0), 0.5, Material.Metal(Vec3(0.8, 0.6, 0.2), 0.3)),
    ]

    let aspect = float(width) / float(height)
    let viewport_height = 2.0
    let viewport_width = aspect * viewport_height
    let origin = Vec3(0.0, 0.0, 0.0)
    let horizontal = Vec3(viewport_width, 0.0, 0.0)
    let vertical = Vec3(0.0, viewport_height, 0.0)
    let lower_left = origin - horizontal.scale(0.5) - vertical.scale(0.5) - Vec3(0.0, 0.0, 1.0)

    let rng = Random.with_seed(20260804)
    var out = [f"P3\n{width} {height}\n255"]

    for row in 0..height {
        if row % 25 == 0 { print(f"scanline {row}/{height}") }
        for col in 0..width {
            var color = Vec3(0.0, 0.0, 0.0)
            for _s in 0..samples {
                let u = (col + rng.next_float()) / (width - 1)
                let v = (height - 1 - row + rng.next_float()) / (height - 1)
                let direction = lower_left + horizontal.scale(u) + vertical.scale(v) - origin
                color = color + ray_color(Ray(origin, direction), world, max_depth, rng)
            }
            let r = to_byte(color.x, samples)
            let g = to_byte(color.y, samples)
            let b = to_byte(color.z, samples)
            out.push(f"{r} {g} {b}")
        }
    }

    io.write_file(path, "\n".join(out) + "\n")
    print(f"wrote {path} ({width}x{height}, {samples} samples/pixel)")
    return 0
}
