import ds/array
import physicsengine/vector

class Body
    var pos
    var vel
    var mass
    var radius
    func init(self, x, y, vx, vy, mass, radius)
        self.pos = vec2(x, y)
        self.vel = vec2(vx, vy)
        self.mass = mass
        self.radius = radius
    end
end

func makeBody(x, y, vx, vy, mass, radius)
    return new Body(x, y, vx, vy, mass, radius)
end
