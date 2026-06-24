import ds/array
import physicsengine/vector
import physicsengine/body

class World
    var bodies
    var gravity
    func init(self, gravity)
        self.bodies = []
        self.gravity = gravity
    end
    
    func addBody(self, b)
        arrayPush(self.bodies, b)
    end
    
    func step(self, dt)
        var n = arrayLen(self.bodies)
        var i = 0
        while i < n
            var j = i + 1
            while j < n
                var bi = self.bodies[i]
                var bj = self.bodies[j]
                var delta = sub(bi.pos, bj.pos)
                var distSq = delta[0] * delta[0] + delta[1] * delta[1] + 0.0001
                var dist = sqrt(distSq)
                var force = self.gravity * bi.mass * bj.mass / distSq
                var dir = scale(delta, 1.0 / dist)
                var accI = scale(dir, -force / bi.mass)
                var accJ = scale(dir, force / bj.mass)
                bi.vel = add(bi.vel, scale(accI, dt))
                bj.vel = add(bj.vel, scale(accJ, dt))
                j = j + 1
            end
            i = i + 1
        end
        
        i = 0
        while i < n
            var b = self.bodies[i]
            b.pos = add(b.pos, scale(b.vel, dt))
            i = i + 1
        end
    end
end

func newWorld(gravity)
    return new World(gravity)
end
