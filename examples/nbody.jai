import ds/array

func makeBody(x, y, z, vx, vy, vz, mass)
    return [x, y, z, vx, vy, vz, mass]
end

func offsetMomentum(bodies)
    var px = 0.0
    var py = 0.0
    var pz = 0.0
    var i = 0
    while i < arrayLen(bodies)
        var b = bodies[i]
        var m = b[6]
        px = px + b[3] * m
        py = py + b[4] * m
        pz = pz + b[5] * m
        i = i + 1
    end
    var anchor = bodies[0]
    anchor[3] = 0 - px / anchor[6]
    anchor[4] = 0 - py / anchor[6]
    anchor[5] = 0 - pz / anchor[6]
end

func advance(bodies, dt)
    var i = 0
    while i < arrayLen(bodies)
        var j = i + 1
        while j < arrayLen(bodies)
            var bi = bodies[i]
            var bj = bodies[j]
            var dx = bi[0] - bj[0]
            var dy = bi[1] - bj[1]
            var dz = bi[2] - bj[2]
            var distSq = dx * dx + dy * dy + dz * dz
            var dist = sqrt(distSq)
            var mag = dt / (distSq * dist)
            var mi = bi[6]
            var mj = bj[6]
            bi[3] = bi[3] - dx * mj * mag
            bi[4] = bi[4] - dy * mj * mag
            bi[5] = bi[5] - dz * mj * mag
            bj[3] = bj[3] + dx * mi * mag
            bj[4] = bj[4] + dy * mi * mag
            bj[5] = bj[5] + dz * mi * mag
            j = j + 1
        end
        i = i + 1
    end
    
    i = 0
    while i < arrayLen(bodies)
        var b = bodies[i]
        b[0] = b[0] + dt * b[3]
        b[1] = b[1] + dt * b[4]
        b[2] = b[2] + dt * b[5]
        i = i + 1
    end
end

func energy(bodies)
    var e = 0.0
    var i = 0
    while i < arrayLen(bodies)
        var b = bodies[i]
        var v2 = b[3] * b[3] + b[4] * b[4] + b[5] * b[5]
        e = e + 0.5 * b[6] * v2
        var j = i + 1
        while j < arrayLen(bodies)
            var dx = b[0] - bodies[j][0]
            var dy = b[1] - bodies[j][1]
            var dz = b[2] - bodies[j][2]
            var dist = sqrt(dx * dx + dy * dy + dz * dz)
            e = e - (b[6] * bodies[j][6]) / dist
            j = j + 1
        end
        i = i + 1
    end
    return e
end

func loadSystem()
    var bodies = []
    var pi = 3.141592653589793
    var solarMass = 4.0 * pi * pi
    var daysPerYear = 365.24
    
    arrayPush(bodies, makeBody(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, solarMass))
    
    arrayPush(bodies, makeBody(
        4.84143144246472, -1.16032004402743, -0.103622044471123,
        1.66007664274404 * daysPerYear, 7.6990111841974 * daysPerYear, -0.0000690460016972063 * daysPerYear,
        0.000954791938424326 * solarMass))
    arrayPush(bodies, makeBody(
        8.34336671824458, 4.1247985641243, -0.403523417114321,
        -0.00276742510726862 * daysPerYear, 0.00499852801234917 * daysPerYear, 0.0000230417297573764 * daysPerYear,
        0.00028588598066613 * solarMass))
    arrayPush(bodies, makeBody(
        12.8943695621391, -15.1111514016986, -0.223307578892656,
        0.00296460137564762 * daysPerYear, 0.00237847173959481 * daysPerYear, -0.0000296589568540238 * daysPerYear,
        0.0000436624404335154 * solarMass))
    arrayPush(bodies, makeBody(
        15.3796971148509, -25.9193146099879, 0.179258772950371,
        0.00268067772490389 * daysPerYear, 0.00162824170038243 * daysPerYear, -0.0000951592254519714 * daysPerYear,
        0.0000515138902046611 * solarMass))
    
    offsetMomentum(bodies)
    return bodies
end

func main()
    var steps = 20000
    var bodies = loadSystem()
    
    print "Initial energy: " + str(energy(bodies))
    
    var i = 0
    while i < steps
        advance(bodies, 0.01)
        i = i + 1
    end
    
    print "Final energy: " + str(energy(bodies))
end

main()
