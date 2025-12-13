import physicsengine/index

func spawnDemoWorld()
    var w = newWorld(1.0)
    w.addBody(makeBody(-4, 0, 0, 0.6, 4.0, 0.4))
    w.addBody(makeBody(4, 0, 0, -0.6, 4.0, 0.4))
    w.addBody(makeBody(0, 5, -0.5, -0.2, 2.0, 0.3))
    w.addBody(makeBody(0, -5, 0.5, 0.2, 2.0, 0.3))
    return w
end

func dump(w, stepNum)
    var i = 0
    print "step " + str(stepNum)
    while i < arrayLen(w.bodies)
        var b = w.bodies[i]
        print "  body " + str(i) + " pos=(" + str(b.pos[0]) + "," + str(b.pos[1]) + ") vel=(" + str(b.vel[0]) + "," + str(b.vel[1]) + ")"
        i = i + 1
    end
end

func main()
    var w = spawnDemoWorld()
    var steps = 500
    var dt = 0.02
    var s = 0
    while s < steps
        w.step(dt)
        if s % 50 == 0
            dump(w, s)
        end
        s = s + 1
    end
    dump(w, steps)
end

main()
