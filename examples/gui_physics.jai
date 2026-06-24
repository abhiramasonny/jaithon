import gui/window
import ds/array
import physicsengine/body

var WIN_WIDTH = 960
var WIN_HEIGHT = 640
var PANEL_W = 180
var TARGET_FPS = 1000
var SCALE = 60.0
var FLOOR_MARGIN = 30
var GRAVITY = 9.8
var BOUNCE = 0.82
var FRICTION = 0.995
var MAX_DT = 0.05
var MIN_RADIUS = 0.35
var RAND_RADIUS = 0.25
var BG_COLOR = rgb(13, 17, 23)
var FLOOR_COLOR = rgb(30, 34, 40)
var PANEL_BG = rgb(20, 24, 30)
var PANEL_ACCENT = rgb(50, 155, 255)
var PANEL_TEXT = rgb(220, 220, 220)
var BUTTON_BG = rgb(35, 40, 50)
var BUTTON_HI = rgb(60, 80, 110)
var COLOR0 = rgb(110, 197, 220)
var COLOR1 = rgb(244, 162, 97)
var COLOR2 = rgb(231, 111, 81)
var COLOR3 = rgb(42, 157, 143)
var COLOR4 = rgb(233, 196, 106)
var COLOR5 = rgb(38, 70, 83)
var MODE_BOUNCE = 0
var MODE_RAIN = 1
var MODE_ORBIT = 2
var RAIN_INTERVAL = 0.1
var MAX_BALLS = 300
var ORBIT_PULL = 22.0
var ORBIT_DAMP = 0.999

func colorFromIndex(i)
    var idx = i % 6
    if idx == 0
        return COLOR0
    end
    if idx == 1
        return COLOR1
    end
    if idx == 2
        return COLOR2
    end
    if idx == 3
        return COLOR3
    end
    if idx == 4
        return COLOR4
    end
    return COLOR5
end

func drawGlyph(win, x, y, ch, color)
    var w = 3
    var h = 5
    var p = [0,0,0,0,0]
    var ok = false
    if ch == "0"
        p = [7,5,5,5,7]
        ok = true
    end
    if not ok and ch == "1"
        p = [2,6,2,2,7]
        ok = true
    end
    if not ok and ch == "2"
        p = [7,1,7,4,7]
        ok = true
    end
    if not ok and ch == "3"
        p = [7,1,7,1,7]
        ok = true
    end
    if not ok and ch == "4"
        p = [5,5,7,1,1]
        ok = true
    end
    if not ok and ch == "5"
        p = [7,4,7,1,7]
        ok = true
    end
    if not ok and ch == "6"
        p = [7,4,7,5,7]
        ok = true
    end
    if not ok and ch == "7"
        p = [7,1,2,2,2]
        ok = true
    end
    if not ok and ch == "8"
        p = [7,5,7,5,7]
        ok = true
    end
    if not ok and ch == "9"
        p = [7,5,7,1,7]
        ok = true
    end
    if not ok and ch == "A"
        p = [2,5,7,5,5]
        ok = true
    end
    if not ok and ch == "B"
        p = [6,5,6,5,6]
        ok = true
    end
    if not ok and ch == "C"
        p = [7,4,4,4,7]
        ok = true
    end
    if not ok and ch == "D"
        p = [6,5,5,5,6]
        ok = true
    end
    if not ok and ch == "E"
        p = [7,4,6,4,7]
        ok = true
    end
    if not ok and ch == "I"
        p = [7,2,2,2,7]
        ok = true
    end
    if not ok and ch == "M"
        p = [5,7,7,5,5]
        ok = true
    end
    if not ok and ch == "N"
        p = [5,7,7,7,5]
        ok = true
    end
    if not ok and ch == "O"
        p = [7,5,5,5,7]
        ok = true
    end
    if not ok and ch == "R"
        p = [7,5,7,6,5]
        ok = true
    end
    if not ok and ch == "T"
        p = [7,2,2,2,2]
        ok = true
    end
    if not ok and ch == "U"
        p = [5,5,5,5,7]
        ok = true
    end
    if not ok and ch == " "
        p = [0,0,0,0,0]
        ok = true
    end
    if not ok
        return 4
    end
    var row = 0
    while row < h
        var col = 0
        while col < w
            var shift = w - col - 1
            var mask = 1
            var s = 0
            while s < shift
                mask = mask * 2
                s = s + 1
            end
            if ((p[row] / mask) % 2) == 1
                win.putPixel(x + col, y + row, color)
            end
            col = col + 1
        end
        row = row + 1
    end
    return w + 1
end

func drawText(win, x, y, text, color)
    var i = 0
    var cx = x
    while i < len(text)
        cx = cx + drawGlyph(win, cx, y, charAt(text, i), color)
        i = i + 1
    end
end

func drawWorld(win, balls, scale, floorScreenY, cx, cy)
    win.clear(BG_COLOR)
    win.fillRect(0, floorScreenY, win.width, win.height - floorScreenY, FLOOR_COLOR)
    var i = 0
    while i < arrayLen(balls)
        var ball = balls[i]
        var px = cx + ball.pos[0] * scale
        var py = cy - ball.pos[1] * scale
        var r = toInt(ball.radius * scale)
        if r < 2
            r = 2
        end
        win.fillCircle(toInt(px), toInt(py), r, colorFromIndex(i))
        i = i + 1
    end
end

func spawnBall(balls, mx, my, cx, cy, scale, floorY)
    var worldX = (mx - cx) / scale
    var worldY = (cy - my) / scale
    var radius = MIN_RADIUS + randFloat(0, 1) * RAND_RADIUS
    if worldY - radius < floorY
        worldY = floorY + radius + 0.01
    end
    var mass = radius * radius * 10
    var b = makeBody(worldX, worldY, 0, 0, mass, radius)
    arrayPush(balls, b)
end

func clearBalls(balls)
    while arrayLen(balls) > 0
        arrayPop(balls)
    end
end

func handleCollisions(balls, restitution)
    var n = arrayLen(balls)
    var i = 0
    while i < n
        var j = i + 1
        while j < n
            var a = balls[i]
            var b = balls[j]
            var dx = a.pos[0] - b.pos[0]
            var dy = a.pos[1] - b.pos[1]
            var distSq = dx * dx + dy * dy
            var minDist = a.radius + b.radius
            if distSq > 0 and distSq < minDist * minDist
                var dist = sqrt(distSq)
                var nx = dx / dist
                var ny = dy / dist
                var relvx = a.vel[0] - b.vel[0]
                var relvy = a.vel[1] - b.vel[1]
                var velAlongNormal = relvx * nx + relvy * ny
                if velAlongNormal < 0
                    var invMass = (1.0 / a.mass) + (1.0 / b.mass)
                    var jimp = -(1 + restitution) * velAlongNormal / invMass
                    var impX = nx * jimp
                    var impY = ny * jimp
                    a.vel[0] = a.vel[0] + impX / a.mass
                    a.vel[1] = a.vel[1] + impY / a.mass
                    b.vel[0] = b.vel[0] - impX / b.mass
                    b.vel[1] = b.vel[1] - impY / b.mass
                end
                var overlap = minDist - dist
                var correction = overlap * 0.5
                a.pos[0] = a.pos[0] + nx * correction
                a.pos[1] = a.pos[1] + ny * correction
                b.pos[0] = b.pos[0] - nx * correction
                b.pos[1] = b.pos[1] - ny * correction
            end
            j = j + 1
        end
        i = i + 1
    end
end

func drawPanel(win, mode)
    win.fillRect(0, 0, PANEL_W, WIN_HEIGHT, PANEL_BG)
    drawText(win, 12, 16, "MODE", PANEL_TEXT)
    var optionH = 50
    var y = 60
    var i = 0
    while i < 3
        var isSel = mode == i
        var bg = BUTTON_BG
        if isSel
            bg = BUTTON_HI
        end
        win.fillRect(10, y - 6, PANEL_W - 20, optionH - 8, bg)
        if i == 0
            drawText(win, 20, y, "BOUNCE", PANEL_TEXT)
        end
        if i == 1
            drawText(win, 20, y, "RAIN", PANEL_TEXT)
        end
        if i == 2
            drawText(win, 20, y, "ORBIT", PANEL_TEXT)
        end
        y = y + optionH
        i = i + 1
    end
end

func initMode(mode, balls, cx, cy, scale, floorY)
    clearBalls(balls)
    if mode == MODE_BOUNCE
        spawnBall(balls, cx, cy - 200, cx, cy, scale, floorY)
    end
    if mode == MODE_RAIN
        var i = 0
        while i < 5
            var xoff = -80 + i * 40
            spawnBall(balls, cx + xoff, 40, cx, cy, scale, floorY)
            i = i + 1
        end
    end
    if mode == MODE_ORBIT
        var i = 0
        while i < 6
            var angle = (i * 1.047)
            var rx = 2.5 * cos(angle)
            var ry = 2.5 * sin(angle)
            var vx = -ry * 0.6
            var vy = rx * 0.6
            var b = makeBody(rx, ry, vx, vy, 1.5, 0.35)
            arrayPush(balls, b)
            i = i + 1
        end
    end
end

func main()
    var win = new Window()
    var width = WIN_WIDTH
    var height = WIN_HEIGHT
    win.initWithFPS(width, height, "physics sim", TARGET_FPS)

    var scale = SCALE
    var floorY = -((height / 2) - FLOOR_MARGIN) / scale
    var worldW = width - PANEL_W
    var cx = PANEL_W + worldW / 2
    var cy = height / 2
    var floorScreenY = toInt(cy - floorY * scale)
    var balls = []
    var mode = MODE_BOUNCE
    var rainTimer = RAIN_INTERVAL
    initMode(mode, balls, cx, cy, scale, floorY)
    var gravity = GRAVITY
    var bounce = BOUNCE
    var friction = FRICTION
    var clickHeld = false
    var lastPrint = 0.0
    var halfWorld = worldW / (2 * scale)

    while true
        win.poll()
        var dt = win.getDeltaTime()
        if dt > MAX_DT
            dt = MAX_DT
        end

        var mousePressed = win.mouseDown(0)
        if mousePressed and not clickHeld
            var m = win.mousePos()
            if m[0] < PANEL_W
                var idx = (m[1] - 60) / 50
                if idx >= 0 and idx <= 2
                    mode = idx
                    rainTimer = RAIN_INTERVAL
                    initMode(mode, balls, cx, cy, scale, floorY)
                end
            else
                spawnBall(balls, m[0], m[1], cx, cy, scale, floorY)
            end
            clickHeld = true
        end
        if not mousePressed
            clickHeld = false
        end

        if mode == MODE_RAIN and arrayLen(balls) < MAX_BALLS
            rainTimer = rainTimer - dt
            if rainTimer <= 0
                var range = ((worldW / 2) - 10) / scale
                var wx = randFloat(-range, range)
                var sx = cx + wx * scale
                spawnBall(balls, sx, 10, cx, cy, scale, floorY)
                rainTimer = RAIN_INTERVAL
            end
        end

        var i = 0
        while i < arrayLen(balls)
            var b = balls[i]
            if mode == MODE_ORBIT
                var dx = -b.pos[0]
                var dy = -b.pos[1]
                var distSq = dx * dx + dy * dy + 0.01
                var invDist = 1.0 / sqrt(distSq)
                var acc = ORBIT_PULL / distSq
                b.vel[0] = b.vel[0] + dx * invDist * acc * dt
                b.vel[1] = b.vel[1] + dy * invDist * acc * dt
                b.vel[0] = b.vel[0] * ORBIT_DAMP
                b.vel[1] = b.vel[1] * ORBIT_DAMP
                b.pos[0] = b.pos[0] + b.vel[0] * dt
                b.pos[1] = b.pos[1] + b.vel[1] * dt
            else
                b.vel[1] = b.vel[1] - gravity * dt
                b.pos[1] = b.pos[1] + b.vel[1] * dt
                b.pos[0] = b.pos[0] + b.vel[0] * dt
                if b.pos[1] - b.radius < floorY
                    b.pos[1] = floorY + b.radius
                    b.vel[1] = -b.vel[1] * bounce
                    b.vel[0] = b.vel[0] * friction
                end
                if b.pos[0] - b.radius < -halfWorld
                    b.pos[0] = -halfWorld + b.radius
                    b.vel[0] = -b.vel[0] * bounce
                end
                if b.pos[0] + b.radius > halfWorld
                    b.pos[0] = halfWorld - b.radius
                    b.vel[0] = -b.vel[0] * bounce
                end
            end
            i = i + 1
        end

        if mode != MODE_ORBIT
            handleCollisions(balls, bounce)
        end

        drawWorld(win, balls, scale, floorScreenY, cx, cy)
        drawPanel(win, mode)
        win.present()

        var now = win.getTime()
        if now - lastPrint >= 1.0
            print "FPS " + str(toInt(win.getFPS())) + "  Balls " + str(arrayLen(balls))
            lastPrint = now
        end
    end
end

main()
