# Demo: follow the mouse with a square (requires macOS/Metal)

import lib/modules/gui/window

func main()
    var w = new Window()
    w.init(320, 240, "Follow Mouse")
    var running = 0
    while running > -1 #prolly a cleaner way to do this but cant be bothered
        w.poll()
        var pos = w.mousePos()
        var x = pos[0]
        var y = pos[1]
        w.clear(rgb(0, 0, 0))
        w.fillRect(x - 5, y - 5, 10, 10, rgb(0, 255, 0))
        w.display()
        w.update()
        running = running + 1
    end
end

main()
