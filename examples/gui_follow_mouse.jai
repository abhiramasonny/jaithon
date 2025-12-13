
import lib/modules/gui/window

public static void main()
    var w = new Window()
    w.init(320, 240, "Follow Mouse")
    int frame = 0
    while frame > -1
        w.poll()
        var pos = w.mousePos()
        float x = pos[0]
        float y = pos[1]
        w.clear(rgb(0, 0, 0))
        w.fillRect(x - 5, y - 5, 10, 10, rgb(0, 255, 0))
        w.display()
        w.update()
        frame = frame + 1
    end
end

main()
