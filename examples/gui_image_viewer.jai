import lib/modules/gui/window
import lib/modules/gui/image

public static void main()
    var win = new Window()
    win.init(400, 300, "PNG Viewer")

    var img = loadPNG("assets/images/logo.png")
    var w = img[0]
    var h = img[1]

    while true
        win.poll()
        win.clear(rgb(15, 15, 25))

        var ox = (win.width - w) / 2
        var oy = (win.height - h) / 2
        drawImage(win, img, ox, oy)

        win.display()
        if not win.update()
            break
        end
    end
end

main()
