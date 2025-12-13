import gui/window
import ds/array

func loadPNG(path)
    return gui_load_png(path)
end

func drawImage(win, img, ox, oy)
    if type(img) != "array" or arrayLen(img) < 3
        return
    end
    var w = toInt(img[0])
    var h = toInt(img[1])
    var pixels = img[2]
    gui_blit(ox, oy, w, h, pixels)
end
