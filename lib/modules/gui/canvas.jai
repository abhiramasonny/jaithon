import lib/modules/ds/array
import lib/modules/gui/window

func newCanvas(w, h)
    var pixels = arrayFill(w * h, 0)
    var c = []
    arrayPush(c, w)
    arrayPush(c, h)
    arrayPush(c, pixels)
    return c
end

func canvasWidth(canvas)
    return canvas[0]
end

func canvasHeight(canvas)
    return canvas[1]
end

func canvasPixels(canvas)
    return canvas[2]
end

func canvasClear(canvas, color)
    var pixels = canvasPixels(canvas)
    var i = 0
    while i < arrayLen(pixels)
        pixels[i] = color
        i = i + 1
    end
end

func canvasPutPixel(canvas, x, y, color)
    var w = canvasWidth(canvas)
    var h = canvasHeight(canvas)
    if x < 0 or y < 0 or x >= w or y >= h
        return
    end
    canvasPixels(canvas)[y * w + x] = color
end

func canvasPresent(win, canvas)
    var _ = gui_upload(canvasPixels(canvas))
    return null
end
