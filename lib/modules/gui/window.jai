# =============================================================================
# GUI Module (built for meta on Apple Silicon)
# =============================================================================
# Various functions imlemented from c
#   - gui_init(w, h, title [, fps]) - create window
#   - gui_set_pixel(x, y, color) - set single pixel in buffer
#   - gui_clear(color) - fast clear
#   - gui_fill_rect(x, y, w, h, color) - fast fill
#   - gui_draw_line(x0, y0, x1, y1, color) - fast Bresenham line
#   - gui_draw_circle(xc, yc, r, color) - fast midpoint circle
#   - gui_fill_circle(xc, yc, r, color) - fast filled circle
#   - gui_present() - upload to GPU, render with vsync
#   - gui_get_dt() - delta time (seconds since last frame)
#   - gui_get_fps() - current FPS
#   - gui_get_time() - time since program start
# =============================================================================

#create RGB color
func rgb(r, g, b)
    return (r * 65536) + (g * 256) + b
end

class Window
    var width
    var height
    var targetFPS
    var needsPresent

    func init(self, w, h, title)
        self.width = w
        self.height = h
        self.targetFPS = 60
        self.needsPresent = false
        gui_init(w, h, title, 60)
    end

    func initWithFPS(self, w, h, title, fps)
        self.width = w
        self.height = h
        self.targetFPS = fps
        self.needsPresent = false
        gui_init(w, h, title, fps)
    end
    func putPixel(self, x, y, color)
        gui_set_pixel(x, y, color)
    end

    func display(self)
        self.needsPresent = true
    end

    func update(self)
        if self.needsPresent
            self.needsPresent = false
            return gui_present()
        end
        return false
    end

    func present(self)
        self.display()
        return self.update()
    end

    func getDeltaTime(self)
        return gui_get_dt()
    end

    func getFPS(self)
        return gui_get_fps()
    end

    func getTime(self)
        return gui_get_time()
    end

    func poll(self)
        gui_poll()
    end

    func mousePos(self)
        return gui_mouse_pos()
    end

    func mouseDown(self, button)
        return gui_mouse_down(button)
    end

    func keyDown(self, keycode)
        return gui_key_down(keycode)
    end

    func getKeys(self)
        return gui_get_keys()
    end

    func clear(self, color)
        gui_clear(color)
    end

    func fillRect(self, rx, ry, rw, rh, color)
        gui_fill_rect(rx, ry, rw, rh, color)
    end

    func drawRect(self, rx, ry, rw, rh, color)
        var x = rx
        while x < rx + rw
            gui_set_pixel(x, ry, color)
            gui_set_pixel(x, ry + rh - 1, color)
            x = x + 1
        end
        var y = ry
        while y < ry + rh
            gui_set_pixel(rx, y, color)
            gui_set_pixel(rx + rw - 1, y, color)
            y = y + 1
        end
    end

    func drawLine(self, x0, y0, x1, y1, color)
        var dx = abs(x1 - x0)
        var sx = -1
        if x0 < x1
            sx = 1
        end
        var dy = -abs(y1 - y0)
        var sy = -1
        if y0 < y1
            sy = 1
        end
        var err = dx + dy
        while true
            gui_set_pixel(x0, y0, color)
            if x0 == x1 and y0 == y1
                break
            end
            var e2 = 2 * err
            if e2 >= dy
                err = err + dy
                x0 = x0 + sx
            end
            if e2 <= dx
                err = err + dx
                y0 = y0 + sy
            end
        end
    end

    func drawCircle(self, xc, yc, radius, color)
        gui_draw_circle(xc, yc, radius, color)
    end

    func fillCircle(self, xc, yc, radius, color)
        gui_fill_circle(xc, yc, radius, color)
    end
end

func mousePos()
    return gui_mouse_pos()
end

func mouseDown(button)
    return gui_mouse_down(button)
end

func keyDown(keycode)
    return gui_key_down(keycode)
end
