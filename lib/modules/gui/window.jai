# GUI Module
# Wraps the native GUI functions
# Uses optimized C implementations for drawing operations

class Window
    var width
    var height
    
    func init(self, w, h, title)
        self.width = w
        self.height = h
        gui_init(w, h, title)
    end
    
    func putPixel(self, x, y, color)
        gui_set_pixel(x, y, color)
    end
    
    func update(self)
        return gui_present()
    end
    
    func clear(self, color)
        gui_clear(color)
    end
    
    func fillRect(self, x, y, w, h, color)
        gui_fill_rect(x, y, w, h, color)
    end
    
    func drawRect(self, x, y, w, h, color)
        gui_fill_rect(x, y, w, 1, color)
        gui_fill_rect(x, y + h - 1, w, 1, color)
        gui_fill_rect(x, y, 1, h, color)
        gui_fill_rect(x + w - 1, y, 1, h, color)
    end
    
    func drawLine(self, x0, y0, x1, y1, color)
        gui_draw_line(x0, y0, x1, y1, color)
    end
    
    func drawCircle(self, xc, yc, r, color)
        gui_draw_circle(xc, yc, r, color)
    end
end

func rgb(r, g, b)
    return (r * 65536) + (g * 256) + b
end
