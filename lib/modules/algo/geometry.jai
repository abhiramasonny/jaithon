class Point
    var x
    var y

    func init(self, px, py)
        self.x = px
        self.y = py
    end

    func dist(self, other)
        var dx = self.x - other.x
        var dy = self.y - other.y
        return sqrt(dx * dx + dy * dy)
    end

    func add(self, other)
        return new Point(self.x + other.x, self.y + other.y)
    end

    func sub(self, other)
        return new Point(self.x - other.x, self.y - other.y)
    end

    func scale(self, s)
        return new Point(self.x * s, self.y * s)
    end

    func dot(self, other)
        return self.x * other.x + self.y * other.y
    end

    func cross(self, other)
        return self.x * other.y - self.y * other.x
    end

    func norm(self)
        return sqrt(self.x * self.x + self.y * self.y)
    end

    func normalize(self)
        var n = self.norm()
        if n == 0
            return new Point(0, 0)
        end
        return new Point(self.x / n, self.y / n)
    end

    func rotate(self, angle)
        var c = _cos(angle)
        var s = _sin(angle)
        return new Point(self.x * c - self.y * s, self.x * s + self.y * c)
    end

    func toString(self)
        return concat(concat(concat("(", _str(self.x)), ","), concat(_str(self.y), ")"))
    end
end

func point(x, y)
    return new Point(x, y)
end

func ptDist(p1, p2)
    return p1.dist(p2)
end

func crossProd(o, a, b)
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
end

func triangleArea(p1, p2, p3)
    return abs(crossProd(p1, p2, p3)) / 2
end

func polygonArea(pts)
    var n = _alen(pts)
    if n < 3
        return 0
    end
    var area = 0
    var i = 0
    while i < n
        var j = (i + 1) % n
        area = area + pts[i].x * pts[j].y - pts[j].x * pts[i].y
        i = i + 1
    end
    return abs(area) / 2
end

func polygonPerimeter(pts)
    var n = _alen(pts)
    if n < 2
        return 0
    end
    var total = 0
    var i = 0
    while i < n
        var j = (i + 1) % n
        total = total + ptDist(pts[i], pts[j])
        i = i + 1
    end
    return total
end

func pointInPolygon(p, poly)
    var n = _alen(poly)
    var inside = false
    var j = n - 1
    var i = 0
    while i < n
        var xi = poly[i].x
        var yi = poly[i].y
        var xj = poly[j].x
        var yj = poly[j].y
        if (yi > p.y) != (yj > p.y)
            if p.x < (xj - xi) * (p.y - yi) / (yj - yi) + xi
                inside = not inside
            end
        end
        j = i
        i = i + 1
    end
    return inside
end

func convexHull(pts)
    var n = _alen(pts)
    if n < 3
        return arrayCopy(pts)
    end
    var sorted = arrayCopy(pts)
    var i = 1
    while i < n
        var key = sorted[i]
        var j = i - 1
        while j >= 0 and (sorted[j].x > key.x or (sorted[j].x == key.x and sorted[j].y > key.y))
            _set(sorted, j + 1, sorted[j])
            j = j - 1
        end
        _set(sorted, j + 1, key)
        i = i + 1
    end
    var lower = []
    i = 0
    while i < n
        while _alen(lower) >= 2 and crossProd(lower[_alen(lower) - 2], lower[_alen(lower) - 1], sorted[i]) <= 0
            _pop(lower)
        end
        _push(lower, sorted[i])
        i = i + 1
    end
    var upper = []
    i = n - 1
    while i >= 0
        while _alen(upper) >= 2 and crossProd(upper[_alen(upper) - 2], upper[_alen(upper) - 1], sorted[i]) <= 0
            _pop(upper)
        end
        _push(upper, sorted[i])
        i = i - 1
    end
    _pop(lower)
    _pop(upper)
    var hull = []
    i = 0
    while i < _alen(lower)
        _push(hull, lower[i])
        i = i + 1
    end
    i = 0
    while i < _alen(upper)
        _push(hull, upper[i])
        i = i + 1
    end
    return hull
end

func lineIntersect(p1, p2, p3, p4)
    var d1x = p2.x - p1.x
    var d1y = p2.y - p1.y
    var d2x = p4.x - p3.x
    var d2y = p4.y - p3.y
    var denom = d1x * d2y - d1y * d2x
    if denom == 0
        return null
    end
    var t = ((p3.x - p1.x) * d2y - (p3.y - p1.y) * d2x) / denom
    return new Point(p1.x + t * d1x, p1.y + t * d1y)
end

func segmentsIntersect(p1, p2, p3, p4)
    var d1 = crossProd(p3, p4, p1)
    var d2 = crossProd(p3, p4, p2)
    var d3 = crossProd(p1, p2, p3)
    var d4 = crossProd(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))
        return true
    end
    return false
end

func closestPair(pts)
    var n = _alen(pts)
    if n < 2
        return null
    end
    var best = pts[0].dist(pts[1])
    var bestI = 0
    var bestJ = 1
    var i = 0
    while i < n
        var j = i + 1
        while j < n
            var d = pts[i].dist(pts[j])
            if d < best
                best = d
                bestI = i
                bestJ = j
            end
            j = j + 1
        end
        i = i + 1
    end
    return [pts[bestI], pts[bestJ], best]
end

func pointOnSegment(p, a, b)
    if abs(crossProd(a, b, p)) > 0.000001
        return false
    end
    return min(a.x, b.x) <= p.x and p.x <= max(a.x, b.x) and min(a.y, b.y) <= p.y and p.y <= max(a.y, b.y)
end

func midpoint(p1, p2)
    return new Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)
end

func reflect(p, axis_p1, axis_p2)
    var dx = axis_p2.x - axis_p1.x
    var dy = axis_p2.y - axis_p1.y
    var t = ((p.x - axis_p1.x) * dx + (p.y - axis_p1.y) * dy) / (dx * dx + dy * dy)
    var foot = new Point(axis_p1.x + t * dx, axis_p1.y + t * dy)
    return new Point(2 * foot.x - p.x, 2 * foot.y - p.y)
end

func pointDistToLine(p, a, b)
    var area2 = abs(crossProd(a, b, p))
    var base = ptDist(a, b)
    if base == 0
        return ptDist(p, a)
    end
    return area2 / base
end

func isConvex(pts)
    var n = _alen(pts)
    if n < 3
        return true
    end
    var sign = 0
    var i = 0
    while i < n
        var cp = crossProd(pts[i], pts[(i + 1) % n], pts[(i + 2) % n])
        if cp != 0
            if sign == 0
                if cp > 0
                    sign = 1
                else
                    sign = 0 - 1
                end
            else
                if (cp > 0 and sign < 0) or (cp < 0 and sign > 0)
                    return false
                end
            end
        end
        i = i + 1
    end
    return true
end
