width = 320
height = 240
limit = 100

inside = 0
for py in range(height):
    y0 = float(py) * 2.0 / float(height) - 1.0
    for px in range(width):
        x0 = float(px) * 3.0 / float(width) - 2.0
        x = 0.0
        y = 0.0
        i = 0
        while i < limit:
            x2 = x * x
            y2 = y * y
            if x2 + y2 > 4.0:
                break
            xy = x * y
            y = 2.0 * xy + y0
            x = x2 - y2 + x0
            i += 1
        if i == limit:
            inside += 1
print(limit)
print(inside)
