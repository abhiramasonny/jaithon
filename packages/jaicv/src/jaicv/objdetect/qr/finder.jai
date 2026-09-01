#: Locating the three finder patterns by their dark-light run signature.
import std.math

#: --------------------------------------------------------------------------
#: Finding a symbol in a picture
#: --------------------------------------------------------------------------
pub class Finder {
    pub var y: float
    pub var x: float
    pub var width: float
    pub var count: int

    pub fn init(self, y: float, x: float, width: float) {
        self.y = y
        self.x = x
        self.width = width
        self.count = 1
    }
}

#: Whether five consecutive runs have the 1:1:3:1:1 shape a finder's centre
#: line always has, allowing each run half a module of slack.
fn finder_ratio(runs: list[int]) -> bool {
    var total = 0
    for value in runs { total += value }
    if total < 7 { return false }
    let unit = float(total) / 7.0
    let slack = unit / 2.0
    let wanted = [1.0, 1.0, 3.0, 1.0, 1.0]
    for index in 0..5 {
        if math.fabs(float(runs[index]) - wanted[index] * unit) >= wanted[index] * slack {
            return false
        }
    }
    return true
}

#: Walk one line of the image and report every 1:1:3:1:1 run of dark and light
#: it holds, as (centre along the line, module width).
fn scan_line(values: list[float], start: int, step: int, count: int) -> list[tuple[float, float]] {
    var out: list[tuple[float, float]] = []
    var runs = [0, 0, 0, 0, 0]
    var filled = 0
    var dark = values[start] < 128.0
    var length = 0

    for position in 0..count + 1 {
        let ended = position == count
        let value = if ended { not dark } else { values[start + position * step] < 128.0 }
        if not ended and value == dark {
            length += 1
            continue
        }
        #: A run just ended. It shifts into the window, and if the window now
        #: holds five runs ending on a dark one in the 1:1:3:1:1 proportion,
        #: the middle of the wide run is a finder's centre.
        for index in 0..4 { runs[index] = runs[index + 1] }
        runs[4] = length
        filled += 1
        if filled >= 5 and dark and finder_ratio(runs) {
            var total = 0
            for value_length in runs { total += value_length }
            let middle = float(position - runs[4] - runs[3]) - float(runs[2]) / 2.0
            out.push((middle, float(total) / 7.0))
        }
        dark = value
        length = 1
    }
    return out
}

#: Every finder pattern in a binarised image.
#:
#: The rows are scanned first, which finds each finder many times over; every
#: hit is then confirmed by scanning the column through it, and the confirmed
#: hits are merged. A finder seen on one row and no column is not a finder.
pub fn find_finders(values: list[float], rows: int, cols: int) -> list[Finder] {
    var found = scan_rows(values, rows, cols, 2)
    #: A finder is at least seven modules tall, so every other row still
    #: crosses one three times over and the pass costs half as much. Only a
    #: picture where that was not enough pays for the finer sweep.
    if found.len() < 3 { found = scan_rows(values, rows, cols, 1) }

    #: One row through a finder is a coincidence; several is a finder. A real
    #: finder is met on every row it spans, so it collects the most rows, and
    #: the dense middle of a large symbol throws up a scatter of shapes that
    #: pass the ratio test on two rows and no more.
    var best_count = 0
    for candidate in found {
        if candidate.count > best_count { best_count = candidate.count }
    }
    let floor = max(2, best_count // 2)

    var strong: list[Finder] = []
    #: Strongest first, and no more than sixteen. Every triple is considered,
    #: so the count matters cubically: thirty candidates is four thousand
    #: triples, and each one that passes the shape test is sampled over the
    #: whole symbol. Reading a version 40 symbol took twenty-seven seconds
    #: before this and under a second after.
    var taken: list[bool] = []
    for _index in 0..found.len() { taken.push(false) }
    while strong.len() < 16 {
        var pick = -1
        for index in 0..found.len() {
            if taken[index] or found[index].count < floor { continue }
            if pick < 0 or found[index].count > found[pick].count { pick = index }
        }
        if pick < 0 { break }
        taken[pick] = true
        strong.push(found[pick])
    }
    return strong
}

fn scan_rows(values: list[float], rows: int, cols: int, step: int) -> list[Finder] {
    var found: list[Finder] = []
    var y = 0
    while y < rows {
        for (centre, width) in scan_line(values, y * cols, 1, cols) {
            #: Merge into a finder already seen before confirming, not after.
            #: The confirmation walks a whole column, and a finder is met on
            #: every row it spans: confirming each time made reading a version
            #: 40 symbol thirty seconds' work, nearly all of it re-proving what
            #: the first row had already established.
            var merged = false
            for existing in found {
                if math.fabs(existing.y - float(y)) < existing.width * 4.0 and
                    math.fabs(existing.x - centre) < existing.width * 2.0 {
                    let weight = float(existing.count)
                    existing.x = (existing.x * weight + centre) / (weight + 1.0)
                    existing.width = (existing.width * weight + width) / (weight + 1.0)
                    existing.count += 1
                    merged = true
                    break
                }
            }
            if merged { continue }

            let x = int(centre)
            if x < 0 or x >= cols { continue }
            var confirmed = false
            var centre_y = 0.0
            var size = 0.0
            for hit in scan_line(values, x, cols, rows) {
                if math.fabs(hit[0] - float(y)) > hit[1] * 3.0 { continue }
                confirmed = true
                centre_y = hit[0]
                size = (width + hit[1]) / 2.0
                break
            }
            if not confirmed or size < 1.0 { continue }
            found.push(Finder(centre_y, centre, size))
        }
        y += step
    }
    return found
}
