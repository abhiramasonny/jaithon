#: `imread`, `imwrite`, `imencode`, `imdecode` — the front door for image files.
#:
#: The format is chosen from the data when reading and from the extension when
#: writing, exactly as OpenCV does, and the read flags mean what they mean
#: there: `IMREAD_COLOR` always gives three channels, `IMREAD_GRAYSCALE`
#: always gives one, `IMREAD_UNCHANGED` gives whatever the file holds.
#:
#: ```
#: let frame = imread("photo.jpg")
#: imwrite("edges.png", edges)
#: ```
from std.io import Path, read_bytes, write_bytes
from ..core.mat import Mat, merge
from ..core.types import CV_8U, CV_16U, CV_MAKETYPE, CV_MAT_DEPTH
from ..imgproc.color import COLOR_BGR2GRAY, COLOR_BGRA2BGR, COLOR_GRAY2BGR, cvt_color
import .bmp as bmp
import .jpeg as jpeg
import .png as png
import .pnm as pnm

#: Return whatever the file holds, channels and depth included.
pub const IMREAD_UNCHANGED = -1
#: Return a single-channel image, converting if the file holds colour.
pub const IMREAD_GRAYSCALE = 0
#: Return a three-channel image, converting if the file holds grey.
pub const IMREAD_COLOR = 1
#: Keep sixteen-bit data at sixteen bits rather than scaling it down.
pub const IMREAD_ANYDEPTH = 2
#: Keep whatever channel count the file holds.
pub const IMREAD_ANYCOLOR = 4

#: JPEG quality, 0 to 100.
pub const IMWRITE_JPEG_QUALITY = 1
#: PNG compression level, 0 to 9.
pub const IMWRITE_PNG_COMPRESSION = 16

#: Which codec a blob belongs to, or the empty string when none recognises it.
pub fn sniff(data: bytes) -> str {
    if png_magic(data) { return "png" }
    if jpeg.looks_like(data) { return "jpeg" }
    if bmp.looks_like(data) { return "bmp" }
    if pnm.looks_like(data) { return "pnm" }
    return ""
}

fn png_magic(data: bytes) -> bool {
    if data.len() < 8 { return false }
    for index in 0..8 {
        if data[index] != png.SIGNATURE[index] { return false }
    }
    return true
}

fn apply_flags(image: Mat, flags: int) -> Mat {
    if flags == IMREAD_UNCHANGED { return image }
    var out = image
    let wants_deep = flags & IMREAD_ANYDEPTH != 0
    if not wants_deep and out.depth() == CV_16U {
        #: A sixteen-bit file read without `IMREAD_ANYDEPTH` comes back as
        #: eight-bit, scaled rather than truncated.
        out = out.convert_to(CV_MAKETYPE(CV_8U, out.channels()), 1.0 / 257.0, 0.0)
    }
    if flags == IMREAD_GRAYSCALE {
        if out.channels() == 1 { return out }
        if out.channels() == 4 { out = cvt_color(out, COLOR_BGRA2BGR) }
        return cvt_color(out, COLOR_BGR2GRAY)
    }
    if flags & IMREAD_COLOR != 0 and flags & IMREAD_ANYCOLOR == 0 {
        if out.channels() == 3 { return out }
        if out.channels() == 1 { return cvt_color(out, COLOR_GRAY2BGR) }
        return cvt_color(out, COLOR_BGRA2BGR)
    }
    return out
}

#: Decode an image already in memory.
pub fn imdecode(data: bytes, flags: int = IMREAD_COLOR) -> Mat {
    let kind = sniff(data)
    if kind == "" { throw ValueError("this data is not an image jaicv can read") }
    let image = if kind == "png" {
        png.decode(data)
    } elif kind == "jpeg" {
        jpeg.decode(data)
    } elif kind == "bmp" {
        bmp.decode(data)
    } else {
        pnm.decode(data)
    }
    return apply_flags(image, flags)
}

#: Read an image from disk. Throws when the file is missing or unreadable,
#: where OpenCV returns an empty `Mat` — a silent empty image costs more
#: debugging than it saves.
pub fn imread(path: str, flags: int = IMREAD_COLOR) -> Mat {
    let file = Path(path)
    if not file.exists() { throw IOError(f"no image at {path}") }
    return imdecode(read_bytes(path), flags)
}

#: Encode an image into memory. `extension` is the format's usual suffix,
#: with or without the dot.
pub fn imencode(extension: str, image: Mat, params: list[int] = []) -> bytes {
    var suffix = extension.lower()
    if suffix.starts_with(".") { suffix = suffix[1:suffix.len()] }
    var quality = 95
    var compression = 6
    var index = 0
    while index + 1 < params.len() {
        if params[index] == IMWRITE_JPEG_QUALITY { quality = params[index + 1] }
        if params[index] == IMWRITE_PNG_COMPRESSION { compression = params[index + 1] }
        index += 2
    }
    if suffix == "png" { return png.encode(image, level: compression) }
    if suffix == "jpg" or suffix == "jpeg" or suffix == "jpe" {
        return jpeg.encode(image, quality: quality)
    }
    if suffix == "bmp" or suffix == "dib" { return bmp.encode(image) }
    if suffix == "pgm" or suffix == "ppm" or suffix == "pnm" { return pnm.encode(image) }
    throw ValueError(f"jaicv cannot write .{suffix} images")
}

#: Write an image to disk, choosing the format from the extension.
pub fn imwrite(path: str, image: Mat, params: list[int] = []) -> bool {
    let file = Path(path)
    let extension = file.extension()
    if extension.len() == 0 { throw ValueError(f"{path} has no extension to choose a format by") }
    write_bytes(path, imencode(extension, image, params))
    return true
}

#: Whether jaicv can write a file with this extension.
pub fn have_image_writer(extension: str) -> bool {
    var suffix = extension.lower()
    if suffix.starts_with(".") { suffix = suffix[1:suffix.len()] }
    return suffix == "png" or
        suffix == "jpg" or
        suffix == "jpeg" or
        suffix == "jpe" or
        suffix == "bmp" or
        suffix == "dib" or
        suffix == "pgm" or
        suffix == "ppm" or
        suffix == "pnm"
}

#: Whether jaicv can read a file with this extension.
pub fn have_image_reader(extension: str) -> bool { return have_image_writer(extension) }
