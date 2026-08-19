#: Writing frames out as a video file.
#:
#: `VideoWriter` writes AVI, which is the one container simple enough to write
#: correctly without a media framework underneath: a RIFF file with a header,
#: a list of frames, and an index. Two codecs are offered — `MJPG`, which puts
#: a JPEG in each frame and is what a recording usually wants, and the
#: uncompressed forms, which are exact and enormous.
#:
#: The header carries the frame count and the total size, neither of which is
#: known until the last frame has been written, so the file is written straight
#: through and the four numbers that could not be known are seeked back to and
#: filled in by `release`. A writer that is never released leaves a file no
#: player will open, which is why `release` is not optional.
#:
#: ```
#: let writer = VideoWriter("out.avi", fourcc("M", "J", "P", "G"), 30.0, Size(640, 480))
#: for frame in frames { writer.write(frame) }
#: writer.release()
#: ```
import std.math
from std.io import open as open_file, File, SEEK_SET
from ..core.mat import Mat
from ..core.types import Size
from ..imgcodecs.jpeg import encode as encode_jpeg

#: Pack four characters into the code a codec is named by.
pub fn fourcc(a: str, b: str, c: str, d: str) -> int {
    return ord(a) | ord(b) << 8 | ord(c) << 16 | ord(d) << 24
}

fn le32(value: int) -> list[int] {
    return [value & 255, value >> 8 & 255, value >> 16 & 255, value >> 24 & 255]
}

fn le16(value: int) -> list[int] {
    return [value & 255, value >> 8 & 255]
}

fn tag(text: str) -> list[int] {
    var out: list[int] = []
    for character in text.chars() { out.push(ord(character)) }
    return out
}

#: Write frames to an AVI file.
pub class VideoWriter {
    #: Where the file is being written.
    pub let filename: str
    #: The codec the frames are stored with.
    pub let codec: int
    #: How many frames a second the file claims.
    pub let fps: float
    #: The size every frame has to be.
    pub let frame_size: Size
    #: Whether the frames are colour.
    pub let is_color: bool
    var handle: File?
    var frames: int
    var offsets: list[int]
    var lengths: list[int]
    var movi_start: int

    pub fn init(
        self,
        filename: str,
        codec: int,
        fps: float,
        frame_size: Size,
        is_color: bool = true
    ) {
        if frame_size.width <= 0 or frame_size.height <= 0 {
            throw ValueError("a video needs a positive frame size")
        }
        if fps <= 0.0 { throw ValueError("a video needs a positive frame rate") }
        self.filename = filename
        self.codec = codec
        self.fps = fps
        self.frame_size = frame_size
        self.is_color = is_color
        self.frames = 0
        self.offsets = []
        self.lengths = []
        self.movi_start = 0
        #: `w+` rather than `w`, because the header has to be seeked back to and
        #: rewritten once the frame count is known.
        self.handle = open_file(filename, "w+")
        self.write_header()
    }

    #: Whether the file was opened.
    pub fn is_opened(self) -> bool { return self.handle is not null }

    fn put(self, values: list[int]) -> void {
        let file = self.handle
        if file is null { throw RuntimeError("this writer has been released") }
        file.write_bytes(bytes(values))
    }

    fn write_header(self) -> void {
        let width = self.frame_size.width
        let height = self.frame_size.height
        let compressed = self.codec != 0
        let bit_count = if self.is_color { 24 } else { 8 }
        let image_size = width * height * (bit_count // 8)

        var header: list[int] = []
        for value in tag("RIFF") { header.push(value) }
        for value in le32(0) { header.push(value) }
        for value in tag("AVI ") { header.push(value) }

        #: A LIST's size counts its four-character type and everything after
        #: it. `hdrl` holds its type, the 56-byte `avih` chunk with its header,
        #: and the whole `strl` list: 4 + (8 + 56) + (8 + 4 + 8 + 56 + 8 + 40).
        #: Declaring it four bytes short -- which this did -- puts every reader
        #: that walks the tree four bytes into the middle of `movi`'s header,
        #: where it finds no chunk and stops, so the file opened, reported its
        #: size and rate correctly, and held no frames.
        var hdrl: list[int] = []
        for value in tag("LIST") { hdrl.push(value) }
        for value in le32(4 + 8 + 56 + 8 + 4 + 8 + 56 + 8 + 40) { hdrl.push(value) }
        for value in tag("hdrl") { hdrl.push(value) }

        for value in tag("avih") { hdrl.push(value) }
        for value in le32(56) { hdrl.push(value) }
        for value in le32(int(1000000.0 / self.fps + 0.5)) { hdrl.push(value) }
        for value in le32(image_size * int(self.fps + 0.5)) { hdrl.push(value) }
        for value in le32(0) { hdrl.push(value) }
        #: `HAS_INDEX`, which is what tells a player the index at the end is
        #: there to be used.
        for value in le32(0x10) { hdrl.push(value) }
        for value in le32(0) { hdrl.push(value) }
        for value in le32(0) { hdrl.push(value) }
        for value in le32(1) { hdrl.push(value) }
        for value in le32(image_size) { hdrl.push(value) }
        for value in le32(width) { hdrl.push(value) }
        for value in le32(height) { hdrl.push(value) }
        for _index in 0..4 {
            for value in le32(0) { hdrl.push(value) }
        }

        for value in tag("LIST") { hdrl.push(value) }
        for value in le32(4 + 8 + 56 + 8 + 40) { hdrl.push(value) }
        for value in tag("strl") { hdrl.push(value) }

        for value in tag("strh") { hdrl.push(value) }
        for value in le32(56) { hdrl.push(value) }
        for value in tag("vids") { hdrl.push(value) }
        for value in le32(self.codec) { hdrl.push(value) }
        for value in le32(0) { hdrl.push(value) }
        for value in le16(0) { hdrl.push(value) }
        for value in le16(0) { hdrl.push(value) }
        for value in le32(0) { hdrl.push(value) }
        #: Rate over scale is the frame rate; keeping scale at a thousand lets
        #: a fractional rate be written exactly.
        for value in le32(1000) { hdrl.push(value) }
        for value in le32(int(self.fps * 1000.0 + 0.5)) { hdrl.push(value) }
        for value in le32(0) { hdrl.push(value) }
        for value in le32(0) { hdrl.push(value) }
        for value in le32(image_size) { hdrl.push(value) }
        for value in le32(0xFFFFFFFF) { hdrl.push(value) }
        for value in le32(0) { hdrl.push(value) }
        for value in le16(0) { hdrl.push(value) }
        for value in le16(0) { hdrl.push(value) }
        for value in le16(width) { hdrl.push(value) }
        for value in le16(height) { hdrl.push(value) }

        for value in tag("strf") { hdrl.push(value) }
        for value in le32(40) { hdrl.push(value) }
        for value in le32(40) { hdrl.push(value) }
        for value in le32(width) { hdrl.push(value) }
        for value in le32(height) { hdrl.push(value) }
        for value in le16(1) { hdrl.push(value) }
        for value in le16(bit_count) { hdrl.push(value) }
        for value in le32(if compressed { self.codec } else { 0 }) { hdrl.push(value) }
        for value in le32(image_size) { hdrl.push(value) }
        for value in le32(0) { hdrl.push(value) }
        for value in le32(0) { hdrl.push(value) }
        for value in le32(0) { hdrl.push(value) }
        for value in le32(0) { hdrl.push(value) }

        for value in hdrl { header.push(value) }
        for value in tag("LIST") { header.push(value) }
        for value in le32(0) { header.push(value) }
        for value in tag("movi") { header.push(value) }
        self.put(header)
        self.movi_start = header.len() - 4
    }

    #: Add one frame.
    pub fn write(self, frame: Mat) -> void {
        let file = self.handle
        if file is null { throw RuntimeError("this writer has been released") }
        if frame.cols != self.frame_size.width or frame.rows != self.frame_size.height {
            throw ValueError(f"this writer takes {self.frame_size} frames, got {frame.size()}")
        }
        frame.flush()

        var payload: list[int] = []
        if self.codec == 0 {
            #: An uncompressed AVI frame is a bottom-up bitmap, the same way
            #: round a BMP is.
            let values = frame.to_bytes()
            let cn = frame.channels()
            let width = self.frame_size.width
            let height = self.frame_size.height
            let stride = (width * cn + 3) // 4 * 4
            for step in 0..height {
                let y = height - 1 - step
                for x in 0..width {
                    for c in 0..cn { payload.push(values[(y * width + x) * cn + c]) }
                }
                for _pad in 0..stride - width * cn { payload.push(0) }
            }
        } else {
            for value in encode_jpeg(frame, 90).to_list() { payload.push(value) }
        }

        var chunk = tag("00dc")
        for value in le32(payload.len()) { chunk.push(value) }
        for value in payload { chunk.push(value) }
        if payload.len() % 2 == 1 { chunk.push(0) }

        self.offsets.push(file.tell() - self.movi_start)
        self.lengths.push(payload.len())
        self.put(chunk)
        self.frames += 1
    }

    #: Finish the file.
    #:
    #: The index is written and the sizes that were left at zero are filled in.
    #: Until this runs the file is not a video.
    pub fn release(self) -> void {
        let file = self.handle
        if file is null { return }
        let movi_end = file.tell()

        var index = tag("idx1")
        for value in le32(self.frames * 16) { index.push(value) }
        for at in 0..self.frames {
            for value in tag("00dc") { index.push(value) }
            for value in le32(0x10) { index.push(value) }
            for value in le32(self.offsets[at]) { index.push(value) }
            for value in le32(self.lengths[at]) { index.push(value) }
        }
        file.write_bytes(bytes(index))
        let total = file.tell()

        #: `RIFF` counts everything after its own size field; the `movi` list
        #: counts everything after its; and the frame count sits in the main
        #: header. None of the three could be known while writing.
        file.seek(4, SEEK_SET)
        file.write_bytes(bytes(le32(total - 8)))
        file.seek(48, SEEK_SET)
        file.write_bytes(bytes(le32(self.frames)))
        file.seek(self.movi_start - 4, SEEK_SET)
        file.write_bytes(bytes(le32(movi_end - self.movi_start)))
        file.seek(140, SEEK_SET)
        file.write_bytes(bytes(le32(self.frames)))
        file.close()
        self.handle = null
    }
}
