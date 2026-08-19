#: Reading a video file back.
#:
#: AVI, which is what `VideoWriter` writes and the one container that can be
#: read correctly without a media framework underneath: a RIFF tree whose
#: `movi` list holds one chunk per frame, and whose `hdrl` list says how large
#: the frames are, how fast they run and how they are compressed.
#:
#: The file is INDEXED, not loaded. Opening walks the chunk headers and records
#: where each frame's bytes begin and how many there are; a frame's payload is
#: read and decoded when it is asked for. So a long recording costs an offset
#: and a length per frame rather than its own size in memory, and `seek` is
#: free -- which is what makes `CAP_PROP_POS_FRAMES` writable rather than a
#: fiction.
#:
#: The index at the end of the file (`idx1`) is deliberately not used. It is
#: optional, it is wrong in plenty of files that carry it, and walking `movi`
#: answers the same question from the data itself.
#:
#: ```
#: let video = AviReader("clip.avi")
#: for index in 0..video.frame_count() { process(video.frame(index)) }
#: ```
from std.io import open as open_file, File, SEEK_SET
from ..core.mat import Mat, merge
from ..core.types import CV_8U, CV_MAKETYPE, Size
from ..imgcodecs.jpeg import decode as decode_jpeg, looks_like as looks_like_jpeg

fn le16_at(data: bytes, at: int) -> int { return data[at] | data[at + 1] << 8 }

fn le32_at(data: bytes, at: int) -> int {
    return data[at] | data[at + 1] << 8 | data[at + 2] << 16 | data[at + 3] << 24
}

fn tag_at(data: bytes, at: int) -> str {
    var out = ""
    for index in 0..4 { out += chr(data[at + index]) }
    return out
}

#: Whether a four-character code names motion JPEG.
fn is_mjpeg(code: str) -> bool {
    let upper = code.upper().strip()
    return upper == "MJPG" or upper == "MJPEG" or upper == "JPEG" or upper == "AVC1MJPG"
}

#: One AVI file, opened and indexed.
pub class AviReader {
    #: Where the file is.
    pub let filename: str
    #: Frame width, in pixels.
    pub var width: int
    #: Frame height, in pixels.
    pub var height: int
    #: Frames a second, as the stream header states it.
    pub var fps: float
    #: The four-character code naming the codec, as text.
    pub var codec: str
    #: Bits per pixel, from the bitmap header.
    pub var bit_count: int
    var handle: File?
    var offsets: list[int]
    var lengths: list[int]

    pub fn init(self, filename: str) {
        self.filename = filename
        self.width = 0
        self.height = 0
        self.fps = 0.0
        self.codec = ""
        self.bit_count = 24
        self.offsets = []
        self.lengths = []
        self.handle = null

        let file = open_file(filename, "r")
        self.handle = file
        let total = file.size()
        if total < 12 {
            self.release()
            throw ValueError(f"{filename} is too short to be an AVI file")
        }
        file.seek(0, SEEK_SET)
        let header = file.read_bytes(12)
        if tag_at(header, 0) != "RIFF" or tag_at(header, 8) != "AVI " {
            self.release()
            throw ValueError(f"{filename} is not a RIFF AVI file")
        }
        self.walk(file, 12, total, 0)
        if self.width <= 0 or self.height <= 0 {
            self.release()
            throw ValueError(f"{filename} carries no video stream header")
        }
    }

    #: Walk one run of chunks, recursing into the lists that hold others.
    #:
    #: `stream` counts `strl` lists as they are met, which is what numbers the
    #: streams: the frames of stream zero are the chunks named `00dc` or
    #: `00db`, and an AVI with a soundtrack has its samples under `01wb`, which
    #: this steps over rather than mistaking for pictures.
    fn walk(self, file: File, start: int, end: int, depth: int) -> void {
        if depth > 8 { return }
        var at = start
        var stream = 0
        var video_stream = -1
        while at + 8 <= end {
            file.seek(at, SEEK_SET)
            let head = file.read_bytes(8)
            if head.len() < 8 { return }
            let id = tag_at(head, 0)
            let size = le32_at(head, 4)
            if size < 0 or at + 8 + size > end + 8 { return }
            let body = at + 8

            if id == "LIST" or id == "RIFF" {
                let kind = tag_at(file.read_bytes(4), 0)
                if kind == "hdrl" or kind == "strl" or kind == "movi" or kind == "rec " {
                    self.walk(file, body + 4, body + size, depth + 1)
                }
                if kind == "strl" { stream += 1 }
            } elif id == "strh" {
                let head_body = file.read_bytes(if size < 40 { size } else { 40 })
                if head_body.len() >= 28 and tag_at(head_body, 0) == "vids" {
                    video_stream = stream
                    self.codec = tag_at(head_body, 4)
                    let scale = le32_at(head_body, 20)
                    let rate = le32_at(head_body, 24)
                    if scale > 0 { self.fps = float(rate) / float(scale) }
                }
            } elif id == "strf" and video_stream == stream {
                let format = file.read_bytes(if size < 20 { size } else { 20 })
                if format.len() >= 20 {
                    self.width = le32_at(format, 4)
                    #: A negative height means the rows are stored top-down;
                    #: the sign is the only thing that says so, and the
                    #: magnitude is still the height.
                    let signed = le32_at(format, 8)
                    self.height = if signed >= 0x80000000 { 0x100000000 - signed } else { signed }
                    self.bit_count = le16_at(format, 14)
                    let compression = le32_at(format, 16)
                    if compression != 0 and self.codec == "" {
                        var code = ""
                        for shift in 0..4 { code += chr(compression >> shift * 8 & 255) }
                        self.codec = code
                    }
                }
            } elif id.len() == 4 and (id[2:4] == "dc" or id[2:4] == "db") {
                let which = int(id[0:2]) ?? -1
                if which == 0 or which == video_stream {
                    if size > 0 {
                        self.offsets.push(body)
                        self.lengths.push(size)
                    }
                }
            }

            at = body + size + size % 2
        }
    }

    #: How many frames the file holds.
    pub fn frame_count(self) -> int { return self.offsets.len() }

    #: Whether the file is open.
    pub fn is_opened(self) -> bool { return self.handle is not null }

    #: The size every frame has.
    pub fn frame_size(self) -> Size { return Size(self.width, self.height) }

    #: One frame, decoded, as a `CV_8U` image in BGR order.
    pub fn frame(self, index: int) -> Mat {
        let file = self.handle
        if file is null { throw RuntimeError("this reader has been released") }
        if index < 0 or index >= self.offsets.len() {
            throw IndexError(f"frame {index} of {self.offsets.len()}")
        }
        file.seek(self.offsets[index], SEEK_SET)
        let payload = file.read_bytes(self.lengths[index])
        if looks_like_jpeg(payload) or is_mjpeg(self.codec) { return decode_jpeg(payload) }
        return self.unpack(payload)
    }

    #: An uncompressed frame: a bottom-up bitmap, the same way round a BMP is,
    #: with every row padded out to a multiple of four bytes.
    fn unpack(self, payload: bytes) -> Mat {
        let channels = if self.bit_count >= 24 { self.bit_count // 8 } else { 1 }
        let width = self.width
        let height = self.height
        let stride = (width * channels + 3) // 4 * 4
        if payload.len() < stride * height {
            throw ValueError(
                f"a {width}x{height} frame needs {stride * height} bytes, the file holds {payload.len()}"
            )
        }
        var out: list[int] = []
        for step in 0..height {
            let y = height - 1 - step
            let row = y * stride
            for x in 0..width {
                for c in 0..channels { out.push(payload[row + x * channels + c]) }
            }
        }
        let image = Mat.from_bytes(height, width, CV_MAKETYPE(CV_8U, channels), bytes(out))
        #: Four channels means an alpha the caller did not ask for: OpenCV hands
        #: back BGR, so the last plane is dropped rather than passed on.
        if channels != 4 { return image }
        let planes = image.split()
        var colour: list[Mat] = []
        for index in 0..3 { colour.push(planes[index]) }
        return merge(colour)
    }

    #: Close the file. Reading afterwards raises.
    pub fn release(self) -> void {
        let file = self.handle
        if file is null { return }
        file.close()
        self.handle = null
    }
}
