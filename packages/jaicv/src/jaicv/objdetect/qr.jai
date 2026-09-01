#: Reading QR codes, with OpenCV's `QRCodeDetector` interface.
#:
#: ```
#: let detector = QRCodeDetector()
#: let (text, corners) = detector.detect_and_decode(frame)
#: ```
#:
#: The reader is the standard's, in the standard's order: find the three finder
#: patterns, take the perspective they imply, sample the module grid through
#: it, read the format information, unmask, de-interleave the blocks,
#: Reed-Solomon them (`galois.jai`), and decode the bit stream.
#:
#: What it covers: every version, one to forty; every error-correction level;
#: numeric, alphanumeric, byte and ECI segments, and structured-append headers,
#: which it steps over rather than reassembling. Kanji segments are decoded to
#: their Shift-JIS bytes rather than to text, because that mapping is a table
#: this does not carry.
#:
#: What it does not cover: micro QR, which is a different symbol with a
#: different layout, and inverted (light-on-dark) codes.
#:
#: The reader, the writer and the detector live in `qr/`; this re-exports them.
from .qr.decode import QRCode, decode_grid
from .qr.detect import QRCodeDetector
from .qr.encode import encode_grid, qr_encode, smallest_version
from .qr.finder import Finder, find_finders
from .qr.symbol import alignment_centres, decode_format, decode_version, function_map
from .qr.symbol import is_function_module, mask_bit, version_size
from .qr.tables import LEVEL_NAMES

export { QRCode, decode_grid, QRCodeDetector }
export { encode_grid, qr_encode, smallest_version }
export { Finder, find_finders, LEVEL_NAMES }
export { alignment_centres, decode_format, decode_version, function_map }
export { is_function_module, mask_bit, version_size }
