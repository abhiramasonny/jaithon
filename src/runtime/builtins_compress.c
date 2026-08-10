/* builtins_compress.c — __prim__.deflate and __prim__.adler32.
 *
 * A PNG is a container around one zlib stream, so writing PNGs from Jaithon
 * needs DEFLATE and nothing else needs it at all. That is the whole reason
 * this file exists: everything above it — the chunk layout, the CRC, the
 * filters — is ordinary byte shuffling that lib/std writes in Jaithon, while
 * the compressor is a hash-chain match search over a 32 KB window, which is
 * millions of comparisons per image and cannot be interpreted.
 *
 * Nothing is linked in to do it. zlib is on most machines but not all of them,
 * and a build that silently produces uncompressed PNGs where the library is
 * missing is worse than one that carries its own compressor. So this is a real
 * DEFLATE: LZ77 with lazy matching against a 32 KB window, emitted with the
 * fixed Huffman code of RFC 1951 §3.2.6. A dynamic code would buy a few more
 * percent on large images at the cost of the code-length code, the
 * canonical-code builder and the block-splitting heuristic that decides when
 * it pays — a lot of machinery for a small margin. Level 0 emits stored blocks
 * and is what an encoder asks for when it wants the bytes back untouched.
 *
 * The output carries the zlib wrapper of RFC 1950 — two header bytes and a
 * trailing Adler-32 — because that is what a PNG IDAT chunk holds and what
 * every decompressor on the other side expects.
 */

#include "builtins.h"
#include "runtime.h"

/* ------------------------------------------------------------------ */
/* Bit output                                                           */
/* ------------------------------------------------------------------ */

/* DEFLATE fills a byte from its least significant bit upwards, but a Huffman
 * code is written most significant bit first, which is why writerCode reverses
 * one before handing it over. Getting this backwards produces a stream that
 * looks plausible and decodes to nothing. */
typedef struct {
    uint8_t *bytes;
    size_t   count;
    size_t   capacity;
    uint32_t bits;        /* the partial byte, low bits first */
    int      bitCount;    /* how many of them are live, always 0-7 between calls */
} BitWriter;

static void writerReserve(BitWriter *writer, size_t extra) {
    if (writer->count + extra <= writer->capacity) return;
    size_t capacity = writer->capacity < 512 ? 512 : writer->capacity;
    while (capacity < writer->count + extra) capacity *= 2;
    writer->bytes = JAI_GROW_ARRAY(uint8_t, writer->bytes, writer->capacity, capacity);
    writer->capacity = capacity;
}

/* Appends a whole byte, bypassing the bit buffer. Every caller other than
 * writerBits is placing a byte-aligned field — a stored block's header, the
 * zlib wrapper — and has flushed the partial byte first. */
static void writerPush(BitWriter *writer, uint8_t byte) {
    writerReserve(writer, 1);
    writer->bytes[writer->count++] = byte;
}

/* Up to 16 bits at a time, which covers the longest code (9) and the longest
 * extra-bit field (13) with room to spare. */
static void writerBits(BitWriter *writer, uint32_t value, int count) {
    writer->bits |= (value & ((1u << count) - 1u)) << writer->bitCount;
    writer->bitCount += count;
    while (writer->bitCount >= 8) {
        writerPush(writer, (uint8_t)(writer->bits & 0xFFu));
        writer->bits >>= 8;
        writer->bitCount -= 8;
    }
}

/* Pad the partial byte with zeroes. A stored block and the zlib trailer both
 * start on a byte boundary. */
static void writerAlign(BitWriter *writer) {
    if (writer->bitCount == 0) return;
    writerPush(writer, (uint8_t)(writer->bits & 0xFFu));
    writer->bits = 0;
    writer->bitCount = 0;
}

static void writerCode(BitWriter *writer, uint32_t code, int bits) {
    uint32_t reversed = 0;
    for (int i = 0; i < bits; i++) reversed = (reversed << 1) | ((code >> i) & 1u);
    writerBits(writer, reversed, bits);
}

/* ------------------------------------------------------------------ */
/* The fixed Huffman code (RFC 1951 §3.2.6)                             */
/* ------------------------------------------------------------------ */

static void writeSymbol(BitWriter *writer, int symbol) {
    if (symbol <= 143)      writerCode(writer, 0x30u + (uint32_t)symbol, 8);
    else if (symbol <= 255) writerCode(writer, 0x190u + (uint32_t)(symbol - 144), 9);
    else if (symbol <= 279) writerCode(writer, (uint32_t)(symbol - 256), 7);
    else                    writerCode(writer, 0xC0u + (uint32_t)(symbol - 280), 8);
}

/* Length codes 257-285 and distance codes 0-29, as the RFC tabulates them. */
static const int kLengthBase[29] = {
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51,
    59, 67, 83, 99, 115, 131, 163, 195, 227, 258,
};
static const int kLengthExtra[29] = {
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3,
    3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
};
static const int kDistanceBase[30] = {
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513,
    769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
};
static const int kDistanceExtra[30] = {
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8,
    8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13,
};

static void writeMatch(BitWriter *writer, int length, int64_t distance) {
    int code = 0;
    while (code < 28 && length >= kLengthBase[code + 1]) code++;
    writeSymbol(writer, 257 + code);
    if (kLengthExtra[code] > 0)
        writerBits(writer, (uint32_t)(length - kLengthBase[code]), kLengthExtra[code]);

    int distanceCode = 0;
    while (distanceCode < 29 && distance >= kDistanceBase[distanceCode + 1])
        distanceCode++;
    writerCode(writer, (uint32_t)distanceCode, 5);
    if (kDistanceExtra[distanceCode] > 0) {
        writerBits(writer, (uint32_t)(distance - kDistanceBase[distanceCode]),
                   kDistanceExtra[distanceCode]);
    }
}

/* ------------------------------------------------------------------ */
/* Match finding                                                        */
/* ------------------------------------------------------------------ */

#define WINDOW_SIZE 32768
#define WINDOW_MASK (WINDOW_SIZE - 1)
#define HASH_BITS   15
#define HASH_SIZE   (1 << HASH_BITS)
#define MIN_MATCH   3
#define MAX_MATCH   258

/* head: the most recent position whose next three bytes hash here, or -1.
 * prev: for the position in each window slot, the previous position with the
 * same hash. A slot is only overwritten by a position a full window later, so
 * every entry still inside the window is current — which is what makes the
 * distance test below enough to reject a stale one. */
typedef struct {
    int32_t head[HASH_SIZE];
    int32_t prev[WINDOW_SIZE];
} MatchTable;

static uint32_t hashAt(const uint8_t *data) {
    return (uint32_t)(((uint32_t)data[0] << 10) ^ ((uint32_t)data[1] << 5) ^
                      (uint32_t)data[2]) & (HASH_SIZE - 1u);
}

/* Returns the position that held this hash before, which is where a search
 * from `position` starts: the chain must not begin with `position` itself, or
 * the first match found is the string against itself at distance zero — a
 * distance DEFLATE does not have and every decompressor rejects. */
static int32_t tableInsert(MatchTable *table, const uint8_t *data, int64_t length,
                           int64_t position) {
    if (position + MIN_MATCH > length) return -1;
    uint32_t slot = hashAt(data + position);
    int32_t previous = table->head[slot];
    table->prev[position & WINDOW_MASK] = previous;
    table->head[slot] = (int32_t)position;
    return previous;
}

/* The longest match for the bytes at `position`, or 0 when there is none worth
 * coding. `candidate` is where the hash chain starts, `chainLimit` bounds the
 * search, and `niceLength` stops it early once a match is long enough to not be
 * worth improving — the two knobs the level turns. */
static int findMatch(const MatchTable *table, const uint8_t *data, int64_t length,
                     int64_t position, int64_t candidate, int chainLimit,
                     int niceLength, int64_t *outDistance) {
    int64_t available = length - position;
    if (available < MIN_MATCH) return 0;
    int64_t maxLength = available < MAX_MATCH ? available : MAX_MATCH;
    int64_t oldest = position > WINDOW_SIZE ? position - WINDOW_SIZE : 0;

    const uint8_t *here = data + position;
    int best = 0;
    int64_t bestDistance = 0;

    for (int chain = 0; chain < chainLimit && candidate >= oldest; chain++) {
        const uint8_t *there = data + candidate;
        /* Reject on the byte that would have to improve on the best match so
         * far before comparing anything else; it is the cheapest test that can
         * rule a candidate out. */
        if (there[best] == here[best] && there[0] == here[0] && there[1] == here[1]) {
            int64_t matched = 0;
            while (matched < maxLength && there[matched] == here[matched]) matched++;
            if (matched > best) {
                best = (int)matched;
                bestDistance = position - candidate;
                if (best >= niceLength || best >= maxLength) break;
            }
        }
        candidate = table->prev[candidate & WINDOW_MASK];
    }

    if (best < MIN_MATCH) return 0;
    *outDistance = bestDistance;
    return best;
}

/* ------------------------------------------------------------------ */
/* The two block encodings                                              */
/* ------------------------------------------------------------------ */

/* What writeStoredBlocks will cost: five bytes of header per block, and the
 * bytes themselves. */
static size_t storedSize(size_t length) {
    size_t blocks = length == 0 ? 1 : (length + 65534) / 65535;
    return blocks * 5 + length;
}

/* Level 0: the bytes verbatim, in blocks of at most 65535 as the length field
 * allows. An empty input still emits one final empty block, because a zlib
 * stream with no block in it is not a stream. */
static void writeStoredBlocks(BitWriter *writer, const uint8_t *data, size_t length) {
    size_t offset = 0;
    do {
        size_t chunk = length - offset;
        if (chunk > 65535) chunk = 65535;
        writerBits(writer, (offset + chunk == length) ? 1u : 0u, 1);
        writerBits(writer, 0u, 2);                     /* BTYPE 00, stored */
        writerAlign(writer);
        writerPush(writer, (uint8_t)(chunk & 0xFFu));
        writerPush(writer, (uint8_t)((chunk >> 8) & 0xFFu));
        writerPush(writer, (uint8_t)(~chunk & 0xFFu));
        writerPush(writer, (uint8_t)((~chunk >> 8) & 0xFFu));
        writerReserve(writer, chunk);
        if (chunk > 0) {
            memcpy(writer->bytes + writer->count, data + offset, chunk);
            writer->count += chunk;
        }
        offset += chunk;
    } while (offset < length);
}

/* How hard each level looks. Level 1 takes the first match it can see; level 9
 * walks 4096 candidates and defers every one of them to check whether the next
 * position does better. */
static const int  kChainLimit[10] = {0, 4, 8, 16, 32, 64, 128, 256, 1024, 4096};
static const int  kNiceLength[10] = {0, 16, 24, 32, 48, 64, 128, 192, 258, 258};
static const bool kUseLazy[10] = {
    false, false, false, false, true, true, true, true, true, true,
};

/* Levels 1-9: one fixed-Huffman block over the whole input. Splitting into
 * several blocks only pays for a dynamic code, which this does not emit. */
static void writeFixedBlock(BitWriter *writer, const uint8_t *data, int64_t length,
                            int level) {
    writerBits(writer, 1u, 1);                         /* BFINAL */
    writerBits(writer, 1u, 2);                         /* BTYPE 01, fixed */

    MatchTable *table = JAI_ALLOC(MatchTable, 1);
    memset(table->head, 0xFF, sizeof table->head);     /* -1 in every slot */
    memset(table->prev, 0xFF, sizeof table->prev);

    int chainLimit = kChainLimit[level];
    int niceLength = kNiceLength[level];
    bool useLazy = kUseLazy[level];

    int64_t position = 0;
    int     heldLength = 0;          /* a match found at position - 1, not yet emitted */
    int64_t heldDistance = 0;
    bool    holding = false;

    while (position < length) {
        int64_t chainStart = tableInsert(table, data, length, position);
        int64_t distance = 0;
        int found = findMatch(table, data, length, position, chainStart, chainLimit,
                              niceLength, &distance);

        if (holding) {
            if (found > heldLength) {
                /* One byte later buys a longer match, so the byte we were
                 * holding is worth more as a literal than as the start of the
                 * shorter one. */
                writeSymbol(writer, data[position - 1]);
                heldLength = found;
                heldDistance = distance;
                position++;
                continue;
            }
            writeMatch(writer, heldLength, heldDistance);
            int64_t end = position - 1 + heldLength;
            for (int64_t i = position + 1; i < end; i++)
                (void)tableInsert(table, data, length, i);
            position = end;
            holding = false;
            continue;
        }

        if (found >= MIN_MATCH) {
            if (useLazy && position + 1 < length && found < niceLength) {
                heldLength = found;
                heldDistance = distance;
                holding = true;
                position++;
                continue;
            }
            writeMatch(writer, found, distance);
            for (int64_t i = position + 1; i < position + found; i++)
                (void)tableInsert(table, data, length, i);
            position += found;
            continue;
        }

        writeSymbol(writer, data[position]);
        position++;
    }

    /* A held match is at least three bytes long, so it always leaves two behind
     * it and the loop cannot end while one is held. Emitting it anyway keeps
     * that argument off the critical path. */
    if (holding) writeMatch(writer, heldLength, heldDistance);

    writeSymbol(writer, 256);                          /* end of block */
    JAI_FREE(MatchTable, table);
}

/* ------------------------------------------------------------------ */
/* Adler-32                                                             */
/* ------------------------------------------------------------------ */

/* 5552 is the most bytes that can be summed before the second accumulator can
 * overflow 32 bits, which is what lets the modulo happen once per block. */
#define ADLER_BASE  65521u
#define ADLER_BLOCK 5552u

static uint32_t adler32Of(const uint8_t *data, size_t length) {
    uint32_t low = 1, high = 0;
    while (length > 0) {
        size_t block = length < ADLER_BLOCK ? length : ADLER_BLOCK;
        for (size_t i = 0; i < block; i++) {
            low += data[i];
            high += low;
        }
        low %= ADLER_BASE;
        high %= ADLER_BASE;
        data += block;
        length -= block;
    }
    return (high << 16) | low;
}

/* ------------------------------------------------------------------ */
/* The primitives                                                       */
/* ------------------------------------------------------------------ */

static bool compressArgBytes(Value v, int index, const char *fnName, ObjBytes **out) {
    if (!IS_BYTES(v)) return jaiBuiltinArgTypeError(index, fnName, "bytes", v);
    *out = AS_BYTES(v);
    return true;
}

/* deflate(data, level) -> bytes
 *
 * A complete zlib stream: the two-byte header of RFC 1950, one DEFLATE stream,
 * and the Adler-32 of the input in big-endian order. That is exactly what a
 * PNG IDAT chunk carries. */
static bool nDeflate(int argc, Value *args, Value *out) {
    (void)argc;
    ObjBytes *data;
    int64_t level;
    if (!compressArgBytes(args[0], 1, "deflate", &data)) return false;
    if (!jaiArgInt(args[1], 2, "deflate", &level)) return false;
    if (level < 0 || level > 9) {
        return jaiThrow(vm.cValueError,
                        "deflate(): level must be between 0 and 9, got %lld",
                        (long long)level);
    }

    /* FLEVEL is advisory — a decompressor ignores it — but FCHECK is not: the
     * two header bytes read as a big-endian number have to be a multiple of
     * 31. */
    uint32_t flevel = level == 0 ? 0u : (level < 6 ? 1u : (level == 6 ? 2u : 3u));
    uint32_t header = (0x78u << 8) | (flevel << 6);    /* deflate, 32 KB window */
    header += 31u - (header % 31u);

    BitWriter writer = {NULL, 0, 0, 0, 0};
    writerPush(&writer, (uint8_t)((header >> 8) & 0xFFu));
    writerPush(&writer, (uint8_t)(header & 0xFFu));

    size_t blockStart = writer.count;
    if (level == 0) {
        writeStoredBlocks(&writer, data->data, data->length);
    } else {
        writeFixedBlock(&writer, data->data, (int64_t)data->length, (int)level);
        writerAlign(&writer);
        /* Half of the fixed code's literals are nine bits, so data with no
         * matches in it comes out about five percent larger than it went in.
         * Storing it is smaller, just as valid, and what the caller wanted when
         * it asked for compression. */
        if (writer.count - blockStart > storedSize(data->length)) {
            writer.count = blockStart;
            writeStoredBlocks(&writer, data->data, data->length);
        }
    }

    writerAlign(&writer);
    uint32_t checksum = adler32Of(data->data, data->length);
    writerPush(&writer, (uint8_t)((checksum >> 24) & 0xFFu));
    writerPush(&writer, (uint8_t)((checksum >> 16) & 0xFFu));
    writerPush(&writer, (uint8_t)((checksum >> 8) & 0xFFu));
    writerPush(&writer, (uint8_t)(checksum & 0xFFu));

    ObjBytes *result = jaiBytesNew(writer.bytes, writer.count);
    JAI_FREE_ARRAY(uint8_t, writer.bytes, writer.capacity);
    if (result == NULL) return false;                  /* too long; already raised */
    *out = OBJ_VAL(result);
    return true;
}

/* adler32(data) -> int
 *
 * The checksum RFC 1950 puts at the end of a zlib stream: two 16-bit sums
 * modulo 65521, packed high then low. */
static bool nAdler32(int argc, Value *args, Value *out) {
    (void)argc;
    ObjBytes *data;
    if (!compressArgBytes(args[0], 1, "adler32", &data)) return false;
    *out = INT_VAL((int64_t)adler32Of(data->data, data->length));
    return true;
}


/* crc32(data)
 *
 * The CRC-32 PNG puts on every chunk. Table-driven, and native for the same
 * reason adler32 is: a bit-at-a-time loop in Jaithon runs eight iterations per
 * byte, and a megabyte of image data makes that the most expensive part of
 * writing a file whose actual compression is already native. */
static uint32_t sCrcTable[256];
static bool sCrcReady = false;

static void crcInit(void) {
    for (uint32_t n = 0; n < 256; n++) {
        uint32_t c = n;
        for (int k = 0; k < 8; k++) {
            c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        }
        sCrcTable[n] = c;
    }
    sCrcReady = true;
}

static bool nCrc32(int argc, Value *args, Value *out) {
    (void)argc;
    if (!sCrcReady) crcInit();

    ObjBytes *data;
    if (!compressArgBytes(args[0], 1, "crc32", &data)) return false;
    uint32_t crc = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < data->length; i++) {
        crc = sCrcTable[(crc ^ data->data[i]) & 0xFFu] ^ (crc >> 8);
    }
    *out = INT_VAL((int64_t)(crc ^ 0xFFFFFFFFu));
    return true;
}

void jaiRegisterCompressPrimitives(void) {
    jaiDefineNative("__prim__.deflate",  nDeflate,  2, 2);
    jaiDefineNative("__prim__.crc32",   nCrc32,   1, 1);
    jaiDefineNative("__prim__.adler32",  nAdler32,  1, 1);
}
