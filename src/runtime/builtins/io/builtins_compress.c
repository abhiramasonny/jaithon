/* builtins_compress.c — __prim__.deflate/adler32/crc32: a native DEFLATE (RFC 1951 fixed Huffman) and checksums, since zlib is not guaranteed to be linked and PNG output needs it. */

#include "runtime/builtins/builtins.h"
#include "runtime/runtime.h"

/* DEFLATE fills a byte LSB-first but a Huffman code is written MSB-first, so
 * writerCode reverses one before handing it over; getting this backwards
 * produces a stream that looks plausible and decodes to nothing. */
typedef struct {
    uint8_t *bytes;
    size_t   count;
    size_t   capacity;
    uint32_t bits;
    int      bitCount;    /* how many of them are live, always 0-7 between calls */
} BitWriter;

static void writerReserve(BitWriter *writer, size_t extra) {
    if (writer->count + extra <= writer->capacity) return;
    size_t capacity = writer->capacity < 512 ? 512 : writer->capacity;
    while (capacity < writer->count + extra) capacity *= 2;
    writer->bytes = JAI_GROW_ARRAY(uint8_t, writer->bytes, writer->capacity, capacity);
    writer->capacity = capacity;
}

/* Callers other than writerBits place a byte-aligned field (a stored block's
 * header, the zlib wrapper) and must have flushed the partial byte first. */
static void writerPush(BitWriter *writer, uint8_t byte) {
    writerReserve(writer, 1);
    writer->bytes[writer->count++] = byte;
}

/* Never called with more than 16 bits, which is what keeps the accumulator
 * from overflowing while a partial byte (up to 7 bits) is still pending. */
static void writerBits(BitWriter *writer, uint32_t value, int count) {
    writer->bits |= (value & ((1u << count) - 1u)) << writer->bitCount;
    writer->bitCount += count;
    while (writer->bitCount >= 8) {
        writerPush(writer, (uint8_t)(writer->bits & 0xFFu));
        writer->bits >>= 8;
        writer->bitCount -= 8;
    }
}

/* A stored block and the zlib trailer must both start on a byte boundary. */
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

#define WINDOW_SIZE 32768
#define WINDOW_MASK (WINDOW_SIZE - 1)
#define HASH_BITS   15
#define HASH_SIZE   (1 << HASH_BITS)
#define MIN_MATCH   3
#define MAX_MATCH   258

/* head[hash]: most recent position with that hash, or -1. prev[pos & MASK]:
 * previous position sharing it. A slot is only overwritten a full window
 * later, so any entry still in-window is current -- the distance check below
 * is enough on its own to reject a stale one. */
typedef struct {
    int32_t head[HASH_SIZE];
    int32_t prev[WINDOW_SIZE];
} MatchTable;

static uint32_t hashAt(const uint8_t *data) {
    return (uint32_t)(((uint32_t)data[0] << 10) ^ ((uint32_t)data[1] << 5) ^
                      (uint32_t)data[2]) & (HASH_SIZE - 1u);
}

/* Returns the prior position for this hash (the search start point) -- the
 * chain must not begin with `position` itself, or the first match found is
 * the string against itself at distance zero, which DEFLATE has no code for. */
static int32_t tableInsert(MatchTable *table, const uint8_t *data, int64_t length,
                           int64_t position) {
    if (position + MIN_MATCH > length) return -1;
    uint32_t slot = hashAt(data + position);
    int32_t previous = table->head[slot];
    table->prev[position & WINDOW_MASK] = previous;
    table->head[slot] = (int32_t)position;
    return previous;
}

/* Longest match at `position`, or 0 if none is worth coding. `chainLimit` and
 * `niceLength` are the two knobs the compression level tunes. */
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
        /* Cheapest test first: reject on the byte that would have to beat the
         * best match so far before comparing anything else. */
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

/* Cost of writeStoredBlocks: five header bytes per block plus the data. */
static size_t storedSize(size_t length) {
    size_t blocks = length == 0 ? 1 : (length + 65534) / 65535;
    return blocks * 5 + length;
}

/* Blocks are capped at 65535 bytes (the 16-bit length field). An empty input
 * still emits one empty block -- a zlib stream needs at least one. */
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

/* Level 1 takes the first match it sees; level 9 walks up to 4096 candidates
 * and defers each to check whether the next position does better. */
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
                /* One byte later buys a longer match, so the held byte is
                 * worth more as a literal than as the start of the shorter one. */
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

    /* A held match is >= 3 bytes, so the loop can never actually end while one
     * is held; emitting it anyway keeps that argument off the critical path. */
    if (holding) writeMatch(writer, heldLength, heldDistance);

    writeSymbol(writer, 256);                          /* end of block */
    JAI_FREE(MatchTable, table);
}

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

static bool compressArgBytes(Value v, int index, const char *fnName, ObjBytes **out) {
    if (!IS_BYTES(v)) return jaiBuiltinArgTypeError(index, fnName, "bytes", v);
    *out = AS_BYTES(v);
    return true;
}

/* deflate(data, level) -> bytes: a complete zlib stream (RFC 1950 header +
 * DEFLATE stream + big-endian Adler-32), exactly what a PNG IDAT chunk holds. */
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
        /* Data with no matches comes out ~5% larger under the fixed code (half
         * its literals are nine bits); fall back to stored if that happens. */
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

/* adler32(data) -> int: RFC 1950's zlib-trailer checksum -- two 16-bit sums
 * modulo 65521, packed high then low. */
static bool nAdler32(int argc, Value *args, Value *out) {
    (void)argc;
    ObjBytes *data;
    if (!compressArgBytes(args[0], 1, "adler32", &data)) return false;
    *out = INT_VAL((int64_t)adler32Of(data->data, data->length));
    return true;
}


/* crc32(data): PNG's per-chunk CRC-32, table-driven and native for the same
 * reason adler32 is -- a bit-at-a-time Jaithon loop would dominate runtime. */
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
