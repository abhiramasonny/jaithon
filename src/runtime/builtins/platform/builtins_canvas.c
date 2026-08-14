/* builtins_canvas.c — __prim__.canvas_*, the bulk pixel operations behind
 * std.gui.canvas: scanline fills, blends and blits over its `list[int]` back
 * buffer. Compositing matches `blend` in lib/std/gui/canvas.jai byte for byte,
 * and every write is bounds- and type-checked before anything is written, so a
 * failed call leaves the buffer untouched rather than half drawn. */

#include <math.h>

#include "runtime/runtime.h"

/* A pixel is a packed 0xAARRGGBB value: 32 bits, nothing above them. */
#define CANVAS_PIXEL_MAX 0xFFFFFFFFLL

/* ------------------------------------------------------------------ */
/* Compositing                                                          */
/* ------------------------------------------------------------------ */

static uint32_t canvasOver(uint32_t source, uint32_t destination,
                           uint32_t alpha, uint32_t inverse) {
    return (source * alpha + destination * inverse + 127u) / 255u;
}

static uint32_t canvasBlend(uint32_t destination, uint32_t source) {
    uint32_t alpha = (source >> 24) & 0xFFu;
    if (alpha == 255u) return source;
    if (alpha == 0u) return destination;

    uint32_t inverse = 255u - alpha;
    uint32_t red = canvasOver((source >> 16) & 0xFFu, (destination >> 16) & 0xFFu,
                              alpha, inverse);
    uint32_t green = canvasOver((source >> 8) & 0xFFu, (destination >> 8) & 0xFFu,
                                alpha, inverse);
    uint32_t blue = canvasOver(source & 0xFFu, destination & 0xFFu, alpha, inverse);
    uint32_t outAlpha = alpha + ((((destination >> 24) & 0xFFu) * inverse + 127u) / 255u);
    if (outAlpha > 255u) outAlpha = 255u;
    return (outAlpha << 24) | (red << 16) | (green << 8) | blue;
}

/* ------------------------------------------------------------------ */
/* Argument checking                                                    */
/* ------------------------------------------------------------------ */

static bool canvasArgColor(Value v, int index, const char *fnName, uint32_t *out) {
    int64_t raw;
    if (!jaiArgInt(v, index, fnName, &raw)) return false;
    if (raw < 0 || raw > CANVAS_PIXEL_MAX) {
        return jaiThrow(vm.cValueError,
                        "%s(): colour must be a packed 0xAARRGGBB value between 0 "
                        "and %lld, got %lld",
                        fnName, (long long)CANVAS_PIXEL_MAX, (long long)raw);
    }
    *out = (uint32_t)raw;
    return true;
}

static bool canvasCheckSpan(const ObjList *pixels, int64_t offset, int64_t count,
                            const char *fnName) {
    if (count < 0) {
        return jaiThrow(vm.cValueError, "%s(): count must be non-negative, got %lld",
                        fnName, (long long)count);
    }
    if (offset < 0 || offset > (int64_t)pixels->count ||
        count > (int64_t)pixels->count - offset) {
        return jaiThrow(vm.cIndexError,
                        "%s(): %lld pixels from offset %lld leave a buffer of %d "
                        "pixels",
                        fnName, (long long)count, (long long)offset, pixels->count);
    }
    return true;
}

static bool canvasCheckShape(const ObjList *pixels, int64_t width, int64_t height,
                             const char *fnName, const char *role) {
    if (width <= 0 || height <= 0) {
        return jaiThrow(vm.cValueError,
                        "%s(): the %s buffer must be at least 1x1, got %lldx%lld",
                        fnName, role, (long long)width, (long long)height);
    }
    if (width > (int64_t)pixels->count || height > (int64_t)pixels->count ||
        width * height != (int64_t)pixels->count) {
        return jaiThrow(vm.cValueError,
                        "%s(): the %s buffer holds %d pixels, not the %lldx%lld it "
                        "was described as",
                        fnName, role, pixels->count, (long long)width,
                        (long long)height);
    }
    return true;
}

static bool canvasCheckRect(int64_t x, int64_t y, int64_t width, int64_t height,
                            int64_t bufferWidth, int64_t bufferHeight,
                            const char *fnName, const char *role) {
    if (width < 0 || height < 0) {
        return jaiThrow(vm.cValueError,
                        "%s(): the %s rectangle must have a non-negative size, got "
                        "%lldx%lld",
                        fnName, role, (long long)width, (long long)height);
    }
    if (x < 0 || y < 0 || x > bufferWidth || y > bufferHeight ||
        width > bufferWidth - x || height > bufferHeight - y) {
        return jaiThrow(vm.cIndexError,
                        "%s(): the %s rectangle %lldx%lld at (%lld, %lld) leaves a "
                        "%lldx%lld buffer",
                        fnName, role, (long long)width, (long long)height,
                        (long long)x, (long long)y, (long long)bufferWidth,
                        (long long)bufferHeight);
    }
    return true;
}

static bool canvasCheckRectValues(const ObjList *pixels, int64_t bufferWidth,
                                  int64_t x, int64_t y, int64_t width, int64_t height,
                                  const char *fnName, const char *role) {
    for (int64_t row = 0; row < height; row++) {
        int64_t base = (y + row) * bufferWidth + x;
        for (int64_t column = 0; column < width; column++) {
            Value pixel = pixels->items[base + column];
            if (!IS_INT(pixel)) {
                return jaiThrow(vm.cTypeError,
                                "%s(): %s pixel %lld is %s, expected a packed "
                                "0xAARRGGBB int",
                                fnName, role, (long long)(base + column),
                                jaiTypeNameStatic(pixel));
            }
        }
    }
    return true;
}

static uint32_t canvasPixelAt(const ObjList *pixels, int64_t index) {
    return (uint32_t)(uint64_t)AS_INT(pixels->items[index]);
}

/* ------------------------------------------------------------------ */
/* Spans                                                                */
/* ------------------------------------------------------------------ */

static bool nCanvasFillSpan(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *pixels;
    int64_t offset, count;
    uint32_t color;
    if (!jaiArgList(args[0], 1, "canvas_fill_span", &pixels)) return false;
    if (!jaiArgInt(args[1], 2, "canvas_fill_span", &offset)) return false;
    if (!jaiArgInt(args[2], 3, "canvas_fill_span", &count)) return false;
    if (!canvasArgColor(args[3], 4, "canvas_fill_span", &color)) return false;
    if (!canvasCheckSpan(pixels, offset, count, "canvas_fill_span")) return false;

    Value packed = INT_VAL((int64_t)color);
    for (int64_t i = 0; i < count; i++) pixels->items[offset + i] = packed;
    if (count > 0) jaiListTouch(pixels);

    *out = NULL_VAL;
    return true;
}

static bool nCanvasWriteSpan(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *pixels, *source;
    int64_t offset;
    if (!jaiArgList(args[0], 1, "canvas_write_span", &pixels)) return false;
    if (!jaiArgInt(args[1], 2, "canvas_write_span", &offset)) return false;
    if (!jaiArgList(args[2], 3, "canvas_write_span", &source)) return false;
    if (!canvasCheckSpan(pixels, offset, source->count, "canvas_write_span"))
        return false;

    for (int i = 0; i < source->count; i++) {
        Value pixel = source->items[i];
        if (!IS_INT(pixel)) {
            return jaiThrow(vm.cTypeError,
                            "canvas_write_span(): source %d is %s, expected an int",
                            i, jaiTypeNameStatic(pixel));
        }
        pixels->items[offset + i] = pixel;
    }
    if (source->count > 0) jaiListTouch(pixels);

    *out = NULL_VAL;
    return true;
}

static bool nCanvasScanlines(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *pixels;
    int64_t width, height;
    if (!jaiArgList(args[0], 1, "canvas_scanlines", &pixels)) return false;
    if (!jaiArgInt(args[1], 2, "canvas_scanlines", &width)) return false;
    if (!jaiArgInt(args[2], 3, "canvas_scanlines", &height)) return false;

    if (width <= 0 || height <= 0) {
        return jaiThrow(vm.cValueError,
                        "canvas_scanlines(): size must be positive, got %lldx%lld",
                        (long long)width, (long long)height);
    }
    if ((int64_t)pixels->count < width * height) {
        return jaiThrow(vm.cValueError,
                        "canvas_scanlines(): a %lldx%lld image needs %lld pixels, got %d",
                        (long long)width, (long long)height,
                        (long long)(width * height), pixels->count);
    }

    size_t stride = (size_t)width * 4u + 1u;
    size_t total = stride * (size_t)height;
    uint8_t *raw = (uint8_t *)jaiRealloc(NULL, 0, total);
    if (raw == NULL) {
        return jaiThrow(vm.cRuntimeError,
                        "canvas_scanlines(): out of memory for %zu bytes", total);
    }

    for (int64_t row = 0; row < height; row++) {
        uint8_t *line = raw + (size_t)row * stride;
        *line++ = 0;
        const Value *source = pixels->items + (size_t)row * (size_t)width;
        for (int64_t column = 0; column < width; column++) {
            Value pixel = source[column];
            if (!IS_INT(pixel)) {
                (void)jaiRealloc(raw, total, 0);
                return jaiThrow(vm.cTypeError,
                                "canvas_scanlines(): pixel %lld is %s, expected an int",
                                (long long)(row * width + column),
                                jaiTypeNameStatic(pixel));
            }
            uint32_t packed = (uint32_t)AS_INT(pixel);
            *line++ = (uint8_t)(packed >> 16);
            *line++ = (uint8_t)(packed >> 8);
            *line++ = (uint8_t)packed;
            *line++ = (uint8_t)(packed >> 24);
        }
    }

    ObjBytes *result = jaiBytesNew(raw, total);
    (void)jaiRealloc(raw, total, 0);
    if (result == NULL) {
        return jaiThrow(vm.cRuntimeError, "canvas_scanlines(): out of memory");
    }
    *out = OBJ_VAL(result);
    return true;
}

static bool nCanvasFillConvex(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *pixels, *points;
    int64_t width, height, clipLeft, clipTop, clipRight, clipBottom;
    uint32_t color;
    if (!jaiArgList(args[0], 1, "canvas_fill_convex", &pixels)) return false;
    if (!jaiArgInt(args[1], 2, "canvas_fill_convex", &width)) return false;
    if (!jaiArgInt(args[2], 3, "canvas_fill_convex", &height)) return false;
    if (!jaiArgList(args[3], 4, "canvas_fill_convex", &points)) return false;
    if (!canvasArgColor(args[4], 5, "canvas_fill_convex", &color)) return false;
    if (!jaiArgInt(args[5], 6, "canvas_fill_convex", &clipLeft)) return false;
    if (!jaiArgInt(args[6], 7, "canvas_fill_convex", &clipTop)) return false;
    if (!jaiArgInt(args[7], 8, "canvas_fill_convex", &clipRight)) return false;
    if (!jaiArgInt(args[8], 9, "canvas_fill_convex", &clipBottom)) return false;

    if (width <= 0 || height <= 0) {
        return jaiThrow(vm.cValueError,
                        "canvas_fill_convex(): size must be positive, got %lldx%lld",
                        (long long)width, (long long)height);
    }
    if ((int64_t)pixels->count < width * height) {
        return jaiThrow(vm.cValueError,
                        "canvas_fill_convex(): a %lldx%lld buffer needs %lld pixels, got %d",
                        (long long)width, (long long)height,
                        (long long)(width * height), pixels->count);
    }
    if (points->count < 6 || (points->count & 1) != 0) {
        return jaiThrow(vm.cValueError,
                        "canvas_fill_convex(): need an even count of at least 6 "
                        "coordinates, got %d", points->count);
    }

    int corners = points->count / 2;
    double *xs = (double *)jaiRealloc(NULL, 0, (size_t)corners * 2 * sizeof(double));
    if (xs == NULL) {
        return jaiThrow(vm.cRuntimeError, "canvas_fill_convex(): out of memory");
    }
    double *ys = xs + corners;
    for (int i = 0; i < corners; i++) {
        Value vx = points->items[i * 2];
        Value vy = points->items[i * 2 + 1];
        if (!IS_FLOAT(vx) && !IS_INT(vx)) { goto bad_coordinate; }
        if (!IS_FLOAT(vy) && !IS_INT(vy)) { goto bad_coordinate; }
        xs[i] = IS_FLOAT(vx) ? AS_FLOAT(vx) : (double)AS_INT(vx);
        ys[i] = IS_FLOAT(vy) ? AS_FLOAT(vy) : (double)AS_INT(vy);
        continue;
    bad_coordinate:
        (void)jaiRealloc(xs, (size_t)corners * 2 * sizeof(double), 0);
        return jaiThrow(vm.cTypeError,
                        "canvas_fill_convex(): coordinate %d is not a number", i * 2);
    }

    double top = ys[0], bottom = ys[0], left = xs[0], right = xs[0];
    for (int i = 1; i < corners; i++) {
        if (ys[i] < top) top = ys[i];
        if (ys[i] > bottom) bottom = ys[i];
        if (xs[i] < left) left = xs[i];
        if (xs[i] > right) right = xs[i];
    }

    int64_t firstRow = (int64_t)floor(top);
    int64_t lastRow = (int64_t)ceil(bottom);
    if (firstRow < clipTop) firstRow = clipTop;
    if (lastRow > clipBottom) lastRow = clipBottom;
    if (firstRow < 0) firstRow = 0;
    if (lastRow > height - 1) lastRow = height - 1;

    int64_t firstColumn = clipLeft < 0 ? 0 : clipLeft;
    int64_t lastColumn = clipRight > width - 1 ? width - 1 : clipRight;

    const int SAMPLES = 4;
    double share = 255.0 / (double)SAMPLES;
    uint32_t sourceAlpha = (color >> 24) & 0xFFu;
    uint32_t sourceRed = (color >> 16) & 0xFFu;
    uint32_t sourceGreen = (color >> 8) & 0xFFu;
    uint32_t sourceBlue = color & 0xFFu;

    for (int64_t row = firstRow; row <= lastRow; row++) {
        double lows[8], highs[8];
        int found = 0;
        for (int sample = 0; sample < SAMPLES; sample++) {
            double y = (double)row + ((double)sample + 0.5) / (double)SAMPLES;
            double lo = 0.0, hi = 0.0;
            int crossed = 0;
            for (int i = 0; i < corners; i++) {
                int j = (i + 1) % corners;
                double y0 = ys[i], y1 = ys[j];
                if (y0 == y1) continue;
                double lower = y0 < y1 ? y0 : y1;
                double upper = y0 < y1 ? y1 : y0;
                if (y < lower || y >= upper) continue;
                double at = xs[i] + (y - y0) * (xs[j] - xs[i]) / (y1 - y0);
                if (!crossed) { lo = at; hi = at; crossed = 1; }
                else if (at < lo) lo = at;
                else if (at > hi) hi = at;
            }
            if (!crossed || hi <= lo) continue;
            lows[found] = lo;
            highs[found] = hi;
            found++;
        }
        if (found == 0) continue;

        double rowLeft = lows[0], rowRight = highs[0];
        for (int i = 1; i < found; i++) {
            if (lows[i] < rowLeft) rowLeft = lows[i];
            if (highs[i] > rowRight) rowRight = highs[i];
        }
        int64_t from = (int64_t)floor(rowLeft);
        int64_t to = (int64_t)ceil(rowRight);
        if (from < firstColumn) from = firstColumn;
        if (to > lastColumn) to = lastColumn;
        if (from > to) continue;

        Value *line = pixels->items + (size_t)row * (size_t)width;
        for (int64_t column = from; column <= to; column++) {
            double coverage = 0.0;
            for (int i = 0; i < found; i++) {
                double a = lows[i] > (double)column ? lows[i] : (double)column;
                double b = highs[i] < (double)column + 1.0 ? highs[i] : (double)column + 1.0;
                if (b > a) coverage += share * (b - a);
            }
            if (coverage <= 0.5) continue;
            uint32_t weight = coverage >= 255.0 ? 255u : (uint32_t)(coverage + 0.5);
            uint32_t alpha = (sourceAlpha * weight + 127u) / 255u;
            if (alpha == 0) continue;

            Value existing = line[column];
            uint32_t destination = IS_INT(existing) ? (uint32_t)AS_INT(existing) : 0u;
            if (alpha == 255u) {
                line[column] = INT_VAL((int64_t)(0xFF000000u | (sourceRed << 16)
                                                 | (sourceGreen << 8) | sourceBlue));
                continue;
            }
            uint32_t inverse = 255u - alpha;
            uint32_t dr = (destination >> 16) & 0xFFu;
            uint32_t dg = (destination >> 8) & 0xFFu;
            uint32_t db = destination & 0xFFu;
            uint32_t da = (destination >> 24) & 0xFFu;
            uint32_t r = (sourceRed * alpha + dr * inverse + 127u) / 255u;
            uint32_t g = (sourceGreen * alpha + dg * inverse + 127u) / 255u;
            uint32_t b = (sourceBlue * alpha + db * inverse + 127u) / 255u;
            uint32_t a = alpha + (da * inverse + 127u) / 255u;
            if (a > 255u) a = 255u;
            line[column] = INT_VAL((int64_t)((a << 24) | (r << 16) | (g << 8) | b));
        }
    }

    (void)jaiRealloc(xs, (size_t)corners * 2 * sizeof(double), 0);
    jaiListTouch(pixels);
    *out = NULL_VAL;
    return true;
}

static bool nCanvasBlendSpan(int argc, Value *args, Value *out) {
    (void)argc;
    ObjList *pixels, *coverage;
    int64_t offset;
    uint32_t color;
    if (!jaiArgList(args[0], 1, "canvas_blend_span", &pixels)) return false;
    if (!jaiArgInt(args[1], 2, "canvas_blend_span", &offset)) return false;
    if (!jaiArgList(args[2], 3, "canvas_blend_span", &coverage)) return false;
    if (!canvasArgColor(args[3], 4, "canvas_blend_span", &color)) return false;
    if (!canvasCheckSpan(pixels, offset, coverage->count, "canvas_blend_span"))
        return false;

    for (int i = 0; i < coverage->count; i++) {
        Value weight = coverage->items[i];
        if (!IS_INT(weight)) {
            return jaiThrow(vm.cTypeError,
                            "canvas_blend_span(): coverage %d is %s, expected an "
                            "int between 0 and 255",
                            i, jaiTypeNameStatic(weight));
        }
        if (AS_INT(weight) < 0 || AS_INT(weight) > 255) {
            return jaiThrow(vm.cValueError,
                            "canvas_blend_span(): coverage %d is %lld, expected a "
                            "value between 0 and 255",
                            i, (long long)AS_INT(weight));
        }
        Value pixel = pixels->items[offset + i];
        if (!IS_INT(pixel)) {
            return jaiThrow(vm.cTypeError,
                            "canvas_blend_span(): pixel %lld is %s, expected a "
                            "packed 0xAARRGGBB int",
                            (long long)(offset + i), jaiTypeNameStatic(pixel));
        }
    }

    uint32_t alpha = (color >> 24) & 0xFFu;
    if (alpha == 0u || coverage->count == 0) {
        *out = NULL_VAL;
        return true;
    }

    for (int i = 0; i < coverage->count; i++) {
        uint32_t weight = (uint32_t)AS_INT(coverage->items[i]);
        if (weight == 0u) continue;
        /* Rounded, like every channel op here; exact at weight 255. */
        uint32_t scaled = (alpha * weight + 127u) / 255u;
        if (scaled == 0u) continue;
        uint32_t source = (color & 0x00FFFFFFu) | (scaled << 24);
        int64_t at = offset + i;
        pixels->items[at] = INT_VAL((int64_t)canvasBlend(canvasPixelAt(pixels, at),
                                                         source));
    }
    jaiListTouch(pixels);

    *out = NULL_VAL;
    return true;
}

/* ------------------------------------------------------------------ */
/* Rectangles                                                           */
/* ------------------------------------------------------------------ */

static bool nCanvasBlitScaled(int argc, Value *args, Value *out) {
    (void)argc;
    const char *fnName = "canvas_blit_scaled";
    ObjList *dst, *src;
    int64_t dstWidth, srcWidth, srcHeight;
    int64_t sx, sy, sw, sh, dx, dy, dw, dh;

    if (!jaiArgList(args[0], 1, fnName, &dst)) return false;
    if (!jaiArgInt(args[1], 2, fnName, &dstWidth)) return false;
    if (!jaiArgList(args[2], 3, fnName, &src)) return false;
    if (!jaiArgInt(args[3], 4, fnName, &srcWidth)) return false;
    if (!jaiArgInt(args[4], 5, fnName, &srcHeight)) return false;
    if (!jaiArgInt(args[5], 6, fnName, &sx)) return false;
    if (!jaiArgInt(args[6], 7, fnName, &sy)) return false;
    if (!jaiArgInt(args[7], 8, fnName, &sw)) return false;
    if (!jaiArgInt(args[8], 9, fnName, &sh)) return false;
    if (!jaiArgInt(args[9], 10, fnName, &dx)) return false;
    if (!jaiArgInt(args[10], 11, fnName, &dy)) return false;
    if (!jaiArgInt(args[11], 12, fnName, &dw)) return false;
    if (!jaiArgInt(args[12], 13, fnName, &dh)) return false;

    if (!canvasCheckShape(src, srcWidth, srcHeight, fnName, "source")) return false;
    if (dstWidth <= 0) {
        return jaiThrow(vm.cValueError,
                        "%s(): dst_width must be positive, got %lld", fnName,
                        (long long)dstWidth);
    }
    if ((int64_t)dst->count % dstWidth != 0) {
        return jaiThrow(vm.cValueError,
                        "%s(): the destination buffer holds %d pixels, which is not "
                        "a whole number of %lld-pixel rows",
                        fnName, dst->count, (long long)dstWidth);
    }
    int64_t dstHeight = (int64_t)dst->count / dstWidth;

    if (!canvasCheckRect(sx, sy, sw, sh, srcWidth, srcHeight, fnName, "source"))
        return false;
    if (!canvasCheckRect(dx, dy, dw, dh, dstWidth, dstHeight, fnName, "destination"))
        return false;

    if (sw == 0 || sh == 0 || dw == 0 || dh == 0) {
        *out = NULL_VAL;
        return true;
    }

    if (!canvasCheckRectValues(src, srcWidth, sx, sy, sw, sh, fnName, "source"))
        return false;
    if (!canvasCheckRectValues(dst, dstWidth, dx, dy, dw, dh, fnName, "destination"))
        return false;

    for (int64_t row = 0; row < dh; row++) {
        int64_t sourceBase = (sy + (row * sh) / dh) * srcWidth + sx;
        int64_t targetBase = (dy + row) * dstWidth + dx;
        for (int64_t column = 0; column < dw; column++) {
            uint32_t source = canvasPixelAt(src, sourceBase + (column * sw) / dw);
            if ((source >> 24) == 0u) continue;
            int64_t at = targetBase + column;
            dst->items[at] = INT_VAL((int64_t)canvasBlend(canvasPixelAt(dst, at),
                                                          source));
        }
    }
    jaiListTouch(dst);

    *out = NULL_VAL;
    return true;
}

static bool nCanvasDownsampleBox(int argc, Value *args, Value *out) {
    (void)argc;
    const char *fnName = "canvas_downsample_box";
    ObjList *dst, *src;
    int64_t width, height, factor;

    if (!jaiArgList(args[0], 1, fnName, &dst)) return false;
    if (!jaiArgList(args[1], 2, fnName, &src)) return false;
    if (!jaiArgInt(args[2], 3, fnName, &width)) return false;
    if (!jaiArgInt(args[3], 4, fnName, &height)) return false;
    if (!jaiArgInt(args[4], 5, fnName, &factor)) return false;

    if (!canvasCheckShape(src, width, height, fnName, "source")) return false;
    if (factor <= 0) {
        return jaiThrow(vm.cValueError, "%s(): factor must be positive, got %lld",
                        fnName, (long long)factor);
    }

    int64_t outWidth = width / factor;
    int64_t outHeight = height / factor;
    if ((int64_t)dst->count != outWidth * outHeight) {
        return jaiThrow(vm.cValueError,
                        "%s(): reducing %lldx%lld by %lld needs a destination of "
                        "%lld pixels, got %d",
                        fnName, (long long)width, (long long)height,
                        (long long)factor, (long long)(outWidth * outHeight),
                        dst->count);
    }
    if (outWidth == 0 || outHeight == 0) {
        *out = NULL_VAL;
        return true;
    }
    if (!canvasCheckRectValues(src, width, 0, 0, outWidth * factor,
                               outHeight * factor, fnName, "source"))
        return false;

    for (int64_t row = 0; row < outHeight; row++) {
        for (int64_t column = 0; column < outWidth; column++) {
            uint64_t alphaSum = 0, redSum = 0, greenSum = 0, blueSum = 0;
            for (int64_t inner = 0; inner < factor; inner++) {
                int64_t base = (row * factor + inner) * width + column * factor;
                for (int64_t step = 0; step < factor; step++) {
                    uint32_t pixel = canvasPixelAt(src, base + step);
                    uint64_t alpha = (pixel >> 24) & 0xFFu;
                    alphaSum += alpha;
                    redSum += ((pixel >> 16) & 0xFFu) * alpha;
                    greenSum += ((pixel >> 8) & 0xFFu) * alpha;
                    blueSum += (pixel & 0xFFu) * alpha;
                }
            }
            uint64_t samples = (uint64_t)factor * (uint64_t)factor;
            uint64_t packed = 0;
            if (alphaSum > 0) {
                uint64_t half = alphaSum / 2;
                packed = (((alphaSum + samples / 2) / samples) << 24) |
                         (((redSum + half) / alphaSum) << 16) |
                         (((greenSum + half) / alphaSum) << 8) |
                         ((blueSum + half) / alphaSum);
            }
            dst->items[row * outWidth + column] = INT_VAL((int64_t)packed);
        }
    }
    jaiListTouch(dst);

    *out = NULL_VAL;
    return true;
}

void jaiRegisterCanvasPrimitives(void) {
    jaiDefineNative("__prim__.canvas_fill_span",      nCanvasFillSpan,      4, 4);
    jaiDefineNative("__prim__.canvas_blend_span",     nCanvasBlendSpan,     4, 4);
    jaiDefineNative("__prim__.canvas_write_span",     nCanvasWriteSpan,     3, 3);
    jaiDefineNative("__prim__.canvas_scanlines",      nCanvasScanlines,     3, 3);
    jaiDefineNative("__prim__.canvas_fill_convex",    nCanvasFillConvex,    9, 9);
    jaiDefineNative("__prim__.canvas_blit_scaled",    nCanvasBlitScaled,   13, 13);
    jaiDefineNative("__prim__.canvas_downsample_box", nCanvasDownsampleBox, 5, 5);
}
