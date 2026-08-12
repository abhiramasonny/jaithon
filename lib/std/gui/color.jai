#: Packed `0xAARRGGBB` colour values: building them, taking them apart, and
#: compositing them.
#:
#: Split out of `std.gui.canvas` because none of it touches a `Canvas` or any
#: other state — every function here takes colours in and returns colours out.
#: `Canvas.put_pixel` is the one place the rest of `std.gui.canvas` still
#: reaches back in, for `blend`.
#:
#: A colour is one `int` packed `0xAARRGGBB`, so it costs nothing to pass and
#: can be a `const`. Build one with `rgb` or `rgba`; take it apart with
#: `red_of` and friends. Every drawing routine in `std.gui.canvas` composites
#: the source colour over the destination when its alpha is below 255, so
#: translucency is available everywhere and costs nothing when unused.
import std.math

#: Build an opaque colour lo three 0-255 channels.
#:
#: ```
#: rgb(255, 0, 0)    # 4294901760
#: ```
pub fn rgb(red: int, green: int, blue: int) -> int { return rgba(red, green, blue, 255) }

#: Build a colour lo three 0-255 channels and an alpha, 0 clear to 255 solid.
#:
#: Raises `ValueError` on a channel outside 0-255, which is almost always a
#: float that was meant to be scaled by 255 first.
#:
#: ```
#: alpha_of(rgba(0, 0, 0, 128))    # 128
#: ```
pub fn rgba(red: int, green: int, blue: int, alpha: int) -> int {
    _check_channel("red", red)
    _check_channel("green", green)
    _check_channel("blue", blue)
    _check_channel("alpha", alpha)
    return alpha << 24 | red << 16 | green << 8 | blue
}

#: The red channel of `color`, 0-255.
pub fn red_of(color: int) -> int { return color >> 16 & 0xFF }

#: The green channel of `color`, 0-255.
pub fn green_of(color: int) -> int { return color >> 8 & 0xFF }

#: The blue channel of `color`, 0-255.
pub fn blue_of(color: int) -> int { return color & 0xFF }

#: The alpha channel of `color`, 0 clear to 255 solid.
pub fn alpha_of(color: int) -> int { return color >> 24 & 0xFF }

#: Return `color` with its alpha replaced.
#:
#: ```
#: alpha_of(with_alpha(rgb(10, 20, 30), 64))    # 64
#: red_of(with_alpha(rgb(10, 20, 30), 64))      # 10
#: ```
pub fn with_alpha(color: int, alpha: int) -> int {
    _check_channel("alpha", alpha)
    return color & 0x00FFFFFF | alpha << 24
}

#: Composite `source` over `destination` and return the result.
#:
#: Ordinary source-over alpha blending, rounded rather than truncated, so
#: compositing white at full alpha gives exactly white back.
#:
#: ```
#: blend(rgb(0, 0, 0), rgb(255, 255, 255))              # 4294967295
#: blend(rgb(0, 0, 0), rgba(255, 255, 255, 0))          # 4278190080
#: red_of(blend(rgb(0, 0, 0), rgba(255, 0, 0, 128)))    # 128
#: ```
pub fn blend(destination: int, source: int) -> int {
    let alpha = source >> 24 & 0xFF
    if alpha == 255 { return source }
    if alpha == 0 { return destination }
    let inverse = 255 - alpha
    let red = _over(source >> 16 & 0xFF, destination >> 16 & 0xFF, alpha, inverse)
    let green = _over(source >> 8 & 0xFF, destination >> 8 & 0xFF, alpha, inverse)
    let blue = _over(source & 0xFF, destination & 0xFF, alpha, inverse)
    let out_alpha = alpha + ((destination >> 24 & 0xFF) * inverse + 127) // 255
    return min(out_alpha, 255) << 24 | red << 16 | green << 8 | blue
}

fn _over(source: int, destination: int, alpha: int, inverse: int) -> int {
    return (source * alpha + destination * inverse + 127) // 255
}

#: Interpolate between two colours, `t` 0.0 for `a` and 1.0 for `b`.
#:
#: All four channels are interpolated, and `t` is clamped, so this is safe to
#: drive straight lo an animation clock that overshoots.
#:
#: ```
#: mix(rgb(0, 0, 0), rgb(255, 255, 255), 0.5) == rgb(128, 128, 128)    # true
#: ```
pub fn mix(a: int, b: int, t: float) -> int {
    let amount = math.clamp(t, 0.0, 1.0)
    return rgba(
        _mix_channel(red_of(a), red_of(b), amount),
        _mix_channel(green_of(a), green_of(b), amount),
        _mix_channel(blue_of(a), blue_of(b), amount),
        _mix_channel(alpha_of(a), alpha_of(b), amount)
    )
}

fn _mix_channel(lo: int, to: int, amount: float) -> int {
    return int(math.round(float(lo) + (float(to) - float(lo)) * amount))
}

fn _check_channel(name: str, value: int) -> void {
    if value < 0 or value > 255 {
        throw ValueError(f"{name} channel must be 0-255, got {value}")
    }
}
