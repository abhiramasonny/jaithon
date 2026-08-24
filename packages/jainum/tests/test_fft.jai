#: Tests for `jainum.fft`.
#:
#: Nothing here is checked against numbers this package printed. The reference
#: is `reference_dft` below — the sum in the definition, written out on the
#: host with no factorisation and no shared code with the module — plus the
#: identities a transform has to satisfy whatever it is implemented as:
#: a round trip, Parseval, and the spectrum of a signal whose transform can be
#: worked out on paper.
#:
#: The lengths are chosen to reach each of the three paths on purpose: 8 and
#: 64 are the radix-4 chain, 16 hits the m = 1 scalar radix-4 stage, 32 adds
#: the leading radix-2 one, 12 and 20 and 45 are the mixed-radix stages, and
#: 13 and 31 and 7 have no factor the stages know so they can only come back
#: through Bluestein.
#:
#: The layout helpers are pure and run without a device; everything that
#: touches storage returns early when there is no GPU.
import std.gpu as gpu
import std.math
from std.test import assert_eq, assert_near, assert_throws, assert_true
from jainum.array import COMPLEX64, FLOAT32, NDArray, sl
from jainum.fft import conj, conjugate, fft, fft2, fftfreq, fftn, fftshift
from jainum.fft import full_spectrum, half_spectrum, hfft, ifft, ifft2, ifftn
from jainum.fft import ifftshift, ihfft, irfft, irfft2, is_fast_length
from jainum.fft import next_fast_length, next_power_of_two, radix_plan
from jainum.fft import rfft, rfft2, rfftfreq
from jainum.manipulate import transpose

fn have_gpu() -> bool { return gpu.is_available() }

#: A deterministic signal, so a failure is the same failure twice.
#:
#: A plain congruential generator rather than `random`: this file must not
#: depend on another module's idea of a distribution to decide whether a
#: transform is right.
fn signal(count: int, seed: int) -> list[float] {
    var state = seed
    var out: list[float] = []
    for _index in 0..count {
        state = (state * 1103515245 + 12345) % 2147483648
        out.push(float(state) / 1073741824.0 - 1.0)
    }
    return out
}

#: The transform, straight from its definition, on the host.
#:
#: Values are interleaved re/im both ways, which is what `COMPLEX64` hands
#: back from `to_list`. Quadratic and unashamed — it is the second, independent
#: computation, and it only ever runs over a few dozen points.
fn reference_dft(values: list[float], inverse: bool) -> list[float] {
    let n = values.len() // 2
    let sign = if inverse { 1.0 } else { -1.0 }
    var out: list[float] = []
    for k in 0..n {
        var re = 0.0
        var im = 0.0
        for j in 0..n {
            let angle = sign * 2.0 * math.PI * float(j) * float(k) / float(n)
            let c = math.cos(angle)
            let s = math.sin(angle)
            re += values[j * 2] * c - values[j * 2 + 1] * s
            im += values[j * 2] * s + values[j * 2 + 1] * c
        }
        out.push(re)
        out.push(im)
    }
    return out
}

#: Real values as the interleaved pairs `reference_dft` reads.
fn as_complex(values: list[float]) -> list[float] {
    var out: list[float] = []
    for value in values {
        out.push(value)
        out.push(0.0)
    }
    return out
}

fn magnitude(value: float) -> float { return if value < 0.0 { -value } else { value } }

#: Compare against the reference with a tolerance that follows the size of the
#: answer.
#:
#: A spectrum of an n-point signal has entries of order n times the input, so
#: a fixed absolute bound would either pass everything at n = 64 or fail
#: everything at n = 4. float32 carries about seven digits and a stage chain
#: sums in a tree, so 1e-5 of the largest entry is a real bound and not a
#: rubber one.
fn assert_lists_near(
    actual: list[float],
    expected: list[float],
    tolerance: float,
    what: str
) -> void {
    assert_eq(actual.len(), expected.len(), f"{what}: length")
    var scale = 1.0
    for value in expected {
        if magnitude(value) > scale { scale = magnitude(value) }
    }
    for index in 0..actual.len() {
        assert_near(actual[index], expected[index], tolerance * scale, f"{what} at {index}")
    }
}

fn complex_of(values: list[float]) -> NDArray {
    let a = NDArray([values.len() // 2], COMPLEX64)
    a.upload(values)
    return a
}

fn real_of(values: list[float]) -> NDArray {
    let a = NDArray([values.len()], FLOAT32)
    a.upload(values)
    return a
}

fn test_the_plan_covers_the_lengths_it_claims() -> void {
    assert_eq(radix_plan(1), [], "a single point needs no stage")
    assert_eq(radix_plan(2), [2])
    assert_eq(radix_plan(4), [4])
    assert_eq(radix_plan(8), [2, 4], "an odd exponent takes the radix-2 stage first")
    assert_eq(radix_plan(16), [4, 4])
    assert_eq(radix_plan(12), [4, 3], "the odd radices go last")
    assert_eq(radix_plan(1000), [2, 4, 5, 5, 5])

    #: Everything the chain claims must actually multiply back to the length,
    #: and the radix-4 stages must sit where their `m` is a power of two —
    #: which is the precondition of the paired float4 kernel.
    for n in [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20, 24, 32, 45, 64, 100, 1024] {
        let plan = radix_plan(n)
        assert_true(plan is not null, f"{n} is 5-smooth")
        var product = 1
        var m = 1
        if plan is not null {
            for radix in plan {
                if radix == 4 {
                    assert_eq(next_power_of_two(m), m, f"m is a power of two at {n}")
                }
                product *= radix
                m *= radix
            }
        }
        assert_eq(product, n, f"the plan for {n} multiplies back")
    }

    assert_true(radix_plan(7) is null, "seven has no stage")
    assert_true(radix_plan(13) is null)
    assert_true(radix_plan(0) is null)
    assert_true(not is_fast_length(31))
    assert_eq(next_fast_length(31), 32)
    assert_eq(next_fast_length(1001), 1024)
    assert_eq(next_power_of_two(25), 32)
    assert_eq(next_power_of_two(1), 1)
}

fn test_matches_a_hand_written_dft() -> void {
    if not have_gpu() { return }
    for n in [1, 2, 4, 8, 12, 16, 20, 32, 45, 64] {
        let values = signal(n * 2, 7 + n)
        let a = complex_of(values)
        let spectrum = fft(a)
        assert_lists_near(
            spectrum.to_list(),
            reference_dft(values, false),
            1.0e-5,
            f"fft of {n} points"
        )
        let back = ifft(a)
        var scaled: list[float] = []
        for value in reference_dft(values, true) { scaled.push(value / float(n)) }
        assert_lists_near(back.to_list(), scaled, 1.0e-5, f"ifft of {n} points")
        back.free()
        spectrum.free()
        a.free()
    }
}

fn test_bluestein_matches_the_same_dft() -> void {
    if not have_gpu() { return }
    #: Lengths with a prime factor past five, so the stage chain cannot take
    #: them and the chirp-z path is the only thing that can answer.
    for n in [7, 11, 13, 14, 22, 31, 49, 97] {
        assert_true(not is_fast_length(n), f"{n} must reach Bluestein")
        let values = signal(n * 2, 3 + n)
        let a = complex_of(values)
        let spectrum = fft(a)
        #: Three transforms and two products deeper than a direct length, so
        #: it carries a digit less.
        assert_lists_near(
            spectrum.to_list(),
            reference_dft(values, false),
            2.0e-4,
            f"Bluestein fft of {n} points"
        )
        spectrum.free()
        a.free()
    }
}

fn test_a_round_trip_returns_the_signal() -> void {
    if not have_gpu() { return }
    for n in [3, 5, 8, 13, 16, 31, 32, 60, 64, 128] {
        let values = signal(n * 2, 101 + n)
        let a = complex_of(values)
        let spectrum = fft(a)
        let back = ifft(spectrum)
        assert_lists_near(back.to_list(), values, 2.0e-4, f"round trip at {n}")
        back.free()
        spectrum.free()
        a.free()
    }
}

fn test_ortho_is_unitary_both_ways() -> void {
    if not have_gpu() { return }
    for n in [8, 12, 13] {
        let values = signal(n * 2, 55 + n)
        let a = complex_of(values)
        let spectrum = fft(a, null, -1, "ortho")
        let back = ifft(spectrum, null, -1, "ortho")
        assert_lists_near(back.to_list(), values, 2.0e-4, f"ortho round trip at {n}")

        #: Unitary means Parseval holds with no factor at all, which is the
        #: whole reason the convention exists.
        var input_energy = 0.0
        for value in values { input_energy += value * value }
        var output_energy = 0.0
        for value in spectrum.to_list() { output_energy += value * value }
        assert_near(output_energy, input_energy, 1.0e-4 * input_energy, f"ortho energy at {n}")
        back.free()
        spectrum.free()
        a.free()
    }
}

fn test_parseval_holds_without_a_norm() -> void {
    if not have_gpu() { return }
    for n in [8, 15, 17, 64] {
        let values = signal(n * 2, 200 + n)
        let a = complex_of(values)
        let spectrum = fft(a)
        var input_energy = 0.0
        for value in values { input_energy += value * value }
        var output_energy = 0.0
        for value in spectrum.to_list() { output_energy += value * value }
        #: sum |x|^2 == (1/n) sum |X|^2 for the unscaled forward transform.
        assert_near(
            output_energy / float(n),
            input_energy,
            1.0e-4 * input_energy,
            f"Parseval at {n}"
        )
        spectrum.free()
        a.free()
    }
}

#: Lengths past what the reference sum can be asked to check, where the only
#: affordable statements are the two that hold for every length.
#:
#: These are here for the stage chain's depth rather than its width. Below 128
#: a radix-4 plan is three or four stages and `m` never grows past 32; at 4096
#: it is six, and the paired kernel's precondition — `k + 1` staying inside
#: one block — has to hold at every one of them. A chain that lost a twiddle
#: at the last stage would still round-trip near enough to pass at 64.
fn test_a_deep_stage_chain_still_holds_together() -> void {
    if not have_gpu() { return }
    for n in [256, 1024, 4096, 1000, 1200] {
        assert_true(is_fast_length(n), f"{n} is meant to stay on the stage chain")
        let values = signal(n * 2, 900 + n)
        let a = complex_of(values)
        let spectrum = fft(a)
        let back = ifft(spectrum)
        assert_lists_near(back.to_list(), values, 1.0e-4, f"round trip at {n}")

        var input_energy = 0.0
        for value in values { input_energy += value * value }
        var output_energy = 0.0
        for value in spectrum.to_list() { output_energy += value * value }
        assert_near(
            output_energy / float(n),
            input_energy,
            1.0e-4 * input_energy,
            f"Parseval at {n}"
        )
        back.free()
        spectrum.free()
        a.free()
    }
}

#: A batch is `lines` independent transforms, and nothing may leak across the
#: line boundary.
#:
#: Every other test here transforms one line, where `line` is always zero and
#: a stage kernel that dropped the `line * groups * radix` term entirely would
#: still be right. This one gives each row a different signal and checks each
#: against its own reference, so a stage that addressed the wrong line has
#: nowhere to hide.
fn test_every_row_of_a_batch_is_its_own_transform() -> void {
    if not have_gpu() { return }
    for columns in [16, 12, 13] {
        let rows = 5
        let flat = signal(rows * columns * 2, 606 + columns)
        let batch = NDArray([rows, columns], COMPLEX64)
        batch.upload(flat)
        let spectrum = fft(batch)
        assert_eq(spectrum.shape, [rows, columns])
        let got = spectrum.to_list()
        for r in 0..rows {
            var row: list[float] = []
            var mine: list[float] = []
            for j in 0..columns * 2 {
                row.push(flat[r * columns * 2 + j])
                mine.push(got[r * columns * 2 + j])
            }
            assert_lists_near(mine, reference_dft(row, false), 2.0e-4, f"row {r} of {columns}")
        }
        spectrum.free()
        batch.free()
    }
}

#: `forward` is the third convention and the one with no other test: it moves
#: the whole 1/n onto the forward transform, so the inverse carries nothing.
fn test_the_forward_convention_scales_the_other_end() -> void {
    if not have_gpu() { return }
    let n = 16
    let values = signal(n * 2, 414)
    let a = complex_of(values)
    defer { a.free() }
    let spectrum = fft(a, null, -1, "forward")
    defer { spectrum.free() }
    var scaled: list[float] = []
    for value in reference_dft(values, false) { scaled.push(value / float(n)) }
    assert_lists_near(spectrum.to_list(), scaled, 1.0e-5, "forward divides going out")

    let back = ifft(spectrum, null, -1, "forward")
    defer { back.free() }
    assert_lists_near(back.to_list(), values, 2.0e-4, "and multiplies by nothing coming back")
}

fn test_a_spectrum_that_can_be_worked_out_on_paper() -> void {
    if not have_gpu() { return }

    #: A unit impulse transforms to a flat spectrum of ones.
    let impulse = real_of([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    defer { impulse.free() }
    let flat = fft(impulse)
    defer { flat.free() }
    for k in 0..8 {
        assert_near(flat.to_list()[k * 2], 1.0, 1.0e-5, f"impulse bin {k} real")
        assert_near(flat.to_list()[k * 2 + 1], 0.0, 1.0e-5, f"impulse bin {k} imaginary")
    }

    #: The four-point ramp, which is the textbook example: 6, -2+2i, -2, -2-2i.
    let ramp = real_of([0.0, 1.0, 2.0, 3.0])
    defer { ramp.free() }
    let spectrum = fft(ramp)
    defer { spectrum.free() }
    assert_lists_near(
        spectrum.to_list(),
        [6.0, 0.0, -2.0, 2.0, -2.0, 0.0, -2.0, -2.0],
        1.0e-5,
        "the four-point ramp"
    )

    #: A cosine at bin 2 of an eight-point window puts n/2 at bins 2 and 6 and
    #: nothing anywhere else.
    var tone: list[float] = []
    for j in 0..8 { tone.push(math.cos(2.0 * math.PI * 2.0 * float(j) / 8.0)) }
    let wave = real_of(tone)
    defer { wave.free() }
    let lines = fft(wave)
    defer { lines.free() }
    let bins = lines.to_list()
    for k in 0..8 {
        let expected = if k == 2 or k == 6 { 4.0 } else { 0.0 }
        assert_near(bins[k * 2], expected, 1.0e-4, f"tone bin {k} real")
        assert_near(bins[k * 2 + 1], 0.0, 1.0e-4, f"tone bin {k} imaginary")
    }
}

fn test_padding_and_truncation_change_the_length() -> void {
    if not have_gpu() { return }
    let values = [1.0, 2.0, 3.0, 4.0, 5.0]
    let a = real_of(values)
    defer { a.free() }

    #: A longer `n` is the signal followed by zeros, which is a different
    #: transform and not an interpolation of the short one.
    let padded = fft(a, 8)
    defer { padded.free() }
    assert_eq(padded.shape, [8])
    var extended = as_complex(values)
    for _index in 0..3 {
        extended.push(0.0)
        extended.push(0.0)
    }
    assert_lists_near(padded.to_list(), reference_dft(extended, false), 1.0e-5, "zero padded")

    let cut = fft(a, 3)
    defer { cut.free() }
    assert_eq(cut.shape, [3])
    assert_lists_near(
        cut.to_list(),
        reference_dft(as_complex([1.0, 2.0, 3.0]), false),
        1.0e-5,
        "truncated"
    )
}

fn test_an_axis_that_is_not_the_last_one() -> void {
    if not have_gpu() { return }
    let rows = 3
    let columns = 8
    let values = signal(rows * columns, 77)
    let a = NDArray([rows, columns], FLOAT32)
    defer { a.free() }
    a.upload(values)

    let down = fft(a, null, 0)
    defer { down.free() }
    assert_eq(down.shape, [rows, columns])
    let got = down.to_list()
    for column in 0..columns {
        var line: list[float] = []
        for row in 0..rows {
            line.push(values[row * columns + column])
            line.push(0.0)
        }
        let expected = reference_dft(line, false)
        for row in 0..rows {
            let at = (row * columns + column) * 2
            assert_near(got[at], expected[row * 2], 1.0e-4, f"column {column} row {row} real")
            assert_near(
                got[at + 1],
                expected[row * 2 + 1],
                1.0e-4,
                f"column {column} row {row} imaginary"
            )
        }
    }
}

fn test_a_strided_view_transforms_like_its_copy() -> void {
    if not have_gpu() { return }
    let values = signal(24, 909)
    let a = NDArray([3, 8], FLOAT32)
    defer { a.free() }
    a.upload(values)

    #: A transpose is a stride rewrite, so this feeds the packing copy an
    #: input whose fast axis is the one it has to gather across.
    let flipped = transpose(a)
    let spectrum = fft(flipped)
    defer { spectrum.free() }
    assert_eq(spectrum.shape, [8, 3])
    let got = spectrum.to_list()
    for column in 0..8 {
        var line: list[float] = []
        for row in 0..3 {
            line.push(values[row * 8 + column])
            line.push(0.0)
        }
        let expected = reference_dft(line, false)
        for k in 0..3 {
            let at = (column * 3 + k) * 2
            assert_near(got[at], expected[k * 2], 1.0e-4, f"view {column} bin {k} real")
            assert_near(got[at + 1], expected[k * 2 + 1], 1.0e-4, f"view {column} bin {k} imag")
        }
    }
}

fn test_two_dimensions_are_the_two_one_dimensional_passes() -> void {
    if not have_gpu() { return }
    let rows = 4
    let columns = 6
    let a = NDArray([rows, columns], FLOAT32)
    defer { a.free() }
    a.upload(signal(rows * columns, 4242))

    let planar = fft2(a)
    defer { planar.free() }
    let stage = fft(a, null, 1)
    defer { stage.free() }
    let stacked = fft(stage, null, 0)
    defer { stacked.free() }
    assert_eq(planar.shape, [rows, columns])
    assert_lists_near(planar.to_list(), stacked.to_list(), 1.0e-5, "fft2 against two passes")

    let back = ifft2(planar)
    defer { back.free() }
    assert_lists_near(back.to_list(), a.astype(COMPLEX64).to_list(), 2.0e-4, "ifft2 round trip")

    #: `fftn` over both axes is the same thing under another name, and over a
    #: single axis it is `fft`.
    let every = fftn(a)
    defer { every.free() }
    assert_lists_near(every.to_list(), planar.to_list(), 1.0e-5, "fftn over every axis")
    let one = fftn(a, null, [1])
    defer { one.free() }
    assert_lists_near(one.to_list(), stage.to_list(), 1.0e-5, "fftn over one axis")
    let undone = ifftn(every)
    defer { undone.free() }
    assert_lists_near(undone.to_list(), a.astype(COMPLEX64).to_list(), 2.0e-4, "ifftn round trip")
}

fn test_rfft_is_the_half_of_the_full_transform() -> void {
    if not have_gpu() { return }
    for n in [4, 5, 8, 9, 13, 16] {
        let values = signal(n, 31 + n)
        let a = real_of(values)
        let half = rfft(a)
        assert_eq(half.shape, [n // 2 + 1], f"rfft keeps n/2+1 bins at {n}")

        let expected = reference_dft(as_complex(values), false)
        var kept: list[float] = []
        for k in 0..n // 2 + 1 {
            kept.push(expected[k * 2])
            kept.push(expected[k * 2 + 1])
        }
        assert_lists_near(half.to_list(), kept, 1.0e-5, f"rfft at {n}")

        #: And the bins it dropped really were the mirror of the ones it kept.
        let widened = full_spectrum(half, n)
        assert_lists_near(widened.to_list(), expected, 1.0e-5, f"the mirrored half at {n}")
        widened.free()
        half.free()
        a.free()
    }
}

fn test_irfft_inverts_rfft_at_both_parities() -> void {
    if not have_gpu() { return }
    for n in [4, 6, 7, 8, 15, 16, 32] {
        let values = signal(n, 501 + n)
        let a = real_of(values)
        let half = rfft(a)
        #: An odd length has to be handed back: `n // 2 + 1` bins came from
        #: both `2 * bins - 1` and `2 * bins - 2` samples.
        let back = irfft(half, n)
        assert_eq(back.shape, [n])
        assert_lists_near(back.to_list(), values, 2.0e-4, f"irfft round trip at {n}")
        back.free()
        half.free()
        a.free()
    }

    #: Without `n` the even reading is the one taken.
    let a = real_of(signal(8, 12))
    defer { a.free() }
    let half = rfft(a)
    defer { half.free() }
    let back = irfft(half)
    defer { back.free() }
    assert_eq(back.shape, [8], "the default output length is 2 * (bins - 1)")
}

fn test_the_hermitian_pair_inverts_itself() -> void {
    if not have_gpu() { return }
    for n in [6, 8, 12] {
        let values = signal(n, 800 + n)
        let a = real_of(values)
        let half = ihfft(a)
        assert_eq(half.shape, [n // 2 + 1])
        let back = hfft(half, n)
        assert_lists_near(back.to_list(), values, 2.0e-4, f"hfft of ihfft at {n}")
        back.free()
        half.free()
        a.free()
    }

    #: `hfft` is the forward transform of a Hermitian signal, so its answer is
    #: the real transform of the full spectrum the half stands for.
    let half = complex_of([1.0, 0.0, 2.0, -1.0, 0.5, 0.0])
    defer { half.free() }
    let spectrum = hfft(half, 4)
    defer { spectrum.free() }
    let mirrored = conj(half)
    defer { mirrored.free() }
    let widened = full_spectrum(mirrored, 4)
    defer { widened.free() }
    let reference = reference_dft(widened.to_list(), true)
    var expected: list[float] = []
    for k in 0..4 { expected.push(reference[k * 2]) }
    assert_lists_near(spectrum.to_list(), expected, 1.0e-4, "hfft against its definition")
}

fn test_rfft2_and_its_inverse() -> void {
    if not have_gpu() { return }
    let rows = 4
    let columns = 8
    let a = NDArray([rows, columns], FLOAT32)
    defer { a.free() }
    let values = signal(rows * columns, 3131)
    a.upload(values)

    let half = rfft2(a)
    defer { half.free() }
    assert_eq(half.shape, [rows, columns // 2 + 1])

    #: The same thing spelled as two ordinary passes and a slice.
    let planar = fft2(a)
    defer { planar.free() }
    let kept = half_spectrum(planar, columns, 1).copy()
    defer { kept.free() }
    assert_lists_near(half.to_list(), kept.to_list(), 1.0e-5, "rfft2 against fft2")

    let back = irfft2(half, [rows, columns])
    defer { back.free() }
    assert_eq(back.shape, [rows, columns])
    assert_lists_near(back.to_list(), values, 2.0e-4, "irfft2 round trip")
}

fn test_conj_flips_only_the_imaginary_lane() -> void {
    if not have_gpu() { return }
    let a = complex_of([1.0, 2.0, -3.0, 4.0, 0.0, -5.0])
    defer { a.free() }
    let flipped = conj(a)
    defer { flipped.free() }
    assert_eq(flipped.to_list(), [1.0, -2.0, -3.0, -4.0, 0.0, 5.0])
    assert_eq(flipped.dtype, COMPLEX64)

    #: Twice is the identity, and the strided path has to agree with the dense
    #: one — a reversed view is the cheapest way to reach it.
    let twice = conjugate(flipped)
    defer { twice.free() }
    assert_eq(twice.to_list(), a.to_list())

    let reversed = a[sl(null, null, -1)]
    let strided = conj(reversed)
    defer { strided.free() }
    assert_eq(strided.to_list(), [0.0, 5.0, -3.0, -4.0, 1.0, -2.0], "the strided conjugate")

    #: A real array conjugates to itself, and to a copy rather than the
    #: argument, so freeing the answer does not free the input.
    let plain = real_of([1.0, 2.0])
    defer { plain.free() }
    let same = conj(plain)
    same.free()
    assert_eq(plain.to_list(), [1.0, 2.0], "the input survived freeing the answer")
}

fn test_the_frequency_ladders_match_their_definition() -> void {
    if not have_gpu() { return }
    for n in [1, 2, 5, 8, 9] {
        let ladder = fftfreq(n, 0.5)
        assert_eq(ladder.shape, [n])
        let got = ladder.to_list()
        for k in 0..n {
            let index = if k < (n + 1) // 2 { float(k) } else { float(k) - float(n) }
            assert_near(got[k], index / (float(n) * 0.5), 1.0e-6, f"fftfreq({n}) at {k}")
        }

        let half = rfftfreq(n, 0.5)
        assert_eq(half.shape, [n // 2 + 1])
        let bins = half.to_list()
        for k in 0..n // 2 + 1 {
            #: Divided by the signal's n, not by the number of bins — the bin
            #: spacing does not change when the mirror is dropped.
            assert_near(bins[k], float(k) / (float(n) * 0.5), 1.0e-6, f"rfftfreq({n}) at {k}")
        }
        half.free()
        ladder.free()
    }
    assert_throws(ValueError, || fftfreq(0))
}

fn test_fftshift_puts_zero_in_the_middle() -> void {
    if not have_gpu() { return }
    #: The point of the shift is that the frequency ladder comes out sorted.
    for n in [4, 5, 8, 9] {
        let ladder = fftfreq(n)
        let sorted = fftshift(ladder)
        let got = sorted.to_list()
        for k in 1..n {
            assert_true(got[k] > got[k - 1], f"fftshift({n}) is ascending at {k}")
        }
        #: And zero lands at n // 2, which is what "centred" means.
        assert_near(got[n // 2], 0.0, 1.0e-7, f"zero is centred at {n}")

        let undone = ifftshift(sorted)
        assert_lists_near(undone.to_list(), ladder.to_list(), 1.0e-6, f"ifftshift undoes at {n}")
        undone.free()
        sorted.free()
        ladder.free()
    }

    #: Two axes at once, and one axis named on its own.
    let a = NDArray([2, 4], FLOAT32)
    defer { a.free() }
    a.upload([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    let both = fftshift(a)
    defer { both.free() }
    assert_eq(both.to_list(), [6.0, 7.0, 4.0, 5.0, 2.0, 3.0, 0.0, 1.0])
    let columns = fftshift(a, 1)
    defer { columns.free() }
    assert_eq(columns.to_list(), [2.0, 3.0, 0.0, 1.0, 6.0, 7.0, 4.0, 5.0])
}

fn test_half_spectrum_is_a_view_and_full_spectrum_is_not() -> void {
    if not have_gpu() { return }
    let a = complex_of([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0])
    defer { a.free() }
    let half = half_spectrum(a, 4)
    assert_eq(half.shape, [3], "four bins keep three")

    #: A view, so a write through the parent is visible in it. That is the
    #: reason it is not a kernel.
    a[0] = 9.0
    assert_near(half.to_list()[0], 9.0, 1.0e-6, "the half spectrum aliases its parent")

    #: The other direction mirrors and conjugates, which no stride can do.
    let widened = full_spectrum(half, 4)
    defer { widened.free() }
    assert_eq(widened.shape, [4])
    let got = widened.to_list()
    assert_near(got[6], got[2], 1.0e-6, "bin 3 mirrors bin 1")
    assert_near(got[7], -got[3], 1.0e-6, "and conjugates it")

    #: A short input is padded out to the bins the length asks for.
    let short = complex_of([1.0, 0.0])
    defer { short.free() }
    let padded = full_spectrum(short, 6)
    defer { padded.free() }
    assert_eq(padded.to_list(), [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
}

fn test_the_arguments_that_have_no_reading() -> void {
    if not have_gpu() { return }
    let a = real_of([1.0, 2.0, 3.0, 4.0])
    defer { a.free() }
    assert_throws(ValueError, || fft(a, 0), "a transform of no points")
    assert_throws(ValueError, || fft(a, 4, 0, "sideways"), "an unknown norm")
    assert_throws(ValueError, || fft(a, 4, 3), "an axis the array does not have")
    assert_throws(ValueError, || fft2(a), "a plane needs two axes")

    let complex = complex_of([1.0, 1.0, 2.0, 2.0])
    defer { complex.free() }
    assert_throws(TypeError, || rfft(complex), "rfft takes a real signal")
    assert_throws(TypeError, || ihfft(complex), "ihfft takes a real signal")
}
