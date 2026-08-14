#: Tests for `jaiplot`: generators, ticks, scales, geometry and both backends.
#:
#: The raster tests render onto an offscreen `Image`, so they need no display
#: and run anywhere the rest of the suite does.
import std.math
import jaiplot
from std.test import assert_eq, assert_near, assert_true, assert_false, assert_throws, assert_ne
from std.gui.canvas import Canvas
from std.gui.color import red_of, rgb
from std.gui.image import Image
from std.gui.path import Path, Winding

from jaiplot import Figure, MarkerShape, Theme
from jaiplot.backend.png import encode
from jaiplot.backend.raster import render_image
from jaiplot.backend.svg import to_svg
from jaiplot.data import arange, extent, linspace, logspace, minmax, percentile
from jaiplot.frame import Frame, runs
from jaiplot.geom import Point, Rect, Transform
from jaiplot.metrics import text_height, text_width
from jaiplot.primitive import DisplayList, Primitive, describe
from jaiplot.scale import LinearScale, LogScale, scale_named
from jaiplot.series import Bar, FillBetween, Grid, Hist, Line, PoseCloud, Rays, Scatter, Step
from jaiplot.style import Color
from jaiplot.ticks import decimals_for_step, format_tick, linear_ticks, log_ticks, nice_bounds
from jaiplot.ticks import log_minor_ticks, tick_step

#: Assert `body` raises `kind`, as a statement.
#:
#: `assert_throws` returns the error it caught, and a test that calls it several
#: times would otherwise need a fresh throwaway binding for each one.
fn expect_error(kind: any, body: fn() -> any) -> void {
    let _caught = assert_throws(kind, body)
}

#: Assert two float series agree elementwise.
#:
#: `std.test.assert_near` takes two scalars only, so comparing a series is not
#: expressible without this.
fn assert_series_near(actual: list[float], expected: list[float], tolerance: float = 1e-9) -> void {
    assert_eq(actual.len(), expected.len())
    for (index, value) in actual.enumerate() {
        assert_near(value, expected[index], tolerance)
    }
}

pub fn test_linspace_hits_both_endpoints_exactly() -> void {
    let values = linspace(0.0, 1.0, 5)
    assert_eq(values.len(), 5)
    assert_true(values[0] == 0.0)
    assert_true(values[4] == 1.0)
    assert_series_near(values, [0.0, 0.25, 0.5, 0.75, 1.0])
}

pub fn test_linspace_rejects_a_count_below_two() -> void {
    expect_error(ValueError, || linspace(0.0, 1.0, 1))
}

pub fn test_arange_stays_below_the_stop() -> void {
    let values = arange(0.0, 1.0, 0.3)
    assert_series_near(values, [0.0, 0.3, 0.6, 0.9])
    for value in values { assert_true(value < 1.0) }
}

pub fn test_arange_rejects_a_zero_step() -> void {
    expect_error(ValueError, || arange(0.0, 1.0, 0.0))
}

pub fn test_logspace_spans_the_decades() -> void {
    assert_series_near(logspace(0.0, 3.0, 4), [1.0, 10.0, 100.0, 1000.0], 1e-9)
}

pub fn test_minmax_skips_nan() -> void {
    let found = minmax([1.0, math.NAN, -2.0, 5.0])
    assert_true(found != null)
    let (low, high) = found ?? (0.0, 0.0)
    assert_near(low, -2.0)
    assert_near(high, 5.0)
}

pub fn test_minmax_of_an_empty_list_is_null() -> void {
    assert_true(minmax([]) == null)
}

pub fn test_extent_of_a_constant_series_is_not_degenerate() -> void {
    # A zero-width extent would divide by zero in the transform, so a constant
    # series has to widen to something drawable.
    let found = extent([4.0, 4.0, 4.0])
    assert_true(found != null)
    let (low, high) = found ?? (0.0, 0.0)
    assert_true(high > low)
}

pub fn test_percentile_matches_a_hand_computed_value() -> void {
    assert_near(percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50.0), 3.0)
    assert_near(percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.0), 1.0)
    assert_near(percentile([1.0, 2.0, 3.0, 4.0, 5.0], 100.0), 5.0)
}

pub fn test_percentile_rejects_a_value_outside_the_range() -> void {
    expect_error(ValueError, || percentile([1.0, 2.0], 101.0))
}

pub fn test_ticks_are_round_and_inside_the_limits() -> void {
    let values = linear_ticks(0.0, 100.0, 6)
    for value in values {
        assert_true(value >= 0.0)
        assert_true(value <= 100.0)
    }
    assert_series_near(values, [0.0, 20.0, 40.0, 60.0, 80.0, 100.0])
}

pub fn test_ticks_ascend() -> void {
    let values = linear_ticks(-3.7, 12.4, 6)
    for index in 1..values.len() {
        assert_true(values[index] > values[index - 1])
    }
}

pub fn test_tick_step_is_one_two_or_five_times_a_power_of_ten() -> void {
    let cases = [(0.0, 1.0), (0.0, 100.0), (-5.0, 5.0), (0.0, 0.003), (0.0, 7000000.0)]
    for (low, high) in cases {
        let step = tick_step(low, high, 6)
        let exponent = math.floor(math.log10(step))
        let mantissa = step / 10.0 ** exponent
        let allowed = math.fabs(mantissa - 1.0) < 1e-9 or
            math.fabs(mantissa - 2.0) < 1e-9 or
            math.fabs(mantissa - 5.0) < 1e-9
        assert_true(allowed)
    }
}

pub fn test_a_constant_interval_yields_one_tick() -> void {
    assert_series_near(linear_ticks(3.0, 3.0, 6), [3.0])
}

pub fn test_ticks_reject_reversed_limits() -> void {
    expect_error(ValueError, || linear_ticks(5.0, 1.0, 6))
}

pub fn test_nice_bounds_enclose_the_data() -> void {
    let (low, high) = nice_bounds(2.3, 97.1, 6)
    assert_true(low <= 2.3)
    assert_true(high >= 97.1)
}

pub fn test_log_ticks_are_powers_of_the_base() -> void {
    assert_series_near(log_ticks(1.0, 1000.0, 10.0, 4), [1.0, 10.0, 100.0, 1000.0])
}

pub fn test_log_minor_ticks_vanish_over_many_decades() -> void {
    assert_true(log_minor_ticks(1.0, 10.0, 10.0).len() > 0)
    assert_eq(log_minor_ticks(1.0, 1.0e12, 10.0).len(), 0)
}

pub fn test_log_ticks_reject_a_non_positive_limit() -> void {
    expect_error(ValueError, || log_ticks(0.0, 10.0, 10.0, 4))
}

pub fn test_labels_share_one_precision_across_an_axis() -> void {
    assert_eq(decimals_for_step(20.0), 0)
    assert_eq(decimals_for_step(0.05), 2)
    assert_eq(format_tick(0.5, 0.05), "0.50")
    assert_eq(format_tick(50.0, 20.0), "50")
}

pub fn test_a_negative_zero_label_loses_its_sign() -> void {
    assert_eq(format_tick(-0.0, 0.5), "0.0")
}

pub fn test_transform_round_trips() -> void {
    let transform = Transform(Rect(0.0, 0.0, 10.0, 4.0), Rect(0.0, 0.0, 100.0, 40.0))
    for index in 0..20 {
        let x = index.to_float() * 0.5
        let y = index.to_float() * 0.2
        let placed = transform.apply(x, y)
        let (back_x, back_y) = transform.invert(placed.x, placed.y)
        assert_near(back_x, x, 1e-9)
        assert_near(back_y, y, 1e-9)
    }
}

pub fn test_transform_flips_the_vertical_axis() -> void {
    let transform = Transform(Rect(0.0, 0.0, 1.0, 1.0), Rect(0.0, 0.0, 10.0, 10.0))
    assert_near(transform.apply(0.0, 0.0).y, 10.0)
    assert_near(transform.apply(0.0, 1.0).y, 0.0)
}

pub fn test_transform_rejects_a_zero_extent() -> void {
    expect_error(ValueError, || Transform(Rect(1.0, 0.0, 1.0, 4.0), Rect(0.0, 0.0, 8.0, 8.0)))
}

pub fn test_rect_normalises_reversed_corners() -> void {
    let rect = Rect(4.0, 9.0, 1.0, 3.0)
    assert_near(rect.left(), 1.0)
    assert_near(rect.bottom(), 3.0)
    assert_near(rect.right(), 4.0)
    assert_near(rect.top(), 9.0)
}

pub fn test_rect_intersection_can_be_absent() -> void {
    assert_true(Rect(0.0, 0.0, 1.0, 1.0).intersect(Rect(5.0, 5.0, 6.0, 6.0)) == null)
}

pub fn test_log_scale_composes_into_a_transform() -> void {
    let scale = LogScale()
    let frame = Frame(
        Rect(1.0, 0.0, 1000.0, 1.0),
        Rect(0.0, 0.0, 100.0, 100.0),
        scale,
        LinearScale()
    )
    let placed = frame.place(10.0, 0.0)
    assert_true(placed != null)
    assert_near((placed ?? Point(0.0, 0.0)).x, 100.0 / 3.0, 1e-9)
}

pub fn test_a_log_scale_refuses_a_non_positive_value() -> void {
    let scale = LogScale()
    assert_false(scale.accepts(0.0))
    assert_false(scale.accepts(-1.0))
    assert_true(scale.accepts(0.001))
    expect_error(ValueError, || scale.forward(0.0))
}

pub fn test_scale_named_rejects_a_misspelling() -> void {
    expect_error(ValueError, || scale_named("linnear"))
}

pub fn test_a_gap_splits_a_series_into_runs() -> void {
    let box = Rect(0.0, 0.0, 10.0, 10.0)
    let frame = Frame(box, box, LinearScale(), LinearScale())
    let pieces = runs(frame, [0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 1.0, math.NAN, 3.0, 4.0])
    assert_eq(pieces.len(), 2)
    assert_eq(pieces[0].len(), 2)
    assert_eq(pieces[1].len(), 2)
}

pub fn test_a_display_list_renders_deterministically() -> void {
    let list = DisplayList()
    list.push(Primitive.PushClip(Rect(0.0, 0.0, 4.0, 2.0)))
    list.push(Primitive.PopClip)
    assert_eq(list.render(), "PushClip 0.000,0.000 4.000x2.000\nPopClip")
    assert_eq(describe(Primitive.PopClip), "PopClip")
}

pub fn test_an_unbalanced_clip_is_rejected() -> void {
    let stray = DisplayList()
    stray.push(Primitive.PopClip)
    expect_error(ValueError, || stray.validate())

    let unclosed = DisplayList()
    unclosed.push(Primitive.PushClip(Rect(0.0, 0.0, 1.0, 1.0)))
    expect_error(ValueError, || unclosed.validate())
}

pub fn test_colour_round_trips_through_hex() -> void {
    for text in ["#1f77b4", "#000000", "#ffffff", "#ff8800"] {
        assert_eq(Color.from_hex(text).to_hex(), text)
    }
}

pub fn test_colour_mixing_is_gamma_correct() -> void {
    let middle = Color.from_rgb(0, 0, 0).lerp(Color.from_rgb(255, 255, 255), 0.5)
    assert_true(middle.red() > 180)
    assert_true(middle.red() < 195)
}

pub fn test_from_hex_rejects_a_malformed_string() -> void {
    expect_error(ValueError, || Color.from_hex("not a colour"))
}

pub fn test_text_metrics_match_the_canvas_font() -> void {
    assert_near(text_width("abc", 1), 17.0)
    assert_near(text_width("abc", 2), 34.0)
    assert_near(text_height("abc", 1), 7.0)
}

pub fn test_a_series_reports_the_box_it_needs() -> void {
    let line = Line([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
    let box = line.extent()
    assert_true(box != null)
    assert_near((box ?? Rect(0.0, 0.0, 0.0, 0.0)).top(), 4.0)
}

pub fn test_a_series_of_mismatched_lengths_is_rejected() -> void {
    expect_error(ValueError, || Line([0.0, 1.0], [0.0]))
    expect_error(ValueError, || Scatter([0.0, 1.0], [0.0]))
    expect_error(ValueError, || Bar([0.0, 1.0], [0.0]))
    expect_error(ValueError, || FillBetween([0.0], [0.0], [0.0, 1.0]))
}

pub fn test_a_bar_chart_includes_its_baseline() -> void {
    let bars = Bar([0.0, 1.0, 2.0], [3.0, 7.0, 5.0])
    let box = bars.extent() ?? Rect(0.0, 0.0, 0.0, 0.0)
    assert_near(box.bottom(), 0.0)
    assert_near(box.top(), 7.0)
}

pub fn test_a_histogram_needs_a_finite_sample() -> void {
    expect_error(ValueError, || Hist([]))
    expect_error(ValueError, || Hist([math.NAN, math.NAN]))
}

pub fn test_a_histogram_counts_every_sample() -> void {
    let counts = Hist([1.0, 1.5, 2.0, 2.5, 3.0], bins: 2)
    assert_eq(counts.bins(), 2)
    var total = 0.0
    for height in counts.heights() { total += height }
    assert_near(total, 5.0)
}

pub fn test_a_step_series_holds_its_value_between_samples() -> void {
    let step = Step([0.0, 1.0], [1.0, 3.0])
    let box = step.extent() ?? Rect(0.0, 0.0, 0.0, 0.0)
    assert_near(box.top(), 3.0)
}

pub fn test_an_explicit_colour_survives_the_cycle() -> void {
    let figure = Figure(width: 200, height: 150)
    let axes = figure.axes()
    let red = Color.from_hex("#ff0000")
    let line = axes.line([0.0, 1.0], [0.0, 1.0], color: red)
    assert_eq(line.extent() == null, false)
    assert_true(figure.render().len() > 0)
}

pub fn test_a_figure_rejects_a_non_positive_size() -> void {
    expect_error(ValueError, || Figure(width: 0, height: 10))
    expect_error(ValueError, || Figure(width: 10, height: -1))
}

pub fn test_a_subplot_grid_that_cannot_fit_is_rejected() -> void {
    let figure = Figure(width: 40, height: 40)
    expect_error(ValueError, || figure.subplots(rows: 50, cols: 50))
}

pub fn test_subplots_are_laid_out_without_overlapping() -> void {
    let figure = Figure(width: 400, height: 300)
    let grid = figure.subplots(rows: 2, cols: 2)
    assert_eq(grid.len(), 2)
    assert_eq(grid[0].len(), 2)
    for row in grid {
        for panel in row { panel.line([0.0, 1.0], [0.0, 1.0]) }
    }
    assert_true(figure.render().len() > 0)
}

pub fn test_axes_reject_reversed_limits() -> void {
    let axes = Figure(width: 100, height: 100).axes()
    expect_error(ValueError, || axes.set_xlim(5.0, 1.0))
    expect_error(ValueError, || axes.set_ylim(5.0, 1.0))
}

pub fn test_the_plot_area_matches_what_render_lays_out() -> void {
    let figure = Figure(width: 400, height: 300)
    let axes = figure.axes()
    axes.line([0.0, 1.0], [0.0, 1.0])
    let plot = axes.plot_area(figure.device())
    let document = to_svg(figure.render(), 400, 300)
    let corner = f"{int(plot.left())},{int(plot.bottom())}"
    assert_true(document.contains(corner))
}

pub fn test_svg_is_well_formed_and_escapes_its_text() -> void {
    let figure = Figure(width: 200, height: 150)
    let axes = figure.axes()
    axes.line([0.0, 1.0], [0.0, 1.0])
    axes.set_title("a < b & c > d")
    let document = to_svg(figure.render(), 200, 150)
    assert_true(document.starts_with("<svg"))
    assert_true(document.ends_with("</svg>\n"))
    assert_true(document.contains("&lt;"))
    assert_true(document.contains("&amp;"))
    assert_true(document.contains("&gt;"))
    assert_false(document.contains("a < b"))
}

pub fn test_svg_rejects_a_non_positive_size() -> void {
    expect_error(ValueError, || to_svg(DisplayList(), 0, 10))
}

pub fn test_the_raster_backend_draws_without_a_window() -> void {
    let figure = Figure(width: 160, height: 120)
    let axes = figure.axes()
    axes.line([0.0, 1.0, 2.0], [0.0, 1.0, 0.5])
    axes.grid(true)
    let image = render_image(figure.render(), 160, 120)
    assert_eq(image.width(), 160)
    assert_eq(image.height(), 120)

    var ink = 0
    for pixel in image.pixels() {
        if red_of(pixel) < 250 { ink += 1 }
    }
    assert_true(ink > 100)
}

pub fn test_filling_a_path_anti_aliases_its_edge() -> void {
    let image = Image(32, 32, 0xFF000000)
    let canvas = Canvas(image)
    let triangle = Path()
    triangle.polygon([(2.0, 2.0), (30.0, 2.0), (2.0, 30.0)])
    canvas.fill_path(triangle, rgb(255, 255, 255), Winding.NonZero)

    var partial = 0
    for row in 0..32 {
        for column in 0..32 {
            let value = red_of(canvas.get_pixel(column, row))
            if value > 0 and value < 255 { partial += 1 }
        }
    }
    assert_true(partial > 0)
}

pub fn test_a_clip_stack_only_ever_narrows() -> void {
    let canvas = Canvas(Image(20, 20))
    canvas.push_clip(0, 0, 10, 10)
    canvas.push_clip(5, 5, 100, 100)
    assert_eq(canvas.clip(), (5, 5, 5, 5))
    canvas.pop_clip()
    assert_eq(canvas.clip(), (0, 0, 10, 10))
    canvas.pop_clip()
}

pub fn test_popping_an_empty_clip_stack_is_rejected() -> void {
    let canvas = Canvas(Image(8, 8))
    expect_error(IndexError, || canvas.pop_clip())
}

pub fn test_a_translucent_stroke_does_not_darken_at_its_joins() -> void {
    let canvas = Canvas(Image(64, 64, 0xFF000000))
    let zigzag = Path()
    zigzag.move_to(4.0, 32.0)
    zigzag.line_to(20.0, 10.0)
    zigzag.line_to(36.0, 54.0)
    canvas.stroke_path(zigzag, 0x80FFFFFF, width: 6.0)
    assert_eq(red_of(canvas.get_pixel(20, 12)), red_of(canvas.get_pixel(12, 21)))
}

pub fn test_png_carries_the_signature_and_the_expected_chunks() -> void {
    let data = encode(Image(8, 8, rgb(10, 20, 30))).to_list()
    let signature = [137, 80, 78, 71, 13, 10, 26, 10]
    for (index, byte) in signature.enumerate() {
        assert_eq(data[index], byte)
    }
    assert_eq(data[12], ord("I"))
    assert_eq(data[13], ord("H"))
    assert_eq(data[14], ord("D"))
    assert_eq(data[15], ord("R"))
    assert_true(data.len() > 60)
}

pub fn test_saving_rejects_an_unknown_extension() -> void {
    let figure = Figure(width: 60, height: 40)
    figure.axes().line([0.0, 1.0], [0.0, 1.0])
    expect_error(ValueError, || jaiplot.save(figure, "chart.gif"))
    expect_error(ValueError, || jaiplot.save(figure, "chart"))
}

pub fn test_a_theme_supplies_a_full_series_palette() -> void {
    assert_true(Theme.light().cycle.len() >= 8)
    assert_true(Theme.dark().cycle.len() >= 8)
    assert_eq(Theme.light().background.to_hex(), "#ffffff")
}

pub fn test_a_marker_shape_renders_in_both_backends() -> void {
    let figure = Figure(width: 120, height: 100)
    let axes = figure.axes()
    axes.scatter([0.0, 1.0], [0.0, 1.0], shape: MarkerShape.Star)
    let list = figure.render()
    assert_true(to_svg(list, 120, 100).contains("polygon"))
    let image = render_image(list, 120, 100)
    assert_eq(image.width(), 120)
}

pub fn test_a_series_can_be_re_fed_without_losing_its_styling() -> void {
    let figure = Figure(width: 200, height: 150)
    let axes = figure.axes()
    let line = axes.line([0.0, 1.0], [0.0, 1.0], label: "moving")
    line.set_data([0.0, 1.0, 2.0], [5.0, 6.0, 7.0])
    let box = line.extent() ?? Rect(0.0, 0.0, 0.0, 0.0)
    assert_near(box.top(), 7.0)
    assert_near(box.right(), 2.0)
    assert_eq(line.label() ?? "", "moving")
}

pub fn test_re_feeding_rejects_mismatched_lengths() -> void {
    let line = Line([0.0, 1.0], [0.0, 1.0])
    expect_error(ValueError, || line.set_data([0.0, 1.0], [0.0]))

    let points = Scatter([0.0, 1.0], [0.0, 1.0])
    expect_error(ValueError, || points.set_data([0.0], [0.0, 1.0]))

    let bars = Bar([0.0, 1.0], [1.0, 2.0])
    expect_error(ValueError, || bars.set_values([1.0]))

    let band = FillBetween([0.0, 1.0], [0.0, 0.0], [1.0, 1.0])
    expect_error(ValueError, || band.set_band([0.0], [0.0], [0.0, 1.0]))
}

pub fn test_successive_frames_differ() -> void {
    let figure = Figure(width: 120, height: 90)
    let axes = figure.axes()
    let _limits = axes.set_ylim(-1.5, 1.5)
    let line = axes.line([0.0, 1.0, 2.0], [0.0, 1.0, 0.0])
    let first = figure.render().render()
    line.set_data([0.0, 1.0, 2.0], [0.0, -1.0, 0.0])
    let second = figure.render().render()
    assert_ne(first, second)
}

pub fn test_a_held_axis_does_not_re_range_between_frames() -> void {
    let figure = Figure(width: 120, height: 90)
    let axes = figure.axes()
    let _limits = axes.set_ylim(-2.0, 2.0)
    let line = axes.line([0.0, 1.0], [0.0, 0.5])
    let (_x_low, _x_high, low, high) = axes.limits()
    line.set_data([0.0, 1.0], [0.0, 1.9])
    let (_next_x_low, _next_x_high, next_low, next_high) = axes.limits()
    assert_near(low, next_low)
    assert_near(high, next_high)
}

pub fn test_a_pose_cloud_batches_its_particles() -> void {
    let figure = Figure(width: 200, height: 160)
    let axes = figure.axes()
    var xs = []
    var ys = []
    var headings = []
    for index in 0..500 {
        xs.push(index.to_float() % 20.0)
        ys.push(index.to_float() % 13.0)
        headings.push(index.to_float() * 0.01)
    }
    let cloud = axes.poses(xs, ys, headings: headings)
    assert_eq(cloud.len(), 500)
    let list = figure.render()
    var batches = 0
    for primitive in list.to_list() {
        match primitive {
            Primitive.Poses(_points, _angles, _size, _fill) => { batches += 1 }
            _ => { }
        }
    }
    assert_eq(batches, 1)
}

pub fn test_a_weighted_cloud_splits_into_opacity_buckets() -> void {
    let figure = Figure(width: 200, height: 160)
    let axes = figure.axes()
    var xs = []
    var ys = []
    var weights = []
    for index in 0..200 {
        xs.push(index.to_float() % 17.0)
        ys.push(index.to_float() % 11.0)
        weights.push(index.to_float() / 200.0)
    }
    let _cloud = axes.poses(xs, ys, weights: weights)
    var batches = 0
    for primitive in figure.render().to_list() {
        match primitive {
            Primitive.Points(_points, _shape, _size, _fill) => { batches += 1 }
            _ => { }
        }
    }
    assert_true(batches > 1)
}

pub fn test_a_cloud_rejects_mismatched_headings() -> void {
    expect_error(ValueError, || PoseCloud([0.0, 1.0], [0.0, 1.0], headings: [0.0]))
    expect_error(ValueError, || PoseCloud([0.0, 1.0], [0.0, 1.0], weights: [1.0]))
    let cloud = PoseCloud([0.0], [0.0])
    expect_error(ValueError, || cloud.set_cloud([0.0, 1.0], [0.0]))
}

pub fn test_rays_are_one_primitive_of_endpoint_pairs() -> void {
    let figure = Figure(width: 200, height: 160)
    let axes = figure.axes()
    let scan = axes.rays([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, -1.0], [1.0, 2.0, 1.0])
    assert_eq(scan.len(), 3)
    for primitive in figure.render().to_list() {
        match primitive {
            Primitive.Segments(points, _stroke) => { assert_eq(points.len(), 6) }
            _ => { }
        }
    }
}

pub fn test_rays_reject_ragged_input() -> void {
    expect_error(ValueError, || Rays([0.0, 1.0], [0.0], [1.0, 2.0], [1.0, 2.0]))
}

pub fn test_a_grid_rejects_a_cell_count_that_does_not_match() -> void {
    expect_error(ValueError, || Grid([0.0, 1.0], 3, 3, Rect(0.0, 0.0, 1.0, 1.0)))
    let grid = Grid([0.0, 1.0, 1.0, 0.0], 2, 2, Rect(0.0, 0.0, 2.0, 2.0))
    expect_error(ValueError, || grid.set_cells([0.0]))
}

pub fn test_a_grid_becomes_one_raster_the_right_way_up() -> void {
    let figure = Figure(width: 200, height: 160)
    let axes = figure.axes()
    let _grid = axes.grid_of([0.0, 0.0, 1.0, 1.0], 2, 2, Rect(0.0, 0.0, 2.0, 2.0))
    var found = false
    for primitive in figure.render().to_list() {
        match primitive {
            Primitive.Raster(_rect, pixels, width, height) => {
                assert_eq(width, 2)
                assert_eq(height, 2)
                assert_eq(pixels.len(), 4)
                assert_ne(pixels[0], pixels[2])
                found = true
            }
            _ => { }
        }
    }
    assert_true(found)
}

pub fn test_equal_aspect_widens_rather_than_crops() -> void {
    let figure = Figure(width: 400, height: 200)
    let axes = figure.axes()
    let _limits = axes.set_xlim(0.0, 10.0)
    let _more = axes.set_ylim(0.0, 10.0)
    let _square = axes.set_aspect(true)
    let frame = axes.frame(figure.device())
    let x_span = frame.data.width()
    let y_span = frame.data.height()
    let plot = axes.plot_area(figure.device())
    # One data unit must be the same length on both axes.
    assert_near(x_span / y_span, plot.width() / plot.height(), 1e-9)
    # And nothing the caller asked to see may have left the view.
    assert_true(frame.data.left() <= 0.0)
    assert_true(frame.data.right() >= 10.0)
}

pub fn test_freezing_splits_the_layers() -> void {
    let figure = Figure(width: 200, height: 160)
    let axes = figure.axes()
    let _map = axes.grid_of([0.0, 1.0, 1.0, 0.0], 2, 2, Rect(0.0, 0.0, 2.0, 2.0))
    let _frozen = figure.freeze()
    let _cloud = axes.poses([0.5, 1.5], [0.5, 1.5])

    assert_true(figure.is_frozen())
    let (background, foreground) = figure.render_layers()
    assert_true(background.len() > foreground.len())
    assert_eq(foreground.render(), figure.render_dynamic().render())
    assert_true(figure.render_dynamic().len() > 0)
}

pub fn test_an_unfrozen_figure_reports_no_background() -> void {
    let figure = Figure(width: 120, height: 90)
    figure.axes().line([0.0, 1.0], [0.0, 1.0])
    assert_false(figure.is_frozen())
}

pub fn test_a_convex_fill_is_anti_aliased() -> void {
    let canvas = Canvas(Image(24, 24, 0xFF000000))
    canvas.fill_convex([(4.0, 4.0), (20.0, 6.0), (12.0, 20.0)], rgb(255, 255, 255))
    var partial = 0
    for row in 0..24 {
        for column in 0..24 {
            let value = red_of(canvas.get_pixel(column, row))
            if value > 0 and value < 255 { partial += 1 }
        }
    }
    assert_true(partial > 0)
}

pub fn test_blitting_rejects_a_buffer_smaller_than_it_claims() -> void {
    let canvas = Canvas(Image(16, 16))
    expect_error(ValueError, || canvas.blit_pixels([0, 0, 0], 4, 4, 0, 0, 4, 4))
}
