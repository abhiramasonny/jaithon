#: OpenCV-compatible computer vision for Jaithon.
#:
#: The names, argument order, and flag constants follow OpenCV, so a program
#: reads the same in both. Everything is re-exported here, which is the
#: `import jaicv as cv` entry point; the submodules are also importable
#: directly when only one is wanted.
#:
#: Images are `Mat`: dense, row-major, channels interleaved, BGR by convention.
#: Storage is one float32 device buffer per allocation, so every operation
#: below runs on the GPU and a `Mat` needs a Metal device to exist.
#:
#: ```
#: import jaicv as cv
#:
#: let frame = cv.imread("photo.jpg")
#: let gray = cv.cvt_color(frame, cv.COLOR_BGR2GRAY)
#: let edges = cv.canny(gray, 80.0, 200.0)
#: cv.imwrite("edges.png", edges)
#: ```
import std.gpu as gpu

from .core.types import CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F
from .core.types import CV_8UC1, CV_8UC2, CV_8UC3, CV_8UC4, CV_8SC1, CV_8SC3
from .core.types import CV_16UC1, CV_16UC3, CV_16SC1, CV_16SC3
from .core.types import CV_32SC1, CV_32SC2, CV_32SC3, CV_32SC4
from .core.types import CV_32FC1, CV_32FC2, CV_32FC3, CV_32FC4, CV_64FC1, CV_64FC3
from .core.types import CV_MAKETYPE, CV_MAT_CN, CV_MAT_DEPTH, type_to_string
from .core.types import depth_is_integer, depth_max, depth_min
from .core.types import Point, Point2f, Point3f, Range, Rect, RotatedRect, Scalar
from .core.types import Size, Size2f, TermCriteria
from .core.types import BORDER_CONSTANT, BORDER_DEFAULT, BORDER_ISOLATED, BORDER_REFLECT
from .core.types import BORDER_REFLECT_101, BORDER_REPLICATE, BORDER_WRAP
from .core.types import INTER_AREA, INTER_CUBIC, INTER_LANCZOS4, INTER_LINEAR, INTER_NEAREST
from .core.types import WARP_INVERSE_MAP
from .core.types import CMP_EQ, CMP_GE, CMP_GT, CMP_LE, CMP_LT, CMP_NE
from .core.types import NORM_HAMMING, NORM_INF, NORM_L1, NORM_L2, NORM_L2SQR, NORM_MINMAX
from .core.types import REDUCE_AVG, REDUCE_MAX, REDUCE_MIN, REDUCE_SUM
from .core.types import SORT_ASCENDING, SORT_DESCENDING, SORT_EVERY_COLUMN, SORT_EVERY_ROW
from .core.types import FILLED, LINE_4, LINE_8, LINE_AA

from .core.mat import Mat, cached_kernel, f32_bits, flip, merge, split, transpose, touch
from .core.arith import OP_ABSDIFF, OP_ADD, OP_AND, OP_DIVIDE, OP_MAX, OP_MIN, OP_MULTIPLY
from .core.arith import OP_OR, OP_POW, OP_RSUBTRACT, OP_SUBTRACT, OP_XOR
from .core.arith import UN_ABS, UN_EXP, UN_LOG, UN_NEGATE, UN_NOT, UN_RECIPROCAL
from .core.arith import UN_SQRT, UN_SQUARE
from .core.arith import abs_mat, absdiff, add, add_weighted, binary_op, bitwise_and
from .core.arith import bitwise_not, bitwise_or, bitwise_xor, cart_to_polar, compare
from .core.arith import compare_scalar, convert_scale_abs, divide, exp_mat, in_range
from .core.arith import log_mat, lut, magnitude, max_mat, min_mat, multiply, phase
from .core.arith import polar_to_cart, pow_mat, scalar_op, scale_add, sqrt_mat, subtract
from .core.arith import unary_op
from .core.stats import channel_stats, count_non_zero, mean, mean_std_dev, mean_value
from .core.stats import min_max, min_max_loc, norm, normalize, reduce, sum_mat, trace

from .imgproc.color import cvt_color, cvt_color_nv12, cvt_color_yuy2
from .imgproc.color import COLOR_BGR2BGRA, COLOR_BGR2GRAY, COLOR_BGR2HLS, COLOR_BGR2HSV
from .imgproc.color import COLOR_BGR2Lab, COLOR_BGR2RGB, COLOR_BGR2RGBA, COLOR_BGR2XYZ
from .imgproc.color import COLOR_BGR2YCrCb, COLOR_BGR2YUV, COLOR_BGRA2BGR, COLOR_BGRA2GRAY
from .imgproc.color import COLOR_BGRA2RGB, COLOR_BGRA2RGBA, COLOR_GRAY2BGR, COLOR_GRAY2BGRA
from .imgproc.color import COLOR_GRAY2RGB, COLOR_GRAY2RGBA, COLOR_HLS2BGR, COLOR_HLS2RGB
from .imgproc.color import COLOR_HSV2BGR, COLOR_HSV2RGB, COLOR_Lab2BGR, COLOR_Lab2RGB
from .imgproc.color import COLOR_RGB2BGR, COLOR_RGB2BGRA, COLOR_RGB2GRAY, COLOR_RGB2HLS
from .imgproc.color import COLOR_RGB2HSV, COLOR_RGB2Lab, COLOR_RGB2RGBA, COLOR_RGB2XYZ
from .imgproc.color import COLOR_RGB2YCrCb, COLOR_RGB2YUV, COLOR_RGBA2BGR, COLOR_RGBA2BGRA
from .imgproc.color import COLOR_RGBA2GRAY, COLOR_RGBA2RGB, COLOR_XYZ2BGR, COLOR_XYZ2RGB
from .imgproc.color import COLOR_YCrCb2BGR, COLOR_YCrCb2RGB, COLOR_YUV2BGR, COLOR_YUV2RGB
from .imgproc.color import COLOR_YUV2BGR_NV12, COLOR_YUV2BGR_YUY2
from .imgproc.color import COLOR_YUV2RGB_NV12, COLOR_YUV2RGB_YUY2

from .imgproc.geometric import copy_make_border, crop, get_affine_transform
from .imgproc.geometric import get_perspective_transform, get_rotation_matrix_2d
from .imgproc.geometric import invert3, invert_transform, perspective_transform
from .imgproc.geometric import pyr_down, pyr_up, remap, resize, solve_dense
from .imgproc.geometric import warp_affine, warp_perspective

from .imgproc.filter import bilateral_filter, blur, box_filter, filter_2d, gaussian_blur
from .imgproc.filter import get_derivative_kernels, get_gaussian_kernel, laplacian
from .imgproc.filter import median_blur, scharr, sep_filter_2d, sobel, sqr_box_filter

from .imgproc.threshold import adaptive_threshold, histogram_256, otsu_threshold
from .imgproc.threshold import threshold, triangle_threshold
from .imgproc.threshold import ADAPTIVE_THRESH_GAUSSIAN_C, ADAPTIVE_THRESH_MEAN_C
from .imgproc.threshold import THRESH_BINARY, THRESH_BINARY_INV, THRESH_OTSU
from .imgproc.threshold import THRESH_TOZERO, THRESH_TOZERO_INV, THRESH_TRIANGLE, THRESH_TRUNC

from .imgproc.morph import dilate, erode, get_structuring_element, morphology_ex
from .imgproc.morph import MORPH_BLACKHAT, MORPH_CLOSE, MORPH_CROSS, MORPH_DILATE
from .imgproc.morph import MORPH_ELLIPSE, MORPH_ERODE, MORPH_GRADIENT, MORPH_HITMISS
from .imgproc.morph import MORPH_OPEN, MORPH_RECT, MORPH_TOPHAT

from .imgproc.edges import canny

from .imgproc.drawing import arrowed_line, circle, clip_line, draw_marker, ellipse
from .imgproc.drawing import ellipse_2_poly, ellipse_rotated, fill_convex_poly, fill_poly
from .imgproc.drawing import line, line_points, polylines, rectangle, rectangle_rect
from .imgproc.drawing import CAP_END, CAP_START, MAX_THICKNESS, XY_ONE, XY_SHIFT
from .imgproc.drawing import MARKER_CROSS, MARKER_DIAMOND, MARKER_SQUARE, MARKER_STAR
from .imgproc.drawing import MARKER_TILTED_CROSS, MARKER_TRIANGLE_DOWN, MARKER_TRIANGLE_UP

from .imgproc.contours import draw_contours, find_contours, Hierarchy
from .imgproc.contours import CHAIN_APPROX_NONE, CHAIN_APPROX_SIMPLE
from .imgproc.contours import RETR_CCOMP, RETR_EXTERNAL, RETR_LIST, RETR_TREE

from .imgproc.shape import approx_poly_dp, arc_length, bounding_rect, box_points
from .imgproc.shape import contour_area, convex_hull, convex_hull_indices, fit_ellipse
from .imgproc.shape import fit_line, hu_moments, is_contour_convex, match_shapes
from .imgproc.shape import min_area_rect, min_enclosing_circle, moments, moments_f
from .imgproc.shape import moments_image, point_polygon_test, Moments
from .imgproc.shape import CONTOURS_MATCH_I1, CONTOURS_MATCH_I2, CONTOURS_MATCH_I3
from .imgproc.shape import DIST_FAIR, DIST_HUBER, DIST_L1, DIST_L2, DIST_WELSCH

from .imgcodecs.codecs import have_image_reader, have_image_writer, imdecode, imencode
from .imgcodecs.codecs import imread, imwrite, sniff
from .imgcodecs.codecs import IMREAD_ANYCOLOR, IMREAD_ANYDEPTH, IMREAD_COLOR
from .imgcodecs.codecs import IMREAD_GRAYSCALE, IMREAD_UNCHANGED
from .imgcodecs.codecs import IMWRITE_JPEG_QUALITY, IMWRITE_PNG_COMPRESSION

from .videoio.capture import VideoCapture, device_count, device_name, device_names
from .videoio.capture import permission, PERMISSION_HELP
from .videoio.capture import CAP_PROP_CHANNEL, CAP_PROP_FPS, CAP_PROP_FRAME_HEIGHT
from .videoio.capture import CAP_PROP_FRAME_WIDTH, CAP_PROP_POS_FRAMES
from .videoio.capture import PERMISSION_DENIED, PERMISSION_GRANTED
from .videoio.capture import PERMISSION_UNAVAILABLE, PERMISSION_UNDECIDED

export { CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F }
export { CV_8UC1, CV_8UC2, CV_8UC3, CV_8UC4, CV_8SC1, CV_8SC3 }
export { CV_16UC1, CV_16UC3, CV_16SC1, CV_16SC3 }
export { CV_32SC1, CV_32SC2, CV_32SC3, CV_32SC4 }
export { CV_32FC1, CV_32FC2, CV_32FC3, CV_32FC4, CV_64FC1, CV_64FC3 }
export { CV_MAKETYPE, CV_MAT_CN, CV_MAT_DEPTH, type_to_string }
export { depth_is_integer, depth_max, depth_min }
export { Point, Point2f, Point3f, Range, Rect, RotatedRect, Scalar, Size, Size2f }
export { TermCriteria }
export { BORDER_CONSTANT, BORDER_DEFAULT, BORDER_ISOLATED, BORDER_REFLECT }
export { BORDER_REFLECT_101, BORDER_REPLICATE, BORDER_WRAP }
export { INTER_AREA, INTER_CUBIC, INTER_LANCZOS4, INTER_LINEAR, INTER_NEAREST }
export { WARP_INVERSE_MAP }
export { CMP_EQ, CMP_GE, CMP_GT, CMP_LE, CMP_LT, CMP_NE }
export { NORM_HAMMING, NORM_INF, NORM_L1, NORM_L2, NORM_L2SQR, NORM_MINMAX }
export { REDUCE_AVG, REDUCE_MAX, REDUCE_MIN, REDUCE_SUM }
export { SORT_ASCENDING, SORT_DESCENDING, SORT_EVERY_COLUMN, SORT_EVERY_ROW }
export { FILLED, LINE_4, LINE_8, LINE_AA }

export { Mat, cached_kernel, f32_bits, flip, merge, split, transpose, touch }
export { OP_ABSDIFF, OP_ADD, OP_AND, OP_DIVIDE, OP_MAX, OP_MIN, OP_MULTIPLY }
export { OP_OR, OP_POW, OP_RSUBTRACT, OP_SUBTRACT, OP_XOR }
export { UN_ABS, UN_EXP, UN_LOG, UN_NEGATE, UN_NOT, UN_RECIPROCAL, UN_SQRT, UN_SQUARE }
export { abs_mat, absdiff, add, add_weighted, binary_op, bitwise_and, bitwise_not }
export { bitwise_or, bitwise_xor, cart_to_polar, compare, compare_scalar }
export { convert_scale_abs, divide, exp_mat, in_range, log_mat, lut, magnitude }
export { max_mat, min_mat, multiply, phase, polar_to_cart, pow_mat, scalar_op }
export { scale_add, sqrt_mat, subtract, unary_op }
export { channel_stats, count_non_zero, mean, mean_std_dev, mean_value }
export { min_max, min_max_loc, norm, normalize, reduce, sum_mat, trace }

export { cvt_color, cvt_color_nv12, cvt_color_yuy2 }
export { COLOR_BGR2BGRA, COLOR_BGR2GRAY, COLOR_BGR2HLS, COLOR_BGR2HSV }
export { COLOR_BGR2Lab, COLOR_BGR2RGB, COLOR_BGR2RGBA, COLOR_BGR2XYZ }
export { COLOR_BGR2YCrCb, COLOR_BGR2YUV, COLOR_BGRA2BGR, COLOR_BGRA2GRAY }
export { COLOR_BGRA2RGB, COLOR_BGRA2RGBA, COLOR_GRAY2BGR, COLOR_GRAY2BGRA }
export { COLOR_GRAY2RGB, COLOR_GRAY2RGBA, COLOR_HLS2BGR, COLOR_HLS2RGB }
export { COLOR_HSV2BGR, COLOR_HSV2RGB, COLOR_Lab2BGR, COLOR_Lab2RGB }
export { COLOR_RGB2BGR, COLOR_RGB2BGRA, COLOR_RGB2GRAY, COLOR_RGB2HLS }
export { COLOR_RGB2HSV, COLOR_RGB2Lab, COLOR_RGB2RGBA, COLOR_RGB2XYZ }
export { COLOR_RGB2YCrCb, COLOR_RGB2YUV, COLOR_RGBA2BGR, COLOR_RGBA2BGRA }
export { COLOR_RGBA2GRAY, COLOR_RGBA2RGB, COLOR_XYZ2BGR, COLOR_XYZ2RGB }
export { COLOR_YCrCb2BGR, COLOR_YCrCb2RGB, COLOR_YUV2BGR, COLOR_YUV2RGB }
export { COLOR_YUV2BGR_NV12, COLOR_YUV2BGR_YUY2, COLOR_YUV2RGB_NV12, COLOR_YUV2RGB_YUY2 }

export { copy_make_border, crop, get_affine_transform, get_perspective_transform }
export { get_rotation_matrix_2d, invert3, invert_transform, perspective_transform }
export { pyr_down, pyr_up, remap, resize, solve_dense, warp_affine, warp_perspective }

export { bilateral_filter, blur, box_filter, filter_2d, gaussian_blur }
export { get_derivative_kernels, get_gaussian_kernel, laplacian, median_blur }
export { scharr, sep_filter_2d, sobel, sqr_box_filter }

export { adaptive_threshold, histogram_256, otsu_threshold, threshold, triangle_threshold }
export { ADAPTIVE_THRESH_GAUSSIAN_C, ADAPTIVE_THRESH_MEAN_C }
export { THRESH_BINARY, THRESH_BINARY_INV, THRESH_OTSU }
export { THRESH_TOZERO, THRESH_TOZERO_INV, THRESH_TRIANGLE, THRESH_TRUNC }

export { dilate, erode, get_structuring_element, morphology_ex }
export { MORPH_BLACKHAT, MORPH_CLOSE, MORPH_CROSS, MORPH_DILATE }
export { MORPH_ELLIPSE, MORPH_ERODE, MORPH_GRADIENT, MORPH_HITMISS }
export { MORPH_OPEN, MORPH_RECT, MORPH_TOPHAT }

export { canny }

export { arrowed_line, circle, clip_line, draw_marker, ellipse, ellipse_2_poly }
export { ellipse_rotated, fill_convex_poly, fill_poly, line, line_points, polylines }
export { rectangle, rectangle_rect }
export { CAP_END, CAP_START, MAX_THICKNESS, XY_ONE, XY_SHIFT }
export { MARKER_CROSS, MARKER_DIAMOND, MARKER_SQUARE, MARKER_STAR }
export { MARKER_TILTED_CROSS, MARKER_TRIANGLE_DOWN, MARKER_TRIANGLE_UP }

export { draw_contours, find_contours, Hierarchy }
export { CHAIN_APPROX_NONE, CHAIN_APPROX_SIMPLE }
export { RETR_CCOMP, RETR_EXTERNAL, RETR_LIST, RETR_TREE }

export { approx_poly_dp, arc_length, bounding_rect, box_points, contour_area }
export { convex_hull, convex_hull_indices, fit_ellipse, fit_line, hu_moments }
export { is_contour_convex, match_shapes, min_area_rect, min_enclosing_circle }
export { moments, moments_f, moments_image, point_polygon_test, Moments }
export { CONTOURS_MATCH_I1, CONTOURS_MATCH_I2, CONTOURS_MATCH_I3 }
export { DIST_FAIR, DIST_HUBER, DIST_L1, DIST_L2, DIST_WELSCH }

export { have_image_reader, have_image_writer, imdecode, imencode, imread, imwrite, sniff }
export { IMREAD_ANYCOLOR, IMREAD_ANYDEPTH, IMREAD_COLOR, IMREAD_GRAYSCALE, IMREAD_UNCHANGED }
export { IMWRITE_JPEG_QUALITY, IMWRITE_PNG_COMPRESSION }

export { VideoCapture, device_count, device_name, device_names, permission, PERMISSION_HELP }
export { CAP_PROP_CHANNEL, CAP_PROP_FPS, CAP_PROP_FRAME_HEIGHT }
export { CAP_PROP_FRAME_WIDTH, CAP_PROP_POS_FRAMES }
export { PERMISSION_DENIED, PERMISSION_GRANTED, PERMISSION_UNAVAILABLE, PERMISSION_UNDECIDED }

#: Whether this machine has the GPU jaicv needs.
pub fn is_available() -> bool { return gpu.is_available() }

#: Name of the device jaicv will run on.
pub fn device_name_gpu() -> str { return gpu.device_name() }

#: Wait for every queued operation to finish.
#:
#: Every jaicv call queues work and returns; reading a `Mat` synchronises on
#: its own. Call this when timing something, or before reading a buffer
#: through a route that does not.
pub fn synchronize() -> void { gpu.synchronize() }
