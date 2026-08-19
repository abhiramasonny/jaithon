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

from .imgproc.segmentation import colour_labels, connected_components
from .imgproc.segmentation import connected_components_with_stats, distance_transform
from .imgproc.segmentation import flood_fill, integral, watershed, ComponentStats
from .imgproc.segmentation import DIST_C, DIST_MASK_3, DIST_MASK_5
from .imgproc.segmentation import FLOODFILL_FIXED_RANGE, FLOODFILL_MASK_ONLY

from .imgproc.hist import calc_back_project, calc_hist, compare_hist, create_clahe
from .imgproc.hist import equalize_hist, CLAHE
from .imgproc.hist import HISTCMP_BHATTACHARYYA, HISTCMP_CHISQR, HISTCMP_CHISQR_ALT
from .imgproc.hist import HISTCMP_CORREL, HISTCMP_INTERSECT, HISTCMP_KL_DIV

from .imgproc.features import corner_harris, corner_min_eigen_val, corner_sub_pix
from .imgproc.features import good_features_to_track, hough_circles, hough_lines
from .imgproc.features import hough_lines_p, match_template
from .imgproc.features import HoughCircle, HoughLine, HoughSegment
from .imgproc.features import HOUGH_GRADIENT, HOUGH_PROBABILISTIC, HOUGH_STANDARD
from .imgproc.features import TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR, TM_CCORR_NORMED
from .imgproc.features import TM_SQDIFF, TM_SQDIFF_NORMED

from .features2d.detect import draw_keypoints, draw_matches, fast
from .features2d.detect import BFMatcher, BRIEF, DMatch, KeyPoint, ORB
from .features2d.detect import SimpleBlobDetector, SimpleBlobDetectorParams
from .features2d.detect import DRAW_MATCHES_FLAGS_DEFAULT, DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
from .features2d.detect import FAST_FEATURE_DETECTOR_TYPE_5_8, FAST_FEATURE_DETECTOR_TYPE_7_12
from .features2d.detect import FAST_FEATURE_DETECTOR_TYPE_9_16, ORB_FAST_SCORE, ORB_HARRIS_SCORE

from .video.optflow import calc_optical_flow_farneback, calc_optical_flow_pyr_lk
from .video.optflow import flow_to_colour, polynomial_expansion
from .video.optflow import OPTFLOW_FARNEBACK_GAUSSIAN, OPTFLOW_USE_INITIAL_FLOW

from .video.background import create_background_subtractor_knn
from .video.background import create_background_subtractor_mog2
from .video.background import BackgroundSubtractorKNN, BackgroundSubtractorMOG2
from .video.background import FOREGROUND_VALUE, SHADOW_VALUE

from .video.track import cam_shift, mean_shift, KalmanFilter

from .highgui.window import create_trackbar, destroy_all_windows, destroy_window
from .highgui.window import get_trackbar_pos, get_window_image_rect, imshow, named_window
from .highgui.window import set_mouse_callback, set_trackbar_pos, set_window_title
from .highgui.window import wait_key, wait_key_ex, window_is_open
from .highgui.window import EVENT_LBUTTONDOWN, EVENT_LBUTTONUP, EVENT_MBUTTONDOWN
from .highgui.window import EVENT_MBUTTONUP, EVENT_MOUSEMOVE, EVENT_MOUSEWHEEL
from .highgui.window import EVENT_RBUTTONDOWN, EVENT_RBUTTONUP
from .highgui.window import WINDOW_AUTOSIZE, WINDOW_KEEPRATIO, WINDOW_NORMAL

from .objdetect.hog import HOGDescriptor, HOG_SIGNED, HOG_UNSIGNED
from .objdetect.cascade import group_rectangles, CascadeClassifier
from .objdetect.cascade import CASCADE_DO_CANNY_PRUNING, CASCADE_DO_ROUGH_SEARCH
from .objdetect.cascade import CASCADE_FIND_BIGGEST_OBJECT, CASCADE_SCALE_IMAGE
from .objdetect.galois import gf_div, gf_exp, gf_inv, gf_mul, rs_correct
from .objdetect.qr import alignment_centres, decode_grid, encode_grid, qr_encode
from .objdetect.qr import smallest_version, version_size, QRCode, QRCodeDetector
from .objdetect.xml import parse_xml, Node

from .calib3d.transform import estimate_affine_2d, estimate_affine_partial_2d
from .calib3d.transform import find_homography, get_optimal_new_camera_matrix
from .calib3d.transform import init_undistort_rectify_map, project_points, refine_homography
from .calib3d.transform import rodrigues, triangulate_points, undistort_points
from .calib3d.transform import LMEDS, RANSAC

from .calib3d.calibrate import calibrate_camera, draw_chessboard_corners
from .calib3d.calibrate import find_chessboard_corners, solve_pnp, undistort
from .calib3d.calibrate import SOLVEPNP_ITERATIVE

from .core.linalg import identity, inverse, levenberg_marquardt, null_space, symmetric_eigen

from .core.dft import dct, dft, get_optimal_dft_size, idft, mul_spectrums
from .core.dft import DFT_COMPLEX_OUTPUT, DFT_INVERSE, DFT_REAL_OUTPUT, DFT_ROWS, DFT_SCALE

from .core.cluster import kmeans, PCA
from .core.cluster import KMEANS_PP_CENTERS, KMEANS_RANDOM_CENTERS, KMEANS_USE_INITIAL_LABELS

from .photo.denoise import detail_enhance, edge_preserving_filter
from .photo.denoise import fast_nl_means_denoising, fast_nl_means_denoising_colored
from .photo.denoise import inpaint, stylization
from .photo.denoise import INPAINT_NS, INPAINT_TELEA, NORMCONV_FILTER, RECURS_FILTER

from .ml.models import ann_mlp_create, d_trees_create, k_nearest_create
from .ml.models import logistic_regression_create, normal_bayes_classifier_create
from .ml.models import r_trees_create, svm_create
from .ml.models import ANN_MLP, DTrees, KNearest, LogisticRegression
from .ml.models import NormalBayesClassifier, RTrees, SVM
from .ml.models import COL_SAMPLE, ROW_SAMPLE, SVM_C_SVC, SVM_EPS_SVR
from .ml.models import SVM_LINEAR, SVM_POLY, SVM_RBF

from .dnn.blob import blob_from_image, blob_from_images, images_from_blob
from .dnn.blob import net_from, nms_boxes, nms_boxes_batched, Net, Runnable

from .core.misc import border_interpolate, calc_covar_matrix, check_range, determinant
from .core.misc import eigen, extract_channel, find_non_zero, hconcat, insert_channel
from .core.misc import mahalanobis, min_max_idx, mul_transposed, patch_nans, randn, randu
from .core.misc import repeat, rotate, set_identity, set_rng_seed, solve_poly
from .core.misc import sort, sort_idx, vconcat
from .core.misc import ROTATE_180, ROTATE_90_CLOCKWISE, ROTATE_90_COUNTERCLOCKWISE

from .imgproc.misc import accumulate, accumulate_product, accumulate_square
from .imgproc.misc import accumulate_weighted, apply_color_map, create_hanning_window
from .imgproc.misc import get_rect_sub_pix, intersect_convex_convex, min_enclosing_triangle
from .imgproc.misc import phase_correlate, pyr_mean_shift_filtering
from .imgproc.misc import COLORMAP_AUTUMN, COLORMAP_BONE, COLORMAP_COOL, COLORMAP_HOT
from .imgproc.misc import COLORMAP_HSV, COLORMAP_JET, COLORMAP_OCEAN, COLORMAP_PINK
from .imgproc.misc import COLORMAP_RAINBOW, COLORMAP_SPRING, COLORMAP_SUMMER, COLORMAP_WINTER

from .imgcodecs.codecs import have_image_reader, have_image_writer, imdecode, imencode
from .imgcodecs.codecs import imread, imwrite, sniff
from .imgcodecs.codecs import IMREAD_ANYCOLOR, IMREAD_ANYDEPTH, IMREAD_COLOR
from .imgcodecs.codecs import IMREAD_GRAYSCALE, IMREAD_UNCHANGED
from .imgcodecs.codecs import IMWRITE_JPEG_QUALITY, IMWRITE_PNG_COMPRESSION

from .videoio.writer import fourcc, VideoWriter

from .videoio.reader import AviReader
from .videoio.capture import VideoCapture, device_count, device_name, device_names
from .videoio.capture import permission, PERMISSION_HELP
from .videoio.capture import CAP_PROP_CHANNEL, CAP_PROP_FPS, CAP_PROP_FRAME_HEIGHT
from .videoio.capture import CAP_PROP_FRAME_COUNT, CAP_PROP_FOURCC
from .videoio.capture import CAP_PROP_FRAME_WIDTH, CAP_PROP_POS_FRAMES
from .videoio.capture import CAP_PROP_POS_AVI_RATIO, CAP_PROP_POS_MSEC
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

export { colour_labels, connected_components, connected_components_with_stats }
export { distance_transform, flood_fill, integral, watershed, ComponentStats }
export { DIST_C, DIST_MASK_3, DIST_MASK_5 }
export { FLOODFILL_FIXED_RANGE, FLOODFILL_MASK_ONLY }

export { calc_back_project, calc_hist, compare_hist, create_clahe, equalize_hist, CLAHE }
export { HISTCMP_BHATTACHARYYA, HISTCMP_CHISQR, HISTCMP_CHISQR_ALT }
export { HISTCMP_CORREL, HISTCMP_INTERSECT, HISTCMP_KL_DIV }

export { corner_harris, corner_min_eigen_val, corner_sub_pix, good_features_to_track }
export { hough_circles, hough_lines, hough_lines_p, match_template }
export { HoughCircle, HoughLine, HoughSegment }
export { HOUGH_GRADIENT, HOUGH_PROBABILISTIC, HOUGH_STANDARD }
export { TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR, TM_CCORR_NORMED }
export { TM_SQDIFF, TM_SQDIFF_NORMED }

export { draw_keypoints, draw_matches, fast }
export { BFMatcher, BRIEF, DMatch, KeyPoint, ORB }
export { SimpleBlobDetector, SimpleBlobDetectorParams }
export { DRAW_MATCHES_FLAGS_DEFAULT, DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS }
export { FAST_FEATURE_DETECTOR_TYPE_5_8, FAST_FEATURE_DETECTOR_TYPE_7_12 }
export { FAST_FEATURE_DETECTOR_TYPE_9_16, ORB_FAST_SCORE, ORB_HARRIS_SCORE }

export { calc_optical_flow_farneback, calc_optical_flow_pyr_lk }
export { flow_to_colour, polynomial_expansion }
export { OPTFLOW_FARNEBACK_GAUSSIAN, OPTFLOW_USE_INITIAL_FLOW }

export { create_background_subtractor_knn, create_background_subtractor_mog2 }
export { BackgroundSubtractorKNN, BackgroundSubtractorMOG2 }
export { FOREGROUND_VALUE, SHADOW_VALUE }

export { cam_shift, mean_shift, KalmanFilter }

export { create_trackbar, destroy_all_windows, destroy_window, get_trackbar_pos }
export { get_window_image_rect, imshow, named_window, set_mouse_callback }
export { set_trackbar_pos, set_window_title, wait_key, wait_key_ex, window_is_open }
export { EVENT_LBUTTONDOWN, EVENT_LBUTTONUP, EVENT_MBUTTONDOWN }
export { EVENT_MBUTTONUP, EVENT_MOUSEMOVE, EVENT_MOUSEWHEEL }
export { EVENT_RBUTTONDOWN, EVENT_RBUTTONUP }
export { WINDOW_AUTOSIZE, WINDOW_KEEPRATIO, WINDOW_NORMAL }

export { group_rectangles, CascadeClassifier }
export { HOGDescriptor, HOG_SIGNED, HOG_UNSIGNED }
export { CASCADE_DO_CANNY_PRUNING, CASCADE_DO_ROUGH_SEARCH }
export { CASCADE_FIND_BIGGEST_OBJECT, CASCADE_SCALE_IMAGE }
export { parse_xml, Node }
export { gf_div, gf_exp, gf_inv, gf_mul, rs_correct }
export { alignment_centres, decode_grid, encode_grid, qr_encode }
export { smallest_version, version_size, QRCode, QRCodeDetector }

export { estimate_affine_2d, estimate_affine_partial_2d, find_homography }
export { get_optimal_new_camera_matrix, init_undistort_rectify_map, project_points }
export { refine_homography, rodrigues, triangulate_points, undistort_points }
export { LMEDS, RANSAC }

export { calibrate_camera, draw_chessboard_corners, find_chessboard_corners }
export { solve_pnp, undistort, SOLVEPNP_ITERATIVE }

export { identity, inverse, levenberg_marquardt, null_space, symmetric_eigen }

export { dct, dft, get_optimal_dft_size, idft, mul_spectrums }
export { DFT_COMPLEX_OUTPUT, DFT_INVERSE, DFT_REAL_OUTPUT, DFT_ROWS, DFT_SCALE }

export { kmeans, PCA }
export { KMEANS_PP_CENTERS, KMEANS_RANDOM_CENTERS, KMEANS_USE_INITIAL_LABELS }

export { detail_enhance, edge_preserving_filter, fast_nl_means_denoising }
export { fast_nl_means_denoising_colored, inpaint, stylization }
export { INPAINT_NS, INPAINT_TELEA, NORMCONV_FILTER, RECURS_FILTER }

export { ann_mlp_create, d_trees_create, k_nearest_create }
export { logistic_regression_create, normal_bayes_classifier_create }
export { r_trees_create, svm_create }
export { ANN_MLP, DTrees, KNearest, LogisticRegression }
export { NormalBayesClassifier, RTrees, SVM }
export { COL_SAMPLE, ROW_SAMPLE, SVM_C_SVC, SVM_EPS_SVR }
export { SVM_LINEAR, SVM_POLY, SVM_RBF }

export { blob_from_image, blob_from_images, images_from_blob }
export { net_from, nms_boxes, nms_boxes_batched, Net, Runnable }

export { border_interpolate, calc_covar_matrix, check_range, determinant, eigen }
export { extract_channel, find_non_zero, hconcat, insert_channel, mahalanobis }
export { min_max_idx, mul_transposed, patch_nans, randn, randu, repeat, rotate }
export { set_identity, set_rng_seed, solve_poly, sort, sort_idx, vconcat }
export { ROTATE_180, ROTATE_90_CLOCKWISE, ROTATE_90_COUNTERCLOCKWISE }

export { accumulate, accumulate_product, accumulate_square, accumulate_weighted }
export { apply_color_map, create_hanning_window, get_rect_sub_pix }
export { intersect_convex_convex, min_enclosing_triangle, phase_correlate }
export { pyr_mean_shift_filtering }
export { COLORMAP_AUTUMN, COLORMAP_BONE, COLORMAP_COOL, COLORMAP_HOT }
export { COLORMAP_HSV, COLORMAP_JET, COLORMAP_OCEAN, COLORMAP_PINK }
export { COLORMAP_RAINBOW, COLORMAP_SPRING, COLORMAP_SUMMER, COLORMAP_WINTER }

export { have_image_reader, have_image_writer, imdecode, imencode, imread, imwrite, sniff }
export { IMREAD_ANYCOLOR, IMREAD_ANYDEPTH, IMREAD_COLOR, IMREAD_GRAYSCALE, IMREAD_UNCHANGED }
export { IMWRITE_JPEG_QUALITY, IMWRITE_PNG_COMPRESSION }

export { fourcc, VideoWriter }

export { AviReader, VideoCapture, device_count, device_name, device_names }
export { permission, PERMISSION_HELP }
export { CAP_PROP_CHANNEL, CAP_PROP_FPS, CAP_PROP_FRAME_HEIGHT }
export { CAP_PROP_FRAME_COUNT, CAP_PROP_FOURCC }
export { CAP_PROP_FRAME_WIDTH, CAP_PROP_POS_FRAMES }
export { CAP_PROP_POS_AVI_RATIO, CAP_PROP_POS_MSEC }
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
