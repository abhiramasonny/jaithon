#: Every algorithm family under `std.algo` in one place: sorting and
#: searching, dynamic programming, number theory, computational geometry,
#: string and text algorithms, statistics, and the codec/compression pair.
#:
#: Each sibling file groups one family and stays independently importable —
#: `from std.algo.sort import sort_by` keeps working exactly as before. This
#: facade exists for the times a caller wants the family, not the file.
#:
#: ```
#: from std.algo import sort_by, natural_order, binary_search
#: let xs = sort_by([3, 1, 2], natural_order)
#: binary_search(xs, 2)   # 1
#: ```
from .bwt import rle_encode, rle_decode, bwt_encode, bwt_decode
from .codecs import utf8_encode, utf8_decode, base64_encode, base64_decode, hex_encode, hex_decode, crc32, fnv1a
from .dp import EditOp, Subarray, Submatrix, fibonacci, fibonacci_matrix, longest_common_subsequence, longest_increasing_subsequence, edit_distance, edit_script, apply_edit_script, knapsack_01, knapsack_unbounded, coin_change_min, coin_change_ways, subset_sum, partition_equal, matrix_chain_order, rod_cutting, max_subarray, max_submatrix, house_robber, jump_game, unique_paths, word_break, palindrome_partitioning, catalan, memoize_recursive
from .geometry import orientation, sign_within, within_bounds
from .huffman import HuffmanCode
from .lz77 import LZ77Token, lz77_compress, lz77_decompress
from .numtheory import is_prime, sieve_of_eratosthenes, segmented_sieve, factorize, prime_factors, divisors, divisor_count, divisor_sum, euler_phi, mobius, extended_gcd, mod_inverse, mod_pow, chinese_remainder, discrete_log, primitive_root, legendre_symbol, jacobi_symbol, tonelli_shanks, continued_fraction, convergents, rational_approximation, next_prime, prev_prime, nth_prime, prime_count
from .phonetics import soundex, metaphone
from .point_cloud import convex_hull, closest_pair, centroid, bounding_box, diameter, minimum_enclosing_circle, bounds_of
from .polygon import Polygon, winding_number, point_in_polygon, point_in_polygon_crossing, polygon_signed_area, polygon_area, polygon_perimeter, polygon_centroid, polygon_clip, triangulate
from .search import AhoCorasick, binary_search, lower_bound, upper_bound, equal_range, binary_search_by, exponential_search, interpolation_search, ternary_search, linear_search, find_all, kmp_search, boyer_moore_search, rabin_karp_search, z_algorithm, aho_corasick
from .shapes import Line, Segment, Rect, Circle, circle_from_points, line_circle_intersection
from .sort import Comparator, natural_order, sort_by, sort_by_key, merge_sort, quick_sort, heap_sort, insertion_sort, tim_sort, counting_sort, radix_sort, bucket_sort, topological_sort, partial_sort, nth_element, is_sorted, merge
from .stat import mean, Summary, LinearFit, median, mode, multimode, quantile, variance, population_variance, stdev, population_stdev, covariance, correlation, spearman, z_score, normalize, standardize, min_max, value_range, iqr, outlier_bounds, outliers, moving_average, exponential_moving_average, linear_regression, polynomial_regression, polynomial_value, histogram, freedman_diaconis_bins, cumulative_distribution, normal_pdf, normal_cdf, erf, erfc, binomial_pmf, poisson_pmf
from .strings import DiffOp, palindrome_radii, longest_palindrome, z_function, prefix_function, least_rotation_index, minimal_rotation, diff, diff_lines
from .suffix_automaton import SuffixAutomaton
from .tokenize import tokenize, word_frequency, top_words, char_ngrams, word_ngrams
from .vectors import EPSILON, Point2, Vec3

export { rle_encode, rle_decode, bwt_encode, bwt_decode }
export { utf8_encode, utf8_decode, base64_encode, base64_decode, hex_encode, hex_decode, crc32, fnv1a }
export { EditOp, Subarray, Submatrix, fibonacci, fibonacci_matrix, longest_common_subsequence, longest_increasing_subsequence, edit_distance, edit_script, apply_edit_script, knapsack_01, knapsack_unbounded, coin_change_min, coin_change_ways, subset_sum, partition_equal, matrix_chain_order, rod_cutting, max_subarray, max_submatrix, house_robber, jump_game, unique_paths, word_break, palindrome_partitioning, catalan, memoize_recursive }
export { orientation, sign_within, within_bounds }
export { HuffmanCode }
export { LZ77Token, lz77_compress, lz77_decompress }
export { is_prime, sieve_of_eratosthenes, segmented_sieve, factorize, prime_factors, divisors, divisor_count, divisor_sum, euler_phi, mobius, extended_gcd, mod_inverse, mod_pow, chinese_remainder, discrete_log, primitive_root, legendre_symbol, jacobi_symbol, tonelli_shanks, continued_fraction, convergents, rational_approximation, next_prime, prev_prime, nth_prime, prime_count }
export { soundex, metaphone }
export { convex_hull, closest_pair, centroid, bounding_box, diameter, minimum_enclosing_circle, bounds_of }
export { Polygon, winding_number, point_in_polygon, point_in_polygon_crossing, polygon_signed_area, polygon_area, polygon_perimeter, polygon_centroid, polygon_clip, triangulate }
export { AhoCorasick, binary_search, lower_bound, upper_bound, equal_range, binary_search_by, exponential_search, interpolation_search, ternary_search, linear_search, find_all, kmp_search, boyer_moore_search, rabin_karp_search, z_algorithm, aho_corasick }
export { Line, Segment, Rect, Circle, circle_from_points, line_circle_intersection }
export { Comparator, natural_order, sort_by, sort_by_key, merge_sort, quick_sort, heap_sort, insertion_sort, tim_sort, counting_sort, radix_sort, bucket_sort, topological_sort, partial_sort, nth_element, is_sorted, merge }
export { mean, Summary, LinearFit, median, mode, multimode, quantile, variance, population_variance, stdev, population_stdev, covariance, correlation, spearman, z_score, normalize, standardize, min_max, value_range, iqr, outlier_bounds, outliers, moving_average, exponential_moving_average, linear_regression, polynomial_regression, polynomial_value, histogram, freedman_diaconis_bins, cumulative_distribution, normal_pdf, normal_cdf, erf, erfc, binomial_pmf, poisson_pmf }
export { DiffOp, palindrome_radii, longest_palindrome, z_function, prefix_function, least_rotation_index, minimal_rotation, diff, diff_lines }
export { SuffixAutomaton }
export { tokenize, word_frequency, top_words, char_ngrams, word_ngrams }
export { EPSILON, Point2, Vec3 }
