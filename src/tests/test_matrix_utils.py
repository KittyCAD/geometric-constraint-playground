import numpy as np

from newton.matrix_utils import compute_rank

EPS = 1e-10


def test_full_rank_square_matrix():
    # Create a full rank square matrix.
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    rank = compute_rank(matrix, EPS)
    assert rank == 2


def test_rank_deficient_square_matrix():
    # Create a rank deficient square matrix; second row is a multiple of the first.
    matrix = np.array([[1.0, 2.0], [2.0, 4.0]])
    rank = compute_rank(matrix, EPS)
    assert rank == 1


def test_tall_rectangular_matrix():
    # Create a tall rectangular matrix (more rows than columns).
    matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    rank = compute_rank(matrix, EPS)
    assert rank == 2


def test_wide_rectangular_matrix():
    # Create a wide rectangular matrix (more columns than rows).
    matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    rank = compute_rank(matrix, EPS)
    assert rank == 2


def test_rank_with_transpose():
    # Test that rank of matrix and its transpose are the same.
    matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    rank = compute_rank(matrix, EPS)
    rank_transpose = compute_rank(matrix.T, EPS)
    assert rank == rank_transpose


def test_tolerance_effect():
    # Test how tolerance affects rank computation.
    # Create a matrix with small but non-zero singular values.
    matrix = np.array([[1.0, 0], [0, 1e-3]])
    rank_strict = compute_rank(matrix, 1e-2)
    rank_loose = compute_rank(matrix, 1e-9)
    assert rank_strict == 1
    assert rank_loose == 2


def test_fallback_to_qr_failure():
    matrix = np.array([[1.0, 1.0], [1.0, 1.0]])

    # Patch the QR function to simulate failure.
    import unittest.mock

    with unittest.mock.patch(
        "numpy.linalg.qr", side_effect=ValueError("Mocked QR failure")
    ):
        rank = compute_rank(matrix, EPS)
        assert rank == 1
