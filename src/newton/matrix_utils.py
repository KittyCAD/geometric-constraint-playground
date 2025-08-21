"""Matrix utilities for the Newton solver."""

from newton.backend import Vector, np


def compute_rank(matrix: Vector, tol: float) -> int:
    """
    Compute the rank of a matrix using the current global backend.

    Args:
        matrix: The input matrix, likely a Jacobian.
        tol: Tolerance for rank determination.

    Returns:
        The rank of the matrix.
    """
    try:
        _, r = np.linalg.qr(matrix)
        diag = np.abs(np.diag(r))
        qr_rank = np.sum(diag > tol)
        if qr_rank == 0 or np.any(np.isnan(diag)):
            raise ValueError("QR failed or is ambiguous.")
        return int(qr_rank)

    except Exception:
        s = np.linalg.svd(matrix, compute_uv=False)
        return int(np.sum(s > tol))
