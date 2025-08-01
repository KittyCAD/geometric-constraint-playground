from types import ModuleType

from numpy.typing import ArrayLike


def compute_rank(matrix: ArrayLike, tol: float, xp: ModuleType) -> int:
    """
    Compute the rank of a matrix using either numpy or jax.numpy.

    Args:
        matrix: The input matrix, likely a Jacobian.
        tol: Tolerance for rank determination.
        xp: The numpy-like module to use (np or jnp).

    Returns:
        The rank of the matrix.
    """
    try:
        _, r = xp.linalg.qr(matrix)
        diag = xp.abs(xp.diag(r))
        qr_rank = xp.sum(diag > tol)
        if qr_rank == 0 or xp.any(xp.isnan(diag)):
            raise ValueError("QR failed or is ambiguous.")
        return int(qr_rank)

    except Exception:
        s = xp.linalg.svd(matrix, compute_uv=False)
        return int(xp.sum(s > tol))
