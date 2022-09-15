from typing import Iterable, Optional, Tuple

import numpy as np
import numpy.linalg as npla


def compute_robust_linear_normed_L(
    A: np.ndarray,
    b: np.ndarray,
    r_x: float,
    r_y: float,
    regcoef_x: float,
    regcoef_y: float,
) -> float:
    """
    Compute Lipschitz constant for Robust Linear Regression function
    (normed, i.e. 1 / num_samples * (...) + regcoef_x/2 ||x||^2 - regcoef_y/2 ||y||^2).
    """

    lam = npla.svd(A.T.dot(A))[1].max() / A.shape[0]
    A_dot_one = npla.norm(A.mean(axis=0))
    L_xx = lam + 2 * r_y * A_dot_one + r_y**2 + regcoef_x
    L_yy = r_x**2 + regcoef_y
    L_xy = 2 * r_x * r_y + 2 * A_dot_one * r_x + b.mean()
    L = 2 * max(L_xx, L_yy, L_xy)
    return L


def compute_robust_linear_normed_delta(
    A: np.ndarray, Am: np.ndarray, r_x: float, r_y: float, num_parts: int
) -> float:
    """
    Compute similarity coefficient delta between whole dataset A and its part Am.
    """

    lam = npla.svd(A.T.dot(A) / num_parts - Am.T.dot(Am))[1].max() / Am.shape[0]
    A_dot_one = npla.norm(A.mean(axis=0) - Am.mean(axis=0))
    delta_xx = lam + 2 * r_y * A_dot_one
    delta_yy = 0
    delta_xy = 2 * r_x * A_dot_one
    delta = 2 * max(delta_xx, delta_yy, delta_xy)
    return delta


def compute_robust_linear_normed_L_delta_mu(
    A: np.ndarray,
    b: np.ndarray,
    part_sizes: Optional[Iterable],
    n_parts: Optional[int],
    r_x: float,
    r_y: float,
    regcoef_x: float,
    regcoef_y: float,
) -> Tuple[float, float, float]:

    if part_sizes is None and n_parts is None:
        raise ValueError("Please specify either part_sizes or n_parts")
    if part_sizes is not None and n_parts is not None:
        raise ValueError("Only one of part_sizes and n_parts should be specified")
    if part_sizes is None:
        size = A.shape[0] // n_parts
        part_sizes = [size] * n_parts
    if n_parts is None:
        n_parts = len(part_sizes)

    L_list = np.empty(n_parts, dtype=np.float32)
    delta_list = np.empty(n_parts, dtype=np.float32)
    start = 0
    for i, size in enumerate(part_sizes):
        L_list[i] = compute_robust_linear_normed_L(
            A[start : start + size],
            b[start : start + size],
            r_x,
            r_y,
            regcoef_x,
            regcoef_y,
        )
        delta_list[i] = compute_robust_linear_normed_delta(
            A, A[start : start + size], r_x, r_y, n_parts
        )
        start += size
    return np.max(L_list), np.max(delta_list), min(regcoef_x, regcoef_y)


def compute_lam_2(mat):
    eigs = np.sort(np.linalg.eigvals(mat))

    return max(np.abs(eigs[0]), np.abs(eigs[-2]))


def compute_lam(
    matrix: np.ndarray,
    eps_real: Optional[float] = 0.00001,
    eps_imag: Optional[float] = 0.00001,
) -> Tuple[float, float]:
    """
    Computes positive minimal and maximum eigen values of the matrix.
    """
    eigs = np.linalg.eigvals(matrix)
    positive_eigs = [
        eig.real for eig in eigs if eig.real > eps_real and eig.imag <= eps_imag
    ]

    return min(positive_eigs), max(positive_eigs)
