import datetime
from typing import Callable, List, Optional, Tuple

import numpy as np
from src.logger import LoggerCentralized
from src.method import ConstraintsL2, Extragradient
from src.oracles import (
    ArrayPair,
    BaseSmoothSaddleOracle,
    LinearCombSaddleOracle,
    create_robust_linear_oracle,
)
from src.utils.compute_params import (
    compute_robust_linear_normed_L,
    compute_robust_linear_normed_L_delta_mu,
)


def grad_finite_diff(func, x, eps=1e-8):
    """
    Returns approximation of the gradient using finite differences:
        result_i := (f(x + eps * e_i) - f(x)) / eps,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    x, fval, dnum = x.astype(np.float64), func(x), np.zeros_like(x)

    grad = []
    n = x.size
    for i in range(n):
        dnum = np.zeros(n)
        dnum[i] = 1
        der = (func(x + eps * dnum) - func(x)) / eps
        grad.append(der)

    return np.array(grad)


def grad_finite_diff_saddle(func: Callable, z: ArrayPair, eps: float = 1e-8):
    """
    Same as grad_finite_diff, but for saddle-point problems

    Parameters
    ----------
    func: Callable
        Function from saddle-point problem. Takes two arguments x and y

    x: np.ndarray
        First argument of func

    y: np.ndarray
        Second argument of func

    eps: float
        Size of argument deviation
    """

    grad_x = []
    n = z.x.size
    for i in range(n):
        dnum = np.zeros(n)
        dnum[i] = 1.0
        arg = z.copy()
        arg.x = z.x + eps * dnum
        der = (func(arg) - func(z)) / eps
        grad_x.append(der)

    grad_y = []
    n = z.y.size
    for i in range(n):
        dnum = np.zeros(n)
        dnum[i] = 1.0
        arg = z.copy()
        arg.y = z.y + eps * dnum
        der = (func(arg) - func(z)) / eps
        grad_y.append(der)

    return np.array(grad_x), np.array(grad_y)


def hess_finite_diff(func, x, eps=1e-5):
    """
    Returns approximation of the Hessian using finite differences:
        result_{ij} := (f(x + eps * e_i + eps * e_j)
                               - f(x + eps * e_i)
                               - f(x + eps * e_j)
                               + f(x)) / eps^2,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    from itertools import combinations_with_replacement

    x, fval, dnum = (
        x.astype(np.float64),
        func(x).astype(np.float64),
        np.zeros((x.size, x.size), dtype=np.float64),
    )
    n = x.size
    hess = np.zeros((n, n))

    for i, j in combinations_with_replacement(range(x.size), 2):
        dnum_i = np.zeros(x.size)
        dnum_i[i] = eps
        dnum_j = np.zeros(x.size)
        dnum_j[j] = eps
        hess[i][j] = (
            func(x + dnum_i + dnum_j) - func(x + dnum_i) - func(x + dnum_j) + fval
        ) / eps**2
        hess[j][i] = hess[i][j]
    return np.array(hess)


def solve_with_extragradient(
    oracle: BaseSmoothSaddleOracle,
    stepsize: float,
    r_x: float,
    r_y: float,
    z_0: ArrayPair,
    tolerance: Optional[float],
    num_iter: int,
    max_time: Optional[float],
    z_true: Optional[ArrayPair] = None,
) -> LoggerCentralized:
    print("Solving with extragradient...")
    logger_extragradient = LoggerCentralized(z_true=z_true)
    extragradient = Extragradient(
        oracle=oracle,
        stepsize=stepsize,
        z_0=z_0,
        tolerance=tolerance,
        stopping_criteria="grad_abs",
        constraints=ConstraintsL2(r_x, r_y),
        logger=logger_extragradient,
    )
    extragradient.run(max_iter=num_iter, max_time=max_time)
    z_true = logger_extragradient.argument_primal_value[-1]
    print(f"steps performed: {logger_extragradient.num_steps}")
    print(f"time elapsed: {str(datetime.timedelta(seconds=extragradient.time))}")
    print(f"grad norm: {oracle.grad(z_true).norm():.4e}")

    return z_true


def solve_with_extragradient_real_data(
    A: np.ndarray,
    b: np.ndarray,
    regcoef_x: float,
    regcoef_y,
    r_x: float,
    r_y: float,
    num_iter: int,
    max_time: int,
    tolerance: float,
) -> ArrayPair:

    L = compute_robust_linear_normed_L(A, b, r_x, r_y, regcoef_x, regcoef_y)
    z_0 = ArrayPair.zeros(A.shape[1])
    oracle = create_robust_linear_oracle(A, b, regcoef_x, regcoef_y, normed=True)
    print(f"L = {L:.3f}")
    return solve_with_extragradient(
        oracle, 1.0 / L, r_x, r_y, z_0, tolerance, num_iter, max_time
    )


def get_oracles(
    A: np.ndarray,
    b: np.ndarray,
    num_nodes: int,
    regcoef_x: float,
    regcoef_y: float,
    r_x: float,
    r_y: float,
) -> Tuple[List[ArrayPair], ArrayPair, float, float, float]:
    oracles = []
    part_sizes = np.empty(num_nodes, dtype=np.int32)
    part_sizes[:] = A.shape[0] // num_nodes
    part_sizes[: A.shape[0] - part_sizes.sum()] += 1
    start = 0
    A_grad = np.zeros([A.shape[1], A.shape[1]])
    b_grad = np.zeros([A.shape[1]])
    for part_size in part_sizes:
        A_small = A[start : start + part_size]
        b_small = b[start : start + part_size]
        oracles.append(
            create_robust_linear_oracle(
                A_small, b_small, regcoef_x, regcoef_y, normed=True
            )
        )
        start += part_size
        n_small = b_small.shape[0]
        A_grad += A_small.T @ A_small / n_small + regcoef_x
        b_grad += A_small.T @ b_small / n_small

    L, delta, mu = compute_robust_linear_normed_L_delta_mu(
        A, b, None, num_nodes, r_x, r_y, regcoef_x, regcoef_y
    )
    print("L = {:.3f}, delta = {:.3f}, mu = {:.3f}".format(L, delta, mu))
    oracle_mean = LinearCombSaddleOracle(oracles, [1 / num_nodes] * num_nodes)
    return oracles, oracle_mean, L, delta, mu, A_grad, b_grad
