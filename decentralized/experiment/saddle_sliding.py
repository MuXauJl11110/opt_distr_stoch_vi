from typing import List, Optional

import numpy as np
from decentralized.loggers.logger import LoggerDecentralized
from decentralized.oracles.base import ArrayPair
from decentralized.runners.decentralized_sliding_runner import (
    DecentralizedSaddleSlidingRunner,
    sliding_comm_per_iter,
)


def run_sliding(
    oracles: List[ArrayPair],
    L: float,
    delta: float,
    mu: float,
    z_0: ArrayPair,
    z_true: ArrayPair,
    g_true: ArrayPair,
    mix_mat: np.ndarray,
    r_x: float,
    r_y: float,
    eps: float,
    comm_budget_experiment: int,
    stepsize_factor: Optional[float] = None,
) -> DecentralizedSaddleSlidingRunner:
    sliding_runner = DecentralizedSaddleSlidingRunner(
        oracles=oracles,
        L=L,
        mu=mu,
        delta=delta,
        mix_mat=mix_mat,
        r_x=r_x,
        r_y=r_y,
        eps=eps,
        logger=LoggerDecentralized(z_true=z_true, g_true=g_true),
    )
    sliding_runner.compute_method_params()

    if stepsize_factor is not None:
        sliding_runner.gamma *= stepsize_factor
        print(f"Running decentralized sliding with stepsize_factor: {stepsize_factor}...")
    else:
        print("Running decentralized sliding...")

    sliding_runner.create_method(z_0)

    print(
        "H_0 = {}, H_1 = {}, T_subproblem = {}".format(
            sliding_runner.con_iters_grad,
            sliding_runner.con_iters_pt,
            sliding_runner.method.inner_iterations,
        )
    )
    sliding_runner.logger.comm_per_iter = sliding_comm_per_iter(sliding_runner.method)
    sliding_runner.run(max_iter=comm_budget_experiment // sliding_runner.logger.comm_per_iter)

    return sliding_runner
