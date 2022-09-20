from typing import List, Optional

import numpy as np
from decentralized.loggers.logger import LoggerDecentralized
from decentralized.oracles.base import ArrayPair, BaseSmoothSaddleOracle
from decentralized.runners.decentralized_extragradient_runner_con import (
    DecentralizedExtragradientConRunner,
)


def run_extragrad_con(
    oracles: List[BaseSmoothSaddleOracle],
    L: float,
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
) -> DecentralizedExtragradientConRunner:
    extragrad_con_runner = DecentralizedExtragradientConRunner(
        oracles=oracles,
        L=L,
        mu=mu,
        mix_mat=mix_mat,
        r_x=r_x,
        r_y=r_y,
        eps=eps,
        logger=LoggerDecentralized(z_true=z_true, g_true=g_true),
    )
    extragrad_con_runner.compute_method_params()

    if stepsize_factor is not None:
        extragrad_con_runner.gamma *= stepsize_factor
        print(
            f"Running decentralized extragradient-con with stepsize_factor: {stepsize_factor}..."
        )
    else:
        print("Running decentralized extragradient-con...")

    extragrad_con_runner.create_method(z_0)
    print("T_consensus = {}".format(extragrad_con_runner.method.con_iters))
    extragrad_con_runner.logger.comm_per_iter = (
        2 * extragrad_con_runner.method.con_iters
    )
    extragrad_con_runner.run(
        max_iter=comm_budget_experiment // extragrad_con_runner.logger.comm_per_iter
    )

    return extragrad_con_runner
