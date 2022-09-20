from typing import List, Optional

import numpy as np
from decentralized.loggers.logger import LoggerDecentralized
from decentralized.oracles.base import ArrayPair, BaseSmoothSaddleOracle
from decentralized.runners.decentralized_extragradient_runner import (
    DecentralizedExtragradientGTRunner,
)


def run_extragrad_gt(
    oracles: List[BaseSmoothSaddleOracle],
    L: float,
    mu: float,
    z_0: ArrayPair,
    z_true: ArrayPair,
    g_true: ArrayPair,
    mix_mat: np.ndarray,
    r_x: float,
    r_y: float,
    comm_budget_experiment: int,
    stepsize_factor: Optional[float] = None,
) -> DecentralizedExtragradientGTRunner:
    extragrad_runner = DecentralizedExtragradientGTRunner(
        oracles=oracles,
        L=L,
        mu=mu,
        gamma=mu,
        mix_mat=mix_mat,
        r_x=r_x,
        r_y=r_y,
        logger=LoggerDecentralized(z_true=z_true, g_true=g_true),
    )
    extragrad_runner.compute_method_params()

    if stepsize_factor is not None:
        extragrad_runner.eta *= stepsize_factor
        print(
            f"Running decentralized extragradient with stepsize_factor: {stepsize_factor}..."
        )
    else:
        print("Running decentralized extragradient...")

    extragrad_runner.create_method(z_0)
    extragrad_runner.logger.comm_per_iter = 2
    extragrad_runner.run(
        max_iter=comm_budget_experiment // extragrad_runner.logger.comm_per_iter
    )

    return extragrad_runner
