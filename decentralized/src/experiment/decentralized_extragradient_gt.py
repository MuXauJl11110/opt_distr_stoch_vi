from typing import List, Optional

from src.loggers.logger import LoggerDecentralized
from src.network.network import Network
from src.oracles.base import ArrayPair, BaseSmoothSaddleOracle
from src.runners.decentralized_extragradient_runner import (
    DecentralizedExtragradientGTRunner,
)


def run_extragrad_gt(
    oracles: List[BaseSmoothSaddleOracle],
    L: float,
    mu: float,
    z_0: ArrayPair,
    z_true: ArrayPair,
    g_true: ArrayPair,
    network: Network,
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
        network=network,
        r_x=r_x,
        r_y=r_y,
    )
    extragrad_runner.compute_method_params()

    if stepsize_factor is not None:
        extragrad_runner.eta *= stepsize_factor
        print(f"Running src extragradient with stepsize_factor: {stepsize_factor}...")
    else:
        print("Running src extragradient...")

    extragrad_runner.create_method(
        z_0,
        logger=LoggerDecentralized(z_true=z_true, g_true=g_true),
    )
    extragrad_runner.method.logger.comm_per_iter = 2
    extragrad_runner.run(max_iter=comm_budget_experiment)

    return extragrad_runner
