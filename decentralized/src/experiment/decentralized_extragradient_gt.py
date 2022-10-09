from typing import Dict, List, Optional

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
    stepsize_factors: Optional[Dict[str, float]] = None,
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

    if stepsize_factors is not None:
        output_str = ""
        for parameter, stepsize in stepsize_factors.items():
            attr = getattr(extragrad_runner, parameter)
            attr *= stepsize
            output_str += f"{parameter}={stepsize}"
        print(f"Running src extragradient with {output_str} parameters...")
    else:
        print("Running src extragradient...")

    extragrad_runner.create_method(
        z_0,
        logger=LoggerDecentralized(z_true=z_true, g_true=g_true),
    )
    extragrad_runner.method.logger.comm_per_iter = 2
    extragrad_runner.run(max_iter=comm_budget_experiment)

    return extragrad_runner
