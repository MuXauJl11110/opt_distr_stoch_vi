from typing import Dict, List, Optional

from src.loggers.logger import LoggerDecentralized
from src.network.network import Network
from src.oracles.base import ArrayPair, BaseSmoothSaddleOracle
from src.runners.decentralized_extragradient_runner_con import (
    DecentralizedExtragradientConRunner,
)


def run_extragrad_con(
    oracles: List[BaseSmoothSaddleOracle],
    L: float,
    mu: float,
    z_0: ArrayPair,
    z_true: ArrayPair,
    g_true: ArrayPair,
    network: Network,
    r_x: float,
    r_y: float,
    eps: float,
    comm_budget_experiment: int,
    stepsize_factors: Optional[Dict[str, float]] = None,
) -> DecentralizedExtragradientConRunner:
    extragrad_con_runner = DecentralizedExtragradientConRunner(
        oracles=oracles,
        L=L,
        mu=mu,
        network=network,
        r_x=r_x,
        r_y=r_y,
        eps=eps,
    )
    extragrad_con_runner.compute_method_params()

    if stepsize_factors is not None:
        output_str = ""
        for parameter, stepsize in stepsize_factors.items():
            attr = getattr(extragrad_con_runner, parameter)
            attr *= stepsize
            output_str += f"{parameter}={stepsize}"
        print(f"Running src extragradient-con with {output_str} parameters...")
    else:
        print("Running src extragradient-con...")

    extragrad_con_runner.create_method(
        z_0,
        logger=LoggerDecentralized(z_true=z_true, g_true=g_true),
    )
    print("T_consensus = {}".format(extragrad_con_runner.method.con_iters))
    extragrad_con_runner.method.logger.comm_per_iter = 2 * extragrad_con_runner.method.con_iters
    extragrad_con_runner.run(max_iter=comm_budget_experiment)

    return extragrad_con_runner
