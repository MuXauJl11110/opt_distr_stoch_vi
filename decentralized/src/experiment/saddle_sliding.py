from typing import Dict, List, Optional

from src.loggers.logger import LoggerDecentralized
from src.network.network import Network
from src.oracles.base import ArrayPair, BaseSmoothSaddleOracle
from src.runners.decentralized_sliding_runner import (
    DecentralizedSaddleSlidingRunner,
    sliding_comm_per_iter,
)


def run_sliding(
    oracles: List[BaseSmoothSaddleOracle],
    L: float,
    delta: float,
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
) -> DecentralizedSaddleSlidingRunner:
    sliding_runner = DecentralizedSaddleSlidingRunner(
        oracles=oracles,
        L=L,
        mu=mu,
        delta=delta,
        network=network,
        r_x=r_x,
        r_y=r_y,
        eps=eps,
    )
    sliding_runner.compute_method_params()

    if stepsize_factors is not None:
        output_str = ""
        for parameter, stepsize in stepsize_factors.items():
            attr = getattr(sliding_runner, parameter)
            attr *= stepsize
            output_str += f"{parameter}={stepsize}"
        print(f"Running src sliding with {output_str} parameters...")
    else:
        print("Running src sliding...")

    sliding_runner.create_method(
        z_0,
        logger=LoggerDecentralized(z_true=z_true, g_true=g_true),
    )

    print(
        "H_0 = {}, H_1 = {}, T_subproblem = {}".format(
            sliding_runner.con_iters_grad,
            sliding_runner.con_iters_pt,
            sliding_runner.method.inner_iterations,
        )
    )
    sliding_runner.method.logger.comm_per_iter = sliding_comm_per_iter(sliding_runner.method)
    sliding_runner.run(max_iter=comm_budget_experiment)

    return sliding_runner
