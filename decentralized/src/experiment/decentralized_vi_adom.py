from typing import Dict, List, Optional

from src.loggers.logger import LoggerDecentralized
from src.network.network import Network
from src.oracles.base import ArrayPair, BaseSmoothSaddleOracle
from src.runners.decentralized_vi_adom_runner import DecentralizedVIADOMRunner


def run_vi_adom(
    num_nodes: int,
    oracles: List[BaseSmoothSaddleOracle],
    b: int,
    L: float,
    L_avg: float,
    mu: float,
    x_0: ArrayPair,
    y_0: ArrayPair,
    z_0: ArrayPair,
    z_true: ArrayPair,
    g_true: ArrayPair,
    network: Network,
    r_x: float,
    r_y: float,
    comm_budget_experiment: int,
    stepsize_factors: Optional[Dict[str, float]] = None,
):
    vi_adom_runner = DecentralizedVIADOMRunner(
        oracles=oracles,
        b=b,
        L=L,
        L_avg=L_avg,
        mu=mu,
        network=network,
        r_x=r_x,
        r_y=r_y,
    )
    x_0_list = [x_0] * num_nodes
    y_0_list = [y_0] * num_nodes
    z_0_list = [z_0] * num_nodes

    vi_adom_runner.compute_method_params()

    if stepsize_factors is not None:
        output_str = ""
        for parameter, stepsize in stepsize_factors.items():
            attr = getattr(vi_adom_runner, parameter)
            # if attr == "eta_z":
            #     vi_adom_runner.eta_z = stepsize / L
            # else:
            attr *= stepsize
            output_str += f"{parameter}={stepsize}"
        print(f"Running src VI ADOM with {output_str} parameters...")
    else:
        print("Running src VI ADOM...")

    vi_adom_runner.create_method(
        x_0_list,
        z_0_list,
        y_0_list,
        logger=LoggerDecentralized(z_true=z_true, g_true=g_true),
    )
    vi_adom_runner.method.logger.comm_per_iter = 1
    vi_adom_runner.run(max_iter=comm_budget_experiment)

    return vi_adom_runner
