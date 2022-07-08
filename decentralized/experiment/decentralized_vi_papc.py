import random
from typing import List, Optional

import numpy as np
from loggers import LoggerDecentralized
from oracles import ArrayPair
from runners import DecentralizedVIPAPCRunner


def generate_y_list(num_nodes: int, z_0: ArrayPair):
    y_list = []
    z_x, z_y = z_0.tuple()
    for i in range(num_nodes // 2):
        x = np.random.rand(z_x.shape[0])
        y = np.random.rand(z_y.shape[0])
        y_list.append(ArrayPair(x, y))
        y_list.append(ArrayPair(-x, -y))
    if num_nodes % 2 == 1:
        y_list.append(ArrayPair(np.zeros_like(z_x), np.zeros_like(z_y)))

    random.shuffle(y_list)
    return y_list


def run_vi_papc(
    num_nodes: int,
    oracles: List[ArrayPair],
    L: float,
    mu: float,
    z_0: ArrayPair,
    z_true: ArrayPair,
    g_true: ArrayPair,
    gos_mat: np.ndarray,
    r_x: float,
    r_y: float,
    comm_budget_experiment: int,
    stepsize_factor: Optional[float] = None,
    random_y_0: Optional[bool] = False,
):
    vi_papc_runner = DecentralizedVIPAPCRunner(
        oracles=oracles,
        L=L,
        mu=mu,
        gos_mat=gos_mat,
        r_x=r_x,
        r_y=r_y,
        logger=LoggerDecentralized(z_true=z_true, g_true=g_true),
    )
    z_0_list = [z_0] * num_nodes
    if random_y_0:
        y_0_list = generate_y_list(num_nodes)
    else:
        y_0_list = [z_0] * num_nodes

    vi_papc_runner.compute_method_params()

    if stepsize_factor is not None:
        vi_papc_runner.eta *= stepsize_factor
        print(f"Running decentralized VI PAPC with stepsize_factor: {stepsize_factor}...")
    else:
        print("Running decentralized VI PAPC...")

    vi_papc_runner.create_method(z_0_list, y_0_list)
    vi_papc_runner.logger.comm_per_iter = 1
    vi_papc_runner.run(max_iter=comm_budget_experiment // vi_papc_runner.logger.comm_per_iter)

    return vi_papc_runner
