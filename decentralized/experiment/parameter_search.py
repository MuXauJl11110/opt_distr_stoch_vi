import os
import pickle
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from decentralized.experiment.decentralized_extragradient_con import run_extragrad_con
from decentralized.experiment.decentralized_extragradient_gt import run_extragrad_gt
from decentralized.experiment.decentralized_vi_papc import run_vi_papc
from decentralized.experiment.plotting import plot_algorithms, preplot_algorithms
from decentralized.experiment.saddle_sliding import run_sliding
from decentralized.experiment.saving import save_algorithms
from decentralized.oracles.base import ArrayPair
from decentralized.utils.generate_matrices import metropolis_weights
from decentralized.utils.utils import get_oracles
from IPython.display import clear_output
from tqdm import tqdm


def run_parameter_search(
    methods: List[str],
    topologies: List[str],
    datasets: List[str],
    adj_mat: Dict[str, np.ndarray],
    gos_mat: Dict[str, np.ndarray],
    num_nodes: int,
    labels: List[str],
    regcoef_x: float,
    regcoef_y: float,
    r_x: float,
    r_y: float,
    eps: float,
    comm_budget_experiment: int,
    experiment_type: str,
    get_A_b: Callable[[str], Tuple[np.ndarray, np.ndarray]],
    stepsize_factors: List[float],
    logs_path: Optional[str] = "./logs",
):
    extragrad_gt_runner = lambda stepsize_factor: run_extragrad_gt(
        oracles=oracles,
        L=L,
        mu=mu,
        z_0=z_0,
        z_true=z_true,
        g_true=g_true,
        mix_mat=mix_mat,
        r_x=r_x,
        r_y=r_y,
        comm_budget_experiment=comm_budget_experiment,
        stepsize_factor=stepsize_factor,
    )
    extragrad_con_runner = lambda stepsize_factor: run_extragrad_con(
        oracles=oracles,
        L=L,
        mu=mu,
        z_0=z_0,
        z_true=z_true,
        g_true=g_true,
        mix_mat=mix_mat,
        r_x=r_x,
        r_y=r_y,
        eps=eps,
        comm_budget_experiment=comm_budget_experiment,
        stepsize_factor=stepsize_factor,
    )
    sliding_runner = lambda stepsize_factor: run_sliding(
        oracles=oracles,
        L=L,
        delta=delta,
        mu=mu,
        z_0=z_0,
        z_true=z_true,
        g_true=g_true,
        mix_mat=mix_mat,
        r_x=r_x,
        r_y=r_y,
        eps=eps,
        comm_budget_experiment=comm_budget_experiment,
        stepsize_factor=stepsize_factor,
    )
    vi_papc_runner = lambda stepsize_factor: run_vi_papc(
        num_nodes=num_nodes,
        oracles=oracles,
        L=L,
        mu=mu,
        z_0=z_0,
        z_true=z_true,
        g_true=g_true,
        gos_mat=W,
        r_x=r_x,
        r_y=r_y,
        comm_budget_experiment=comm_budget_experiment,
        stepsize_factor=stepsize_factor,
    )
    method_to_runner = {
        "extragrad": extragrad_gt_runner,
        "extragrad_con": extragrad_con_runner,
        "sliding": sliding_runner,
        "vi_papc": vi_papc_runner,
    }

    for dataset_name in tqdm(datasets, desc="Datasets..."):
        A, b = get_A_b(dataset_name)
        z_0 = ArrayPair.zeros(A.shape[1])
        oracles, oracle_mean, L, delta, mu, A_grad, b_grad = get_oracles(
            A,
            b,
            num_nodes,
            regcoef_x,
            regcoef_y,
            r_x,
            r_y,
        )
        for topology in tqdm(topologies, desc="Topogies..."):
            print(f"Dataset: {dataset_name}")
            print(f"Topology: {topology}")

            mix_mat = metropolis_weights(adj_mat[topology])
            W = gos_mat[topology]

            path = os.path.join(
                logs_path, experiment_type, topology, f"{num_nodes}_{dataset_name}"
            )
            with open(os.path.join(path, "z_true"), "rb") as f:
                z_true = pickle.load(f)

            g_true = ArrayPair(
                np.zeros((num_nodes, z_true.x.shape[0])),
                np.zeros((num_nodes, z_true.y.shape[0])),
            )

            for method, label in tqdm(zip(methods, labels), desc="Methods..."):
                runner = method_to_runner[method]
                clear_output(wait=True)
                print(f"Method: {label}")
                methods_PS = []
                methods_names_PS = []
                labels_PS = []

                for stepsize_factor in tqdm(
                    stepsize_factors, desc="Stepsize factors..."
                ):
                    print(f"Stepsize factor: {stepsize_factor}")
                    methods_PS.append(runner(stepsize_factor))
                    methods_names_PS.append(f"{method}_{stepsize_factor}.pkl")
                    labels_PS.append(f"{label} {stepsize_factor}")

                preplot_algorithms(
                    topology=topology,
                    num_nodes=num_nodes,
                    data=dataset_name,
                    labels=labels_PS,
                    methods=methods_PS,
                    dist_to_opt_type="argument",
                )
                plt.close()

                save_algorithms(
                    topology=topology,
                    num_nodes=num_nodes,
                    data=dataset_name,
                    methods=methods_PS,
                    method_names=methods_names_PS,
                    z_true=z_true,
                    experiment_type=experiment_type,
                )

                plot_algorithms(
                    topology=topology,
                    num_nodes=num_nodes,
                    data=dataset_name,
                    labels=labels_PS,
                    method_names=methods_names_PS,
                    comm_budget_experiment=comm_budget_experiment,
                    save_folder=f"{dataset_name}",
                    save_to=method,
                )

                methods_PS.clear()
                methods_names_PS.clear()
                labels_PS.clear()
