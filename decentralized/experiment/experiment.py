from typing import Callable, Dict, List, Tuple

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
from decentralized.utils.utils import get_oracles, solve_with_extragradient_real_data
from IPython.display import clear_output
from tqdm import tqdm


def run_experiment(
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
    num_iter_solution: int,
    max_time_solution: int,
    tolerance_solution: float,
    comm_budget_experiment: int,
    experiment_type: str,
    get_A_b: Callable[[str], Tuple[np.ndarray, np.ndarray]],
):
    extragrad_gt_runner = lambda: run_extragrad_gt(
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
    )
    extragrad_con_runner = lambda: run_extragrad_con(
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
    )
    sliding_runner = lambda: run_sliding(
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
    )
    vi_papc_runner = lambda: run_vi_papc(
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
        # For exact solution uncomment next lines
        # x = np.linalg.solve(A_grad, b_grad)
        # z_true = ArrayPair(x, np.zeros(A.shape[1]))
        z_true = solve_with_extragradient_real_data(
            A=A,
            b=b,
            regcoef_x=regcoef_x,
            regcoef_y=regcoef_y,
            r_x=r_x,
            r_y=r_y,
            num_iter=num_iter_solution,
            max_time=max_time_solution,
            tolerance=tolerance_solution,
        )
        g_true = ArrayPair(
            np.zeros((num_nodes, z_true.x.shape[0])),
            np.zeros((num_nodes, z_true.y.shape[0])),
        )

        for topology in tqdm(topologies, desc="Topogies..."):
            clear_output(wait=True)
            print(f"Dataset: {dataset_name}")
            print(f"Topology: {topology}")
            methods_exp = []
            method_names_exp = []
            labels_exp = []
            mix_mat = metropolis_weights(adj_mat[topology])
            W = gos_mat[topology]

            for method, label in tqdm(zip(methods, labels), desc="Methods..."):
                clear_output(wait=True)
                print(f"Method: {label}")

                runner = method_to_runner[method]
                methods_exp.append(runner())
                method_names_exp.append(method)
                labels_exp.append(label)

            preplot_algorithms(
                topology=topology,
                num_nodes=num_nodes,
                data=dataset_name,
                labels=labels,
                methods=methods_exp,
                dist_to_opt_type="argument",
            )
            plt.close()

            save_algorithms(
                topology=topology,
                num_nodes=num_nodes,
                data=dataset_name,
                methods=methods_exp,
                method_names=method_names_exp,
                z_true=z_true,
                experiment_type=experiment_type,
            )

            plot_algorithms(
                topology=topology,
                num_nodes=num_nodes,
                data=dataset_name,
                labels=labels,
                method_names=method_names_exp,
                comm_budget_experiment=comm_budget_experiment,
                save_to=f"{num_nodes}_{dataset_name}",
            )

            methods_exp.clear()
            method_names_exp.clear()
            labels_exp.clear()
