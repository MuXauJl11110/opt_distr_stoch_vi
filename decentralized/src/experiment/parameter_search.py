import os
import pickle
import traceback
from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from definitions import ROOT_DIR
from IPython.display import clear_output
from src.experiment.decentralized_extragradient_con import run_extragrad_con
from src.experiment.decentralized_extragradient_gt import run_extragrad_gt
from src.experiment.decentralized_vi_adom import run_vi_adom
from src.experiment.decentralized_vi_papc import run_vi_papc
from src.experiment.plotting import plot_algorithms, preplot_algorithms
from src.experiment.saddle_sliding import run_sliding
from src.experiment.saving import save_algorithms
from src.network.config_manager import NetworkConfigManager
from src.network.network import Network
from src.oracles.base import ArrayPair
from src.utils.utils import get_oracles, solve_with_extragradient_real_data
from tqdm import tqdm


def parameter_search(
    methods: List[str],
    topologies: List[str],
    datasets: List[str],
    num_nodes: int,
    num_states: int,
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
    stepsize_factors: List[float],
    get_A_b: Callable[[str], Tuple[np.ndarray, np.ndarray]],
    logs_path: Optional[str] = "logs",
):
    extragrad_gt_runner = lambda stepsize_factor: run_extragrad_gt(
        oracles=oracles,
        L=L,
        mu=mu,
        z_0=z_0,
        z_true=z_true,
        g_true=g_true,
        network=network,
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
        network=network,
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
        network=network,
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
        network=network,
        r_x=r_x,
        r_y=r_y,
        comm_budget_experiment=comm_budget_experiment,
        stepsize_factor=stepsize_factor,
    )
    vi_adom_runner = lambda stepsize_factor: run_vi_adom(
        num_nodes=num_nodes,
        oracles=oracles,
        b=batch_size,
        L=L,
        L_avg=L_avg,
        mu=mu,
        x_0=x_0,
        y_0=y_0,
        z_0=z_0,
        z_true=z_true,
        g_true=g_true,
        network=network,
        r_x=r_x,
        r_y=r_y,
        comm_budget_experiment=comm_budget_experiment,
        stepsize_factor=stepsize_factor,
    )
    method_to_runner = {
        "extragrad": extragrad_gt_runner,
        "extragrad_con": extragrad_con_runner,
        "sliding": sliding_runner,
        # "vi_papc": vi_papc_runner,
        "vi_adom": vi_adom_runner,
    }
    method_to_mat = {
        "extragrad": "mix_mat",
        "extragrad_con": "mix_mat",
        "sliding": "mix_mat",
        # "vi_papc": "gos_mat",
        "vi_adom": "gos_mat",
    }
    precomputed_z_true = False
    for dataset_name in tqdm(datasets, desc="Datasets..."):
        A, b = get_A_b(dataset_name)
        x_0 = ArrayPair.zeros(A.shape[1])
        y_0 = ArrayPair.zeros(A.shape[1])
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
        L_avg = L
        batch_size = len(oracles)
        # For exact solution uncomment next lines
        # x = np.linalg.solve(A_grad, b_grad)
        # z_true = ArrayPair(x, np.zeros(A.shape[1]))

        for topology in tqdm(topologies, desc="Topogies..."):
            clear_output(wait=True)
            print(f"Dataset: {dataset_name}")
            print(f"Topology: {topology}")
            network = Network(
                num_states,
                num_nodes,
                "mix_mat",
                NetworkConfigManager(f"src/experiment/configs/{topology}.yaml"),
            )

            path = os.path.join(
                logs_path,
                experiment_type,
                topology,
                f"{num_nodes}_{dataset_name}",
            )
            if os.path.exists(path):
                try:
                    with open(os.path.join(path, "z_true"), "rb") as file:
                        z_true = pickle.load(file)
                except ModuleNotFoundError:
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

            elif precomputed_z_true is False:
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
                precomputed_z_true = True
            g_true = ArrayPair(
                np.zeros((num_nodes, z_true.x.shape[0])),
                np.zeros((num_nodes, z_true.y.shape[0])),
            )

            for method, label in tqdm(zip(methods, labels), desc="Methods..."):
                runner = method_to_runner[method]
                clear_output(wait=True)
                print(f"Method: {label}")
                runners_PS = []
                methods_names_PS = []
                labels_PS = []

                for stepsize_factor in tqdm(
                    stepsize_factors, desc="Stepsize factors..."
                ):
                    print(f"Stepsize factor: {stepsize_factor}")
                    network.current_state = 0
                    network.matrix_type = method_to_mat[method]

                    runners_PS.append(runner(stepsize_factor))
                    methods_names_PS.append(method + f"_{stepsize_factor}.pkl")
                    labels_PS.append(label + f" {stepsize_factor}")

                preplot_algorithms(
                    topology=topology,
                    num_nodes=num_nodes,
                    data=dataset_name,
                    labels=labels_PS,
                    runners=runners_PS,
                    dist_to_opt_type="argument",
                )
                plt.close()

                save_algorithms(
                    topology=topology,
                    num_nodes=num_nodes,
                    data=dataset_name,
                    runners=runners_PS,
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
                    save_folder=f"{dataset_name}",
                    save_to=method,
                )

                runners_PS.clear()
                methods_names_PS.clear()
                labels_PS.clear()

        precomputed_z_true = False
