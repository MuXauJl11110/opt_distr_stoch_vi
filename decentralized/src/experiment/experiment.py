import os
import pickle
from typing import Dict

import ipywidgets as widgets
import numpy as np
from IPython.display import clear_output
from src.config_managers import (
    DatasetConfigManager,
    NetworkConfigManager,
    RunnerConfigManager,
    SchedulerConfigManager,
)
from src.dataset import DatasetLayout
from src.layout import Button, Container, Layout, NumericText, String
from src.network import Network, NetworkLayout
from src.oracles.base import ArrayPair
from src.runner.layout import RunnerLayout
from src.scheduler import Scheduler, SchedulerLayout
from src.utils import gen_matrices_decentralized, get_oracles
from src.utils.utils import call_with_redundant
from tqdm import tqdm


class Experiment(object):
    """
    Instrument for managing experiment.
    """

    def __init__(self):
        self.runner_cm = RunnerConfigManager(
            config_path="src/experiment/configs/runner.pickle"
        )
        self.network_cm = NetworkConfigManager(
            config_path="src/experiment/configs/network.yaml"
        )
        self.scheduler_cm = SchedulerConfigManager(
            config_path="src/experiment/configs/scheduler.yaml"
        )

        self.runner_layout = RunnerLayout()
        self.network_layout = NetworkLayout(self.network_cm)
        self.scheduler_layout = SchedulerLayout(
            self.network_layout.general, self.scheduler_cm
        )
        self.dataset_layout = DatasetLayout()

        self.layout = self.initialize_layout()
        self.output = widgets.Output()

    def initialize_layout(self):
        def run_experiment(b: widgets.widgets.Button):
            self.run()

        run_button = Button(
            "Run experiment!", button_style="danger", on_click=run_experiment
        )

        return Container(
            "VBox",
            [
                String("Text", "Experiment name:", "John Doe", "Type experiment name"),
                String("Text", "Save folder:", "logs", "Type save folder"),
                Layout(
                    "Tab",
                    ["Runner", "Network", "Scheduler"],
                    [
                        Container(
                            "VBox",
                            [
                                Container(
                                    "VBox", [self.runner_layout.layout], "Methods:"
                                ),
                                Container(
                                    "VBox", [self.dataset_layout.layout], "Datasets:"
                                ),
                                Container(
                                    "VBox",
                                    [
                                        NumericText(
                                            "BoundedFloatText",
                                            "$r_x$",
                                            5.0,
                                            0,
                                            100,
                                            0.1,
                                            r_label="r_x",
                                        ),
                                        NumericText(
                                            "BoundedFloatText",
                                            "$r_y$",
                                            0,
                                            0,
                                            100,
                                            0.1,
                                            r_label="r_y",
                                        ),
                                        NumericText(
                                            "BoundedFloatText",
                                            "$regcoef_x$",
                                            2.0,
                                            0,
                                            100,
                                            0.1,
                                            r_label="regcoef_x",
                                        ),
                                        NumericText(
                                            "BoundedFloatText",
                                            "$regcoef_y$",
                                            2.0,
                                            0,
                                            100,
                                            0.1,
                                            r_label="regcoef_y",
                                        ),
                                    ],
                                    "General:",
                                ),
                            ],
                        ),
                        self.network_layout.layout,
                        self.scheduler_layout.layout,
                    ],
                ),
                run_button,
            ],
        )

    def run(self):
        clear_output(wait=True)

        for dataset_name, dataset_cfg in tqdm(self.dataset_layout, desc="Datasets"):
            A, b = call_with_redundant(
                dataset_cfg | {"num_matrices": dataset_cfg["num_nodes"]},
                gen_matrices_decentralized,
            )
            oracles, oracle_mean, L, delta, mu, A_grad, b_grad = get_oracles(
                A,
                b,
                dataset_cfg["num_nodes"],
                **self.layout.value["Runner"]["General:"],
            )
            optional_cfg = {
                "oracles": oracles,
                "L": L,
                "L_avg": L,
                "delta": delta,
                "eps": 1e-10,
                "mu": mu,
                "gamma": mu,
            }

            x = np.linalg.lstsq(A_grad, b_grad, rcond=None)[0]
            z_true = ArrayPair(x, np.zeros(A.shape[1]))
            g_true = ArrayPair(
                np.zeros((dataset_cfg["num_nodes"], z_true.x.shape[0])),
                np.zeros((dataset_cfg["num_nodes"], z_true.y.shape[0])),
            )

            for states in tqdm(
                self.layout.value["network"]["States number:"], desc="States number"
            ):
                self.network_layout.update_config(states)
                self.scheduler_layout.update_config(states)
                for nodes in tqdm(
                    self.layout.value["network"]["Nodes number:"], desc="Nodes number"
                ):
                    for method_name, method_runner, initial_guess in self.runner_layout(
                        d=dataset_cfg["d"],
                        general_cfg=self.layout.value["Runner"]["General:"],
                        optional_cfg=optional_cfg,
                    ):
                        network = Network(
                            int(states),
                            int(nodes),
                            self.runner_layout.runner_сfg[method_name][1],
                            config_manager=self.network_cm,
                        )
                        W_0, _, _ = network.peek()

                        method_runner.create_method(
                            oracles=oracles,
                            z_true=z_true,
                            g_true=g_true,
                            **{
                                **initial_guess,
                                **{self.runner_layout.runner_сfg[method_name][1]: W_0},
                            },
                        )

                        scheduler = Scheduler(
                            method_runner=method_runner,
                            network=network,
                            config_manager=self.scheduler_cm,
                        )

                        for _ in scheduler:
                            pass

                        loggers = {}
                        networks = {
                            "kappas": network.kappas,
                            "min_lambdas": network.min_lambdas,
                            "max_lambdas": network.max_lambdas,
                        }
                        loggers[
                            "argument"
                        ] = method_runner.logger.argument_primal_distance_to_opt
                        loggers[
                            "gradient"
                        ] = method_runner.logger.gradient_primal_distance_to_opt
                        loggers[
                            "consensus"
                        ] = method_runner.logger.argument_primal_distance_to_consensus

                        self.save_result(
                            states,
                            nodes,
                            method_name,
                            dataset_name,
                            loggers,
                            networks,
                        )

    def save_result(
        self,
        states: str,
        nodes: str,
        method_name: str,
        dataset_name: str,
        logger: Dict,
        network: Dict,
    ):
        experiment_name = self.layout.value["Experiment name:"]

        save_folder = os.path.join(
            self.layout.value["Save folder:"],
            experiment_name,
            dataset_name,
            states,
            nodes,
        )
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        with open(os.path.join(save_folder, f"logger_{method_name}.pkl"), "wb") as f:
            pickle.dump(logger, f)
        with open(os.path.join(save_folder, f"network_{method_name}.pkl"), "wb") as f:
            pickle.dump(network, f)
