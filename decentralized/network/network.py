import os
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import schema
import yaml
from schema import And, Or, Schema, Use
from tqdm import tqdm


class NetworkConfigManager(object):
    """
    Instrument for managing configuration for the Network class.
    """

    def __init__(
        self,
        config_path: Optional[str] = "./configs/config_network.yaml",
        general_config_path: Optional[str] = "./configs/general_config_network.yaml",
    ):
        """
        :param config_path: Relative path to the network configuration file.
        :param general_config_path: Relative path to the general configuration file.
        """
        if __name__ == "__main__":
            self.path = os.path.split(sys.argv[0])[0]
        else:
            self.path = os.path.split(__file__)[0]
        self.general_config_path = os.path.join(self.path, general_config_path)
        self.config_path = os.path.join(self.path, config_path)
        with open(self.general_config_path, "r") as cfg:
            self.general_cfg = yaml.unsafe_load(cfg)

        def make_schema_obj(d: dict):
            new_d = dict()
            for k, v in d.items():
                if isinstance(v, str):
                    if v in self.general_cfg["available_graph_types"]:
                        new_d[k] = v
                    else:
                        new_d[k] = eval(v)
                elif isinstance(v, dict):
                    new_d[k] = make_schema_obj(v)
                else:
                    raise ValueError("Unknown schema!")
            return new_d

        self.schema_types = [
            make_schema_obj({**_type, **self.general_cfg["schema"][0][0]})
            for _type in self.general_cfg["schema"][0][1]
        ]
        for _type in self.general_cfg["schema"][1:]:
            if isinstance(_type[1], str):
                setattr(self, "schema_" + _type[0], eval(_type[1]))
            elif isinstance(_type[1], dict):
                setattr(self, "schema_" + _type[0], make_schema_obj(_type[1]))
            else:
                raise ValueError(f"Forbidden value: {_type[1]}!")

        self.config_schema = Schema(
            {
                "default_epoch": self.schema_default_epoch,
                schema.Optional("epochs"): [self.schema_epoch],
            }
        )

    @property
    def config(self):
        with open(self.config_path, "r") as cfg:
            config = yaml.unsafe_load(cfg)
        self.config_schema.validate(config)
        self._config = config
        return self._config

    @config.setter
    def config(self, new_config: Dict):
        self._config = new_config
        with open(self.config_path, "w") as cfg:
            yaml.dump(new_config, cfg)


class Network(object):
    """
    Class for generating network graphs sequence.
    """

    def __init__(
        self,
        num_states: int,
        config_manager: Optional[dict] = NetworkConfigManager(),
    ):
        """
        :param num_states: Number of the network states.
        :param config_manager: Configuration manager.
        """
        if num_states <= 0:
            raise ValueError("Number of the network states should be positive!")

        self.num_states = num_states
        self.config_manager = config_manager

        self.default_epoch_cfg = self.config_manager.config["default_epoch"]

        self.current_state = 0
        self.next_epoch_cnt = 0

        self.current_epoch_start = 0
        self.current_epoch_end = self.default_epoch_cfg["duration"]
        self.eval_next_epoch()

        self.dirpath = tempfile.mkdtemp()
        self.sample_network_states()
        self.current_state = 0

    def sample_network_states(self):
        self.graphs = {}
        self.kappas = [0] * self.num_states
        self.max_lambdas = [0] * self.num_states
        self.min_lambdas = [0] * self.num_states

        while self.current_state < self.num_states:
            self.get_epoch_config()
            for state in range(self.current_epoch_start, self.current_epoch_end):
                if self.current_state == self.num_states:
                    break
                else:
                    if (
                        "duration" in self.current_epoch_cfg
                        and (state - self.current_epoch_start) in self.current_epoch_cfg["states"]
                    ):
                        L = self.get_laplacian(self.current_epoch_cfg["states"][state - self.current_epoch_start])
                    elif "interval" in self.current_epoch_cfg and state in self.current_epoch_cfg["states"]:
                        L = self.get_laplacian(self.current_epoch_cfg["states"][state])
                    else:
                        L = self.get_laplacian(self.current_epoch_cfg["default_state"])

                    self.min_lambdas[state], self.max_lambdas[state] = self.compute_lam(L)
                    self.kappas[state] = self.max_lambdas[state] / self.min_lambdas[state]

                    filename = os.path.join(self.dirpath, str(state))
                    np.save(filename, L)

                    self.current_state += 1

    def get_laplacian(self, config: dict) -> np.ndarray:
        def get_L(G):
            nodelist = None if config["fixed"] else np.random.permutation(config["args"]["n"] + config["bias"])

            L = (
                np.asarray(
                    nx.linalg.laplacianmatrix.directed_laplacian_matrix(G, nodelist=nodelist, walk_type="random")
                )
                if config["directed"] and not config["MST"]
                else nx.linalg.laplacianmatrix.laplacian_matrix(G, nodelist=nodelist).toarray()
            ).astype("float64")

            return L

        if (config["func"], config["directed"]) in self.graphs:
            G = self.graphs[(config["func"], config["directed"])]
        else:
            func = getattr(eval(config["object"]), config["func"])
            config["args"]["n"] = config["args"]["n"] - config["bias"]

            try:
                if config["type"] == "random":
                    G = func(**config["args"], directed=config["directed"])
                    is_connected = nx.is_strongly_connected(G) if config["directed"] else nx.is_connected(G)
                    L = get_L(G)
                    while not (is_connected and self.compute_lam(L)[1] < 100):
                        G = func(**config["args"], directed=config["directed"])
                        L = get_L(G)
                        is_connected = nx.is_strongly_connected(G) if config["directed"] else nx.is_connected(G)
                else:
                    G = func(**config["args"], create_using=nx.DiGraph if config["directed"] else nx.Graph)
            except nx.exception.NetworkXNotImplemented as err:
                print(f"{err}. Undirected orientation is used instead!")
                if config["type"] == "random":
                    G = func(**config["args"])
                else:
                    G = func(**config["args"])

            self.graphs[(config["func"], config["directed"], config["MST"])] = (
                G
                if not config["MST"]
                else (
                    nx.algorithms.tree.minimum_spanning_tree(G)
                    if not config["directed"]
                    else nx.algorithms.tree.minimum_spanning_tree(G.to_undirected())
                )
            )

        return get_L(G)

    def get_epoch_config(self):
        if self.current_state == self.next_epoch_start:
            self.current_epoch_start = min(self.next_epoch_start, self.num_states)
            self.current_epoch_end = min(self.next_epoch_end, self.num_states)
            self.current_epoch_cfg = self.next_epoch_cfg
            self.eval_next_epoch()
        elif self.current_state + self.default_epoch_cfg["duration"] >= self.next_epoch_start:
            self.current_epoch_start = min(self.current_state, self.num_states)
            self.current_epoch_end = min(self.next_epoch_start, self.num_states)
            self.current_epoch_cfg = self.default_epoch_cfg
        else:
            self.current_epoch_start = min(self.current_state, self.num_states)
            self.current_epoch_end = min(self.current_epoch_end + self.default_epoch_cfg["duration"], self.num_states)
            self.current_epoch_cfg = self.default_epoch_cfg

    def eval_next_epoch(self):
        if len(self.config_manager.config["epochs"]) == self.next_epoch_cnt:
            self.next_epoch_cfg = self.default_epoch_cfg
            self.next_epoch_start = sys.maxsize
            self.next_epoch_end = sys.maxsize
        else:
            self.next_epoch_cfg = self.config_manager.config["epochs"][self.next_epoch_cnt]
            self.next_epoch_start, self.next_epoch_end = self.next_epoch_cfg["interval"]
            self.next_epoch_cnt += 1

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[np.ndarray, float, float]:
        if self.current_state >= self.num_states:
            raise StopIteration
        filename = os.path.join(self.dirpath, f"{self.current_state}.npy")
        W = np.load(filename)
        lambda_min = self.min_lambdas[self.current_state]
        lambda_max = self.max_lambdas[self.current_state]
        self.current_state += 1
        return W, lambda_min, lambda_max

    def compute_lam(
        self, matrix: np.ndarray, eps_real: float = 0.00001, eps_imag: float = 0.00001
    ) -> Tuple[float, float]:
        """
        Computes positive minimal and maximum eigen values of the matrix.

        :param matrix: Laplacian matrix.
        :param eps_real: To avoid floating point errors for real part.
        :param eps_imag: To avoid floating point errors for imaginary part.
        """
        eigs = np.linalg.eigvals(matrix)
        positive_eigs = [eig.real for eig in eigs if eig.real > eps_real and eig.imag <= eps_imag]

        return min(positive_eigs), max(positive_eigs)
