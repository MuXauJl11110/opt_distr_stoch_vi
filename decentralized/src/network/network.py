import os
import sys
import tempfile
from collections import deque
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from definitions import ROOT_DIR
from src.network.config_manager import NetworkConfigManager
from src.utils.generate_matrices import metropolis_weights


class Network(object):
    """
    Class for generating network graphs sequence.
    """

    matrix_types = {"gos_mat", "mix_mat"}

    def __init__(
        self,
        num_states: int,
        num_nodes: int,
        matrix_type: str,
        config_manager: Optional[NetworkConfigManager] = NetworkConfigManager(),
        dirpath: Optional[str] = None,
        use_generated: Optional[bool] = False,
        save_to: Optional[str] = None,
    ):
        """
        :param num_states: Number of the network states.
        :param num_nodes: Number of the network nodes.
        :param matrix_type: Matrix type ["gos_mat" - gossip, "mix_mat" - "mixing"]
        :param config_manager: Configuration manager.
        :param dirpath: Relative folder path for saving matrices if None saving into temporary folder.
        :param use_generated: If true matrices from dirpath are used.
        :param save_to: Relative path, where plot should be saved.
        """
        if num_states <= 0:
            raise ValueError("Number of the network states should be positive!")

        self.num_states = num_states
        self.num_nodes = num_nodes
        self.config_manager = config_manager
        self.general_config = config_manager.general_cfg
        self.save_to = save_to
        if matrix_type in Network.matrix_types:
            self.matrix_type = matrix_type
        else:
            raise ValueError(
                f"Unknown matrix_type {matrix_type}! Available matrix types: {Network.matrix_types}"
            )

        self.default_epoch_cfg = self.config_manager.config["default_epoch"]

        self.current_state = 0
        self.next_epoch_cnt = 0
        self.plotting = False

        self.current_epoch_start = 0
        self.current_epoch_end = self.default_epoch_cfg["duration"]
        self.eval_next_epoch()

        if dirpath is not None or (dirpath is None and use_generated is True):
            if isinstance(dirpath, str):
                self.dirpath = os.path.join(ROOT_DIR, dirpath)
                if not os.path.exists(self.dirpath):
                    os.system(f"mkdir -p {self.dirpath}")
            else:
                raise ValueError(f"dirpath must be str type, but has {type(dirpath)}!")
        else:
            self.dirpath = tempfile.mkdtemp()
        if not use_generated:
            self.sample_network_states()
        self.current_state = 0

        self.peeked = deque()

    def sample_network_states(self):
        self.graphs = {}
        self.graphs_for_plotting = []
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
                        and (state - self.current_epoch_start)
                        in self.current_epoch_cfg["states"]
                    ):
                        L = self.get_laplacian(
                            self.current_epoch_cfg["states"][
                                state - self.current_epoch_start
                            ]
                        )
                    elif (
                        "interval" in self.current_epoch_cfg
                        and state in self.current_epoch_cfg["states"]
                    ):
                        L_gos, L_mix = self.get_laplacian(
                            self.current_epoch_cfg["states"][state]
                        )
                    else:
                        L_gos, L_mix = self.get_laplacian(
                            self.current_epoch_cfg["default_state"]
                        )

                    self.min_lambdas[state], self.max_lambdas[state] = self.compute_lam(
                        L_gos
                    )
                    self.kappas[state] = (
                        self.max_lambdas[state] / self.min_lambdas[state]
                    )

                    filename = os.path.join(self.dirpath, f"{state}_gos_mat")
                    np.save(filename, L_gos / self.max_lambdas[state])
                    filename = os.path.join(self.dirpath, f"{state}_mix_mat")
                    np.save(filename, L_mix)

                    self.current_state += 1

    def get_laplacian(self, config: dict) -> np.ndarray:
        topology_config = self.general_config["topology"][config["topology"]]
        num_nodes = self.num_nodes - topology_config["bias"]

        def get_L(G):
            self.nodelist = (
                None if config["fixed"] else np.random.permutation(num_nodes)
            )

            L_gos = (
                np.asarray(
                    nx.linalg.laplacianmatrix.directed_laplacian_matrix(
                        G, nodelist=self.nodelist, walk_type="random"
                    )
                )
                if config["directed"] and not config["MST"]
                else nx.linalg.laplacianmatrix.laplacian_matrix(
                    G, nodelist=self.nodelist
                ).toarray()
            ).astype("float64")
            L_mix = metropolis_weights(
                nx.adjacency_matrix(G, nodelist=self.nodelist).toarray()
            )

            return L_gos, L_mix

        if (config["topology"], config["directed"], config["MST"]) in self.graphs:
            G = self.graphs[(config["topology"], config["directed"], config["MST"])]
        else:
            func = getattr(eval(topology_config["object"]), topology_config["func"])

            try:
                if topology_config["type"] == "random":
                    G = func(**config["args"], n=num_nodes, directed=config["directed"])
                    is_connected = (
                        nx.is_strongly_connected(G)
                        if config["directed"]
                        else nx.is_connected(G)
                    )
                    L_gos, L_mix = get_L(G)
                    while not (is_connected and self.compute_lam(L_gos)[1] < 100):
                        G = func(
                            **config["args"], n=num_nodes, directed=config["directed"]
                        )
                        L_gos, L_mix = get_L(G)
                        is_connected = (
                            nx.is_strongly_connected(G)
                            if config["directed"]
                            else nx.is_connected(G)
                        )
                else:
                    G = func(
                        **config["args"],
                        n=num_nodes,
                        create_using=nx.DiGraph if config["directed"] else nx.Graph,
                    )
            except nx.exception.NetworkXNotImplemented as err:
                print(f"{err}. Undirected orientation is used instead!")
                if topology_config["type"] == "random":
                    G = func(**config["args"], n=num_nodes)
                else:
                    G = func(**config["args"], n=num_nodes)

            self.graphs[(config["topology"], config["directed"], config["MST"])] = (
                G
                if not config["MST"]
                else (
                    nx.algorithms.tree.minimum_spanning_tree(G)
                    if not config["directed"]
                    else nx.algorithms.tree.minimum_spanning_tree(G.to_undirected())
                )
            )

        L_gos, L_mix = get_L(G)

        if config["plot"] is True:
            if self.save_to is None:
                raise ValueError(
                    "Configuration has network state for saving, but parameter 'save_to' isn't specified!"
                )
            self.graphs_for_plotting.append((G, self.nodelist, self.current_state))

            self.plotting = True

        # print(config["func"], config["args"]["p"])
        return L_gos, L_mix

    def get_epoch_config(self):
        if self.current_state == self.next_epoch_start:
            self.current_epoch_start = min(self.next_epoch_start, self.num_states)
            self.current_epoch_end = min(self.next_epoch_end, self.num_states)
            self.current_epoch_cfg = self.next_epoch_cfg
            self.eval_next_epoch()
        elif (
            self.current_state + self.default_epoch_cfg["duration"]
            >= self.next_epoch_start
        ):
            self.current_epoch_start = min(self.current_state, self.num_states)
            self.current_epoch_end = min(self.next_epoch_start, self.num_states)
            self.current_epoch_cfg = self.default_epoch_cfg
        else:
            self.current_epoch_start = min(self.current_state, self.num_states)
            self.current_epoch_end = min(
                self.current_epoch_end + self.default_epoch_cfg["duration"],
                self.num_states,
            )
            self.current_epoch_cfg = self.default_epoch_cfg

    def eval_next_epoch(self):
        if len(self.config_manager.config["epochs"]) == self.next_epoch_cnt:
            self.next_epoch_cfg = self.default_epoch_cfg
            self.next_epoch_start = sys.maxsize
            self.next_epoch_end = sys.maxsize
        else:
            self.next_epoch_cfg = self.config_manager.config["epochs"][
                self.next_epoch_cnt
            ]
            self.next_epoch_start, self.next_epoch_end = self.next_epoch_cfg["interval"]
            self.next_epoch_cnt += 1

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[np.ndarray, float, float]:
        if self.peeked:
            return self.peeked.popleft()

        if self.current_state >= self.num_states:
            if self.plotting:
                self.plot()
            raise StopIteration
        filename = os.path.join(
            self.dirpath, f"{self.current_state}_{self.matrix_type}.npy"
        )
        W = np.load(filename)
        lambda_min = self.min_lambdas[self.current_state]
        lambda_max = self.max_lambdas[self.current_state]
        self.current_state += 1
        return W, lambda_min, lambda_max

    def peek(self, ahead: Optional[int] = 0) -> Tuple[np.ndarray, float, float]:
        while len(self.peeked) <= ahead:
            self.peeked.append(self.__next__())
        return self.peeked[ahead]

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
        positive_eigs = [
            eig.real for eig in eigs if eig.real > eps_real and eig.imag <= eps_imag
        ]

        return min(positive_eigs), max(positive_eigs)

    def plot(self):
        fig, axes = plt.subplots(len(self.graphs_for_plotting), 1, figsize=(4, 10))

        for i, (G, nodelist, iteration) in enumerate(self.graphs_for_plotting):
            nx.draw(
                G,
                nodelist=nodelist,
                with_labels=True,
                font_weight="bold",
                node_color="#a8f0b9",
                ax=axes[i],
            )
            axes[i].set_title(
                f"Iterations {1 + 5 * iteration} - {5 * (iteration + 20)}:", fontsize=26
            )

        plt.tight_layout()
        plt.savefig(os.path.join(ROOT_DIR, self.save_to))
        plt.close(fig)
