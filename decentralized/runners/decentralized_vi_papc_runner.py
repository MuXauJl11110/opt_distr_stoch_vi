from typing import List

import numpy as np
from decentralized.loggers import LoggerDecentralized
from decentralized.methods import ConstraintsL2, DecentralizedVIPAPC
from decentralized.network.network import Network
from decentralized.oracles import ArrayPair, BaseSmoothSaddleOracle


class DecentralizedVIPAPCRunner:
    def __init__(
        self,
        oracles: List[BaseSmoothSaddleOracle],
        L: float,
        mu: float,
        network: Network,
        r_x: float,
        r_y: float,
    ):
        self.oracles = oracles
        self.L = L
        self.mu = mu
        self.network = network
        self.constraints = ConstraintsL2(r_x, r_y)
        self._params_computed = False

    def compute_method_params(self):
        self._lam = self.network.peek()[1]
        self.chi = 1 / self._lam
        self.beta = self.mu / (self.L**2)
        self.theta = min(1 / (2 * self.beta), self.L * np.sqrt(self.chi) / 3)
        self.eta = 1 / (3 * self.L * np.sqrt(self.chi))
        self.alpha = 1 - min(
            1 / (1 + 3 * self.L * np.sqrt(self.chi) / self.mu), 1 / (2 * self.chi)
        )
        self._params_computed = True

    def create_method(
        self, z_0: List[ArrayPair], y_0: List[ArrayPair], logger: LoggerDecentralized
    ):
        if self._params_computed == False:
            raise ValueError("Call compute_method_params first")

        self.method = DecentralizedVIPAPC(
            oracles=self.oracles,
            z_0=z_0,
            y_0=y_0,
            eta=self.eta,
            theta=self.theta,
            alpha=self.alpha,
            beta=self.beta,
            network=self.network,
            logger=logger,
            constraints=self.constraints,
        )

    def run(self, max_iter, max_time=None):
        self.method.run(max_iter, max_time)
