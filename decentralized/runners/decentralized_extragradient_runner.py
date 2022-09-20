from typing import List

import numpy as np
from decentralized.loggers import Logger
from decentralized.methods import ConstraintsL2, DecentralizedExtragradientGT
from decentralized.network import Network
from decentralized.network.network import Network
from decentralized.oracles import BaseSmoothSaddleOracle
from decentralized.utils import compute_lam_2


class DecentralizedExtragradientGTRunner:
    def __init__(
        self,
        oracles: List[BaseSmoothSaddleOracle],
        L: float,
        mu: float,
        gamma: float,
        network: Network,
        r_x: float,
        r_y: float,
    ):
        self.oracles = oracles
        self.L = L
        self.mu = mu
        self.gamma = gamma
        self.network = network
        self.constraints = ConstraintsL2(r_x, r_y)
        self._params_computed = False

    def compute_method_params(self):
        self._lam = compute_lam_2(self.network.peek()[0])
        eta = max(
            (1 - self._lam) ** 2 * self.gamma / (500 * self.L),
            (1 - self._lam) ** (4 / 3) * self.mu ** (1 / 3) / (40 * self.L ** (4 / 3)),
        )
        self.eta = min(eta, (1 - self._lam) ** 2 / (22 * self.L))
        self._params_computed = True

    def create_method(self, z_0, logger: Logger):
        if self._params_computed == False:
            raise ValueError("Call compute_method_params first")

        self.method = DecentralizedExtragradientGT(
            oracles=self.oracles,
            stepsize=self.eta,
            network=self.network,
            z_0=z_0,
            logger=logger,
            constraints=self.constraints,
        )

    def run(self, max_iter, max_time=None):
        self.method.run(max_iter, max_time)
