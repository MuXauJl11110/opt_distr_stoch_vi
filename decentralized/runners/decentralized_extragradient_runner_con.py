from typing import List

import numpy as np
from decentralized import network
from decentralized.loggers import Logger
from decentralized.methods import ConstraintsL2, DecentralizedExtragradientCon
from decentralized.network import Network
from decentralized.oracles import BaseSmoothSaddleOracle
from decentralized.utils import compute_lam_2


class DecentralizedExtragradientConRunner:
    def __init__(
        self,
        oracles: List[BaseSmoothSaddleOracle],
        L: float,
        mu: float,
        network: Network,
        r_x: float,
        r_y: float,
        eps: float,
    ):
        self.oracles = oracles
        self.L = L
        self.mu = mu
        self.network = network
        self.constraints = ConstraintsL2(r_x, r_y)
        self.eps = eps
        self._params_computed = False

    def compute_method_params(self):
        self._lam = compute_lam_2(self.network.peek()[0])
        self.gamma = 1 / (4 * self.L)
        self.gossip_step = (1 - np.sqrt(1 - self._lam**2)) / (
            1 + np.sqrt(1 - self._lam**2)
        )
        eps_0 = self.eps * self.mu * self.gamma * (1 + self.gamma * self.L) ** 2
        self.con_iters = int(np.sqrt(1 / (1 - self._lam)) * np.log(1 / eps_0))
        # rough estimate on con_iters (lower than actual)
        self._params_computed = True

    def create_method(self, z_0, logger: Logger):
        if self._params_computed == False:
            raise ValueError("Call compute_method_params first")

        self.method = DecentralizedExtragradientCon(
            oracles=self.oracles,
            stepsize=self.gamma,
            con_iters=self.con_iters,
            network=self.network,
            gossip_step=self.gossip_step,
            z_0=z_0,
            logger=logger,
            constraints=self.constraints,
        )

    def run(self, max_iter, max_time=None):
        self.method.run(max_iter, max_time)
