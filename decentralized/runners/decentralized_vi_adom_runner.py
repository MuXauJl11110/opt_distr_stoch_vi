from typing import List

import numpy as np
from decentralized.loggers import LoggerDecentralized
from decentralized.methods import ConstraintsL2, DecentralizedVIADOM
from decentralized.network.network import Network
from decentralized.oracles import ArrayPair, BaseSmoothSaddleOracle


class DecentralizedVIADOMRunner:
    def __init__(
        self,
        oracles: List[BaseSmoothSaddleOracle],
        b: int,
        L: float,
        L_avg: float,
        mu: float,
        network: Network,
        r_x: float,
        r_y: float,
    ):
        self.oracles = oracles
        self.b = b
        self.L = L
        self.L_avg = L_avg
        self.mu = mu
        self.network = network
        self.constraints = ConstraintsL2(r_x, r_y)
        self._params_computed = False

    def compute_method_params(self):
        self._lam = self.network.peek()[1]
        self.chi = 1 / self._lam
        self.p = 1 / 16
        self.omega = self.p
        self.theta = 1 / 2
        self.gamma = min(
            self.mu / (16 * (self.L**2)),
            np.inf,  # , self.b * self.omega / (24 * (self.L_avg ** 2) * self.eta_z)
        )
        self.beta = 5 * self.gamma
        self.nu = self.mu / 4
        self.alpha = 1 / 2
        self.tau = min(
            self.mu / (32 * self.L * self.chi),
            self.mu * np.sqrt(self.b * self.p) / (32 * self.L_avg),
        )
        self.eta_x = min(
            1 / (900 * self.chi * self.gamma),
            self.nu / (36 * self.tau * (self.chi**2)),
        )
        self.eta_y = min(1 / (4 * self.gamma), self.nu / (8 * self.tau))
        self.eta_z = min(
            1 / (8 * self.L * self.chi),
            1 / (32 * self.eta_y),
            np.sqrt(self.alpha * self.b * self.omega) / (8 * self.L_avg),
        )
        self._params_computed = True

    def create_method(
        self,
        x_0: List[ArrayPair],
        y_0: List[ArrayPair],
        z_0: List[ArrayPair],
        logger: LoggerDecentralized,
    ):
        if self._params_computed == False:
            raise ValueError("Call compute_method_params first")

        self.method = DecentralizedVIADOM(
            oracles=self.oracles,
            x_0=x_0,
            y_0=y_0,
            z_0=z_0,
            eta_x=self.eta_x,
            eta_y=self.eta_y,
            eta_z=self.eta_z,
            theta=self.theta,
            alpha=self.alpha,
            gamma=self.gamma,
            omega=self.omega,
            tau=self.tau,
            nu=self.nu,
            beta=self.beta,
            network=self.network,
            logger=logger,
            constraints=self.constraints,
        )

    def run(self, max_iter, max_time=None):
        self.method.run(max_iter, max_time)
