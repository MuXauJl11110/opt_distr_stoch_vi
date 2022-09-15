from typing import List

import numpy as np
from decentralized.methods import ConstraintsL2, DecentralizedVIADOM
from decentralized.oracles import ArrayPair, BaseSmoothSaddleOracle
from decentralized.runners.base import BaseRunner
from decentralized.utils import compute_lam


class DecentralizedVIADOMRunner(BaseRunner):
    def __init__(
        self,
        oracles: List[BaseSmoothSaddleOracle],
        b: int,
        L: float,
        L_avg: float,
        mu: float,
        r_x: float,
        r_y: float,
        x_0: List[ArrayPair],
        y_0: List[ArrayPair],
        z_0: List[ArrayPair],
    ):
        self.oracles = oracles
        self.b = b
        self.L = L
        self.L_avg = L_avg
        self.mu = mu
        self.constraints = ConstraintsL2(r_x, r_y)
        self.method = DecentralizedVIADOM(
            oracles=self.oracles,
            x_0=x_0,
            y_0=y_0,
            z_0=z_0,
            eta_x=np.inf,
            eta_y=np.inf,
            eta_z=np.inf,
            theta=np.inf,
            alpha=np.inf,
            gamma=np.inf,
            omega=np.inf,
            tau=np.inf,
            nu=np.inf,
            beta=np.inf,
            logger=None,
        )

    def compute_method_params(self):
        self.method._lam = compute_lam(self.method.gos_mat)[0]
        self.method.chi = 1 / self.method._lam
        self.method.p = 1 / 16
        self.method.omega = self.method.p
        self.method.theta = 1 / 2
        self.method.gamma = min(
            self.mu / (16 * (self.L ** 2)), np.inf  # , self.b * self.omega / (24 * (self.L_avg ** 2) * self.eta_z)
        )
        self.method.beta = 5 * self.method.gamma
        self.method.nu = self.mu / 4
        self.method.alpha = 1 / 2
        self.method.tau = min(
            self.mu / (32 * self.L * self.method.chi), self.mu * np.sqrt(self.b * self.method.p) / (32 * self.L_avg)
        )
        self.method.eta_x = min(
            1 / (900 * self.method.chi * self.method.gamma),
            self.method.nu / (36 * self.method.tau * (self.method.chi ** 2)),
        )
        self.method.eta_y = min(1 / (4 * self.method.gamma), self.method.nu / (8 * self.method.tau))
        self.method.eta_z = min(
            1 / (8 * self.L * self.method.chi),
            1 / (32 * self.method.eta_y),
            np.sqrt(self.method.alpha * self.b * self.method.omega) / (8 * self.L_avg),
        )

    def run(self, max_iter, max_time=None):
        self.method.run(max_iter, max_time)
