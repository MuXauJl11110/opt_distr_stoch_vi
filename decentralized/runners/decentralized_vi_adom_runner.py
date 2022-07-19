from typing import List

import numpy as np
from decentralized.loggers import LoggerDecentralized
from decentralized.methods import ConstraintsL2, DecentralizedVIADOM
from decentralized.oracles import ArrayPair, BaseSmoothSaddleOracle
from decentralized.utils import compute_lam


class DecentralizedVIADOMRunner(object):
    def __init__(
        self,
        oracles: List[BaseSmoothSaddleOracle],
        b: int,
        n: int,
        L: float,
        L_avg: float,
        mu: float,
        gos_mat: np.ndarray,
        r_x: float,
        r_y: float,
        logger: LoggerDecentralized,
    ):
        self.oracles = oracles
        self.b = b
        self.n = n
        self.L = L
        self.L_avg = L_avg
        self.mu = mu
        self.gos_mat = gos_mat
        self.constraints = ConstraintsL2(r_x, r_y)
        self.logger = logger
        self._params_computed = False

    def compute_method_params(self):
        self._lam = compute_lam(self.gos_mat)[0]
        self.chi = 1 / self._lam
        self.p = 1 / 16
        self.omega = self.p
        self.theta = 1 / 2
        self.gamma = np.min(
            self.mu / (16 * (self.L ** 2))  # , self.b * self.omega / (24 * (self.L_avg ** 2) * self.eta_z)
        )
        self.beta = 5 * self.gamma
        self.nu = self.mu / 4
        self.alpha = 1 / 2
        self.tau = np.min(self.mu / (32 * self.L * self.chi), self.mu * np.sqrt(self.b * self.p) / (32 * self.L_avg))
        self.eta_x = np.min(1 / (900 * self.chi * self.gamma), self.nu / (36 * self.tau * (self.chi ** 2)))
        self.eta_y = np.min(1 / (4 * self.gamma), self.nu / (8 * self.tau))
        self.eta_z = np.min(
            1 / (8 * self.L * self.chi),
            1 / (32 * self.eta_y),
            np.sqrt(self.alpha * self.b * self.omega) / (8 * self.L_avg),
        )
        self._params_computed = True

    def create_method(self, x_0: List[ArrayPair], y_0: List[ArrayPair], w_0: List[ArrayPair]):
        if self._params_computed == False:
            raise ValueError("Call compute_method_params first")

        self.method = DecentralizedVIADOM(
            oracles=self.oracles,
            x_0=x_0,
            y_0=y_0,
            w_0=w_0,
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
            b=self.b,
            p=self.p,
            gos_mat=self.gos_mat,
            n=self.n,
            logger=self.logger,
            constraints=self.constraints,
        )

    def run(self, max_iter, max_time=None):
        self.method.run(max_iter, max_time)
