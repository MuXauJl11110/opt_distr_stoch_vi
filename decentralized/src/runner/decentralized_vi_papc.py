from typing import List, Optional

import numpy as np
from src.logger.decentralized import LoggerDecentralized
from src.method import ConstraintsL2, DecentralizedVIPAPC
from src.oracles import ArrayPair, BaseSmoothSaddleOracle
from src.runner.base import BaseRunner
from src.utils import compute_lam


class DecentralizedVIPAPCRunner(BaseRunner):
    def __init__(
        self,
        L: float,
        mu: float,
        r_x: float,
        r_y: float,
    ):
        self.L = L
        self.mu = mu
        self.constraints = ConstraintsL2(r_x, r_y)

    def create_method(
        self,
        oracles: List[BaseSmoothSaddleOracle],
        gos_mat: np.ndarray,
        z_0: List[ArrayPair],
        y_0: List[ArrayPair],
        z_true: Optional[ArrayPair] = None,
        g_true: Optional[ArrayPair] = None,
    ):
        self.oracles = oracles
        self.gos_mat = gos_mat

        self.compute_method_params()
        self.method = DecentralizedVIPAPC(
            oracles=self.oracles,
            z_0=z_0,
            y_0=y_0,
            eta=self.eta,
            theta=self.theta,
            alpha=self.alpha,
            beta=self.beta,
            gos_mat=self.gos_mat,
            constraints=self.constraints,
        )
        self.logger = LoggerDecentralized(self.method, z_true, g_true)

    def compute_chi(self):
        self._lam = compute_lam(self.gos_mat)[0]
        self.chi = 1 / self._lam

    def compute_eta(self, precompute_chi=True):
        if precompute_chi:
            self.compute_chi()
        self.eta = 1 / (3 * self.L * np.sqrt(self.chi))

    def compute_theta(self, precompute_chi=True):
        if precompute_chi:
            self.compute_chi()
        self.theta = min(1 / (2 * self.beta), self.L * np.sqrt(self.chi) / 3)

    def compute_alpha(self, precompute_chi=True):
        if precompute_chi:
            self.compute_chi()
        self.alpha = 1 - min(
            1 / (1 + 3 * self.L * np.sqrt(self.chi) / self.mu), 1 / (2 * self.chi)
        )

    def compute_beta(self):
        self.beta = self.mu / (self.L**2)

    def compute_method_params(self):
        self.compute_beta()
        self.compute_theta()
        self.compute_eta(precompute_chi=False)
        self.compute_alpha(precompute_chi=False)

    def update_method_params(
        self,
        eta: Optional[float] = None,
        theta: Optional[float] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ):
        super().update_method_params(locals())

    def update_matrix(self, gos_mat: np.ndarray):
        return super().update_matrix(**{"gos_mat": gos_mat})
