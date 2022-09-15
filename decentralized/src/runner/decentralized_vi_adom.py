from typing import List, Optional

import numpy as np
from src.logger.decentralized import LoggerDecentralized
from src.method import ConstraintsL2, DecentralizedVIADOM
from src.oracles import ArrayPair, BaseSmoothSaddleOracle
from src.runner.base import BaseRunner
from src.utils import compute_lam


class DecentralizedVIADOMRunner(BaseRunner):
    def __init__(
        self,
        b: int,
        L: float,
        L_avg: float,
        mu: float,
        r_x: float,
        r_y: float,
    ):
        self.b = b
        self.L = L
        self.L_avg = L_avg
        self.mu = mu
        self.constraints = ConstraintsL2(r_x, r_y)

    def create_method(
        self,
        oracles: List[BaseSmoothSaddleOracle],
        gos_mat: np.ndarray,
        x_0: List[ArrayPair],
        y_0: List[ArrayPair],
        z_0: List[ArrayPair],
        z_true: Optional[ArrayPair] = None,
        g_true: Optional[ArrayPair] = None,
    ):
        self.oracles = oracles
        self.gos_mat = gos_mat

        self.compute_method_params()
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
            gos_mat=gos_mat,
        )
        self.logger = LoggerDecentralized(self.method, z_true, g_true)

    def compute_chi(self):
        self._lam = compute_lam(self.gos_mat)[0]
        self.chi = 1 / self._lam

    def compute_omega(self):
        self.p = 1 / 16
        self.omega = self.p

    def compute_theta(self):
        self.theta = 1 / 2

    def compute_gamma(self):
        self.gamma = min(
            self.mu / (16 * (self.L**2)),
            np.inf,  # , self.b * self.omega / (24 * (self.L_avg ** 2) * self.eta_z)
        )

    def compute_beta(self):
        self.beta = 5 * self.gamma

    def compute_nu(self):
        self.nu = self.mu / 4

    def compute_alpha(self):
        self.alpha = 1 / 2

    def compute_tau(self, precompute_chi: Optional[bool] = True):
        if precompute_chi:
            self.compute_chi()
        self.tau = min(
            self.mu / (32 * self.L * self.chi),
            self.mu * np.sqrt(self.b * self.p) / (32 * self.L_avg),
        )

    def compute_eta_x(self, precompute_chi: Optional[bool] = True):
        if precompute_chi:
            self.compute_chi()
        self.eta_x = min(
            1 / (900 * self.chi * self.gamma),
            self.nu / (36 * self.tau * (self.chi**2)),
        )

    def compute_eta_y(self):
        self.eta_y = min(1 / (4 * self.gamma), self.nu / (8 * self.tau))

    def compute_eta_z(self, precompute_chi: Optional[bool] = True):
        if precompute_chi:
            self.compute_chi()
        self.eta_z = min(
            1 / (8 * self.L * self.chi),
            1 / (32 * self.eta_y),
            np.sqrt(self.alpha * self.b * self.omega) / (8 * self.L_avg),
        )

    def compute_method_params(self):
        self.compute_omega()
        self.compute_theta()
        self.compute_gamma()
        self.compute_beta()
        self.compute_nu()
        self.compute_alpha()
        self.compute_tau()
        self.compute_eta_x(precompute_chi=False)
        self.compute_eta_y()
        self.compute_eta_z(precompute_chi=False)

    def update_method_params(
        self,
        compute_omega: Optional[float] = None,
        compute_theta: Optional[float] = None,
        compute_gamma: Optional[float] = None,
        compute_beta: Optional[float] = None,
        compute_nu: Optional[float] = None,
        compute_alpha: Optional[float] = None,
        compute_tau: Optional[float] = None,
        compute_eta_x: Optional[float] = None,
        compute_eta_y: Optional[float] = None,
        compute_eta_z: Optional[float] = None,
    ):
        return super().update_method_params(locals())

    def update_matrix(self, gos_mat: np.ndarray):
        return super().update_matrix(**{"gos_mat": gos_mat})
