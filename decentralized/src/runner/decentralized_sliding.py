from typing import List, Optional

import numpy as np
from src.logger.decentralized import LoggerDecentralized
from src.method import DecentralizedSaddleSliding
from src.oracles import ArrayPair, BaseSmoothSaddleOracle
from src.runner.base import BaseRunner
from src.utils import compute_lam_2


class DecentralizedSaddleSlidingRunner(BaseRunner):
    def __init__(
        self,
        L: float,
        mu: float,
        delta: float,
        r_x: float,
        r_y: float,
        eps: float,
    ):
        self.L = L
        self.mu = mu
        self.delta = delta
        self.r_x = r_x
        self.r_y = r_y
        self.eps = eps

    def create_method(
        self,
        oracles: List[BaseSmoothSaddleOracle],
        mix_mat: np.ndarray,
        z_0: ArrayPair,
        z_true: Optional[ArrayPair] = None,
        g_true: Optional[ArrayPair] = None,
    ):
        self.oracles = oracles
        self.mix_mat = mix_mat

        self.compute_method_params()
        self.method = DecentralizedSaddleSliding(
            oracles=self.oracles,
            stepsize_outer=self.stepsize_outer,
            stepsize_inner=self.stepsize_inner,
            inner_iterations=self.inner_iterations,
            con_iters_grad=self.con_iters_grad,
            con_iters_pt=self.con_iters_pt,
            mix_mat=self.mix_mat,
            gossip_step=self.gossip_step,
            z_0=z_0,
            constraints=None,
        )
        self.logger = LoggerDecentralized(self.method, z_true, g_true)

    def precompute_con_iters_params(self):
        self._omega = 2 * np.sqrt(self.r_x**2 + self.r_y**2)
        self._g = 0.0  # upper bound on gradient at optimum; let it be 0 for now
        self._rho = abs(1 - self._lam)
        self._num_nodes = len(self.oracles)

    def compute_stepsize_outer(self):
        self.stepsize_outer = min(1.0 / (7 * self.delta), 1 / (12 * self.mu))

    def compute_stepsize_inner(self):
        self.stepsize_inner = 0.5 / (self.stepsize_outer * self.L + 1)

    def compute_inner_iterations(self):
        self.e = 0.5 / (
            2
            + 12 * self.stepsize_outer**2 * self.delta**2
            + 4 / (self.stepsize_outer * self.mu)
            + (8 * self.stepsize_outer * self.delta**2) / self.mu
        )
        self.inner_iterations = int(
            (1 + self.stepsize_outer * self.L) * np.log10(1 / self.e)
        )

    def compute_con_iters_grad(self, precompute: Optional[bool] = True):
        if precompute:
            self.precompute_con_iters_params()

        print(self._rho)
        print(self.eps, self.stepsize_outer, self.mu)
        self.con_iters_grad = int(
            1
            / np.sqrt(self._rho)
            * np.log(
                (self.stepsize_outer * 2 + self.stepsize_outer / self.mu)
                * self._num_nodes
                * (self.L * self._omega + self._g) ** 2
                / (self.eps * self.stepsize_outer * self.mu)
            )
        )

    def compute_con_iters_pt(self, precompute: Optional[bool] = True):
        if precompute:
            self.precompute_con_iters_params()

        self.con_iters_pt = int(
            1
            / np.sqrt(self._rho)
            * np.log(
                (
                    1
                    + self.stepsize_outer**2 * self.L**2
                    + self.stepsize_outer * self.L**2 / self.mu
                )
                * self._num_nodes
                * self._omega**2
                / (self.eps * self.stepsize_outer * self.mu)
            )
        )

    def compute_gossip_step(self):
        self._lam = compute_lam_2(self.mix_mat)
        self.gossip_step = (1 - np.sqrt(1 - self._lam**2)) / (
            1 + np.sqrt(1 - self._lam**2)
        )

    def compute_method_params(self):
        self.compute_stepsize_outer()  # outer step-size
        self.compute_stepsize_inner()
        self.compute_inner_iterations()
        self.compute_gossip_step()
        self.compute_con_iters_grad()
        self.compute_con_iters_pt(precompute=False)

    def update_method_params(
        self,
        stepsize_outer: Optional[float] = None,
        stepsize_inner: Optional[float] = None,
        inner_iterations: Optional[int] = None,
        con_iters_grad: Optional[int] = None,
        con_iters_pt: Optional[int] = None,
        gossip_step: Optional[float] = None,
    ):
        super().update_method_params(locals())

    def update_matrix(self, mix_mat: np.ndarray):
        return super().update_matrix(**{"mix_mat": mix_mat})


def sliding_comm_per_iter(runner: DecentralizedSaddleSlidingRunner):
    return 2 * (runner.con_iters_grad + runner.con_iters_pt)
