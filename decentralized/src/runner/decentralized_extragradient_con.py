from typing import List, Optional

import numpy as np
from src.logger.decentralized import LoggerDecentralized
from src.method import ConstraintsL2, DecentralizedExtragradientCon
from src.oracles.base import ArrayPair, BaseSmoothSaddleOracle
from src.runner.base import BaseRunner
from src.utils import compute_lam_2


class DecentralizedExtragradientConRunner(BaseRunner):
    def __init__(
        self,
        L: float,
        mu: float,
        r_x: float,
        r_y: float,
        eps: float,
    ):
        self.L = L
        self.mu = mu
        self.constraints = ConstraintsL2(r_x, r_y)
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
        self.method = DecentralizedExtragradientCon(
            oracles=self.oracles,
            stepsize=self.stepsize,
            con_iters=self.con_iters,
            mix_mat=self.mix_mat,
            gossip_step=self.gossip_step,
            z_0=z_0,
            constraints=self.constraints,
        )
        self.logger = LoggerDecentralized(self.method, z_true, g_true)

    def compute_lam(self):
        self._lam = compute_lam_2(self.mix_mat)

    def compute_stepsize(self):
        self.stepsize = 1 / (4 * self.L)

    def compute_con_iters(self, precompute_lam: Optional[bool] = True):
        if precompute_lam:
            self.compute_lam()
        eps_0 = self.eps * self.mu * self.stepsize * (1 + self.stepsize * self.L) ** 2
        self.con_iters = int(np.sqrt(1 / abs(1 - self._lam)) * np.log(1 / eps_0))

    def compute_gossip_step(self, precompute_lam: Optional[bool] = True):
        if precompute_lam:
            self.compute_lam()
        self.gossip_step = (1 - np.sqrt(abs(1 - self._lam**2))) / (
            1 + np.sqrt(abs(1 - self._lam**2))
        )
        # rough estimate on con_iters (lower than actual)

    def compute_method_params(self):
        self.compute_stepsize()
        self.compute_gossip_step()
        self.compute_con_iters(precompute_lam=False)

    def update_method_params(
        self,
        stepsize: Optional[float] = None,
        con_iter: Optional[float] = None,
        gossip_step: Optional[float] = None,
    ):
        return super().update_method_params(locals())

    def update_matrix(self, mix_mat: np.ndarray):
        super().update_matrix(**{"mix_mat": mix_mat})
