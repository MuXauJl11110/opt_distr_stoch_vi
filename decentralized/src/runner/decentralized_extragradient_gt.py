from typing import List, Optional

import numpy as np
from src.logger.decentralized import LoggerDecentralized
from src.method import ConstraintsL2, DecentralizedExtragradientGT
from src.oracles import BaseSmoothSaddleOracle
from src.oracles.base import ArrayPair
from src.runner.base import BaseRunner
from src.utils import compute_lam_2


class DecentralizedExtragradientGTRunner(BaseRunner):
    def __init__(
        self,
        L: float,
        mu: float,
        gamma: float,
        r_x: float,
        r_y: float,
    ):
        self.L = L
        self.mu = mu
        self.gamma = gamma
        self.constraints = ConstraintsL2(r_x, r_y)

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
        self.method = DecentralizedExtragradientGT(
            oracles=self.oracles,
            stepsize=self.stepsize,
            mix_mat=self.mix_mat,
            z_0=z_0,
            constraints=self.constraints,
        )
        self.logger = LoggerDecentralized(self.method, z_true, g_true)

    def compute_stepsize(self):
        _lam = compute_lam_2(self.mix_mat)
        eta = max(
            (1 - _lam) ** 2 * self.gamma / (500 * self.L),
            (1 - _lam) ** (4 / 3) * self.mu ** (1 / 3) / (40 * self.L ** (4 / 3)),
        )
        self.stepsize = min(eta, (1 - _lam) ** 2 / (22 * self.L))

    def compute_method_params(self):
        self.compute_stepsize()

    def update_method_params(self, eta: Optional[float] = None):
        super().update_method_params(locals())

    def update_matrix(self, mix_mat: np.ndarray):
        super().update_matrix(**{"mix_mat": mix_mat})
