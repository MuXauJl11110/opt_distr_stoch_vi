from typing import List

from decentralized.loggers import Logger
from decentralized.methods import ConstraintsL2
from decentralized.oracles import BaseSmoothSaddleOracle
from decentralized.oracles.base import ArrayPair


class BaseRunner(object):
    def __init__(
        self,
        oracles: List[BaseSmoothSaddleOracle],
        L: float,
        mu: float,
        r_x: float,
        r_y: float,
        eps: float,
        logger: Logger,
    ):
        self.oracles = oracles
        self.L = L
        self.mu = mu
        self.constraints = ConstraintsL2(r_x, r_y)
        self.eps = eps
        self.logger = logger
        self._params_computed = False

    def compute_method_params(self):
        raise NotImplementedError("compute_method_params() is not implemented!")

    def run(self, max_iter: int, max_time=None):
        raise NotImplementedError("run() is not implemented!")
