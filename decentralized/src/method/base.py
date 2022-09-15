import numpy as np
from src.oracles.base import BaseSmoothSaddleOracle


class BaseSaddleMethod(object):
    """
    Base class for algorithms.


    :param oracle: Oracle corresponding to the objective function.
    :param z_0: Initial guess.
    """

    def __init__(
        self,
        oracle: BaseSmoothSaddleOracle,
        z_0: np.ndarray,
    ):
        self.oracle = oracle
        self.z = z_0.copy()

    def step(self):
        raise NotImplementedError("step() not implemented!")
