from typing import List, Optional

from src.logger.centralized import LoggerCentralized
from src.method.base import BaseSaddleMethod
from src.method.constraints import ConstraintsL2
from src.oracles.base import ArrayPair, BaseSmoothSaddleOracle


class BaseRunner(object):
    def __init__(
        self,
        oracles: List[BaseSmoothSaddleOracle],
        L: float,
        mu: float,
        r_x: float,
        r_y: float,
        eps: float,
        z_0: ArrayPair,
        z_true: Optional[ArrayPair] = None,
        g_true: Optional[ArrayPair] = None,
    ):
        self.oracles = oracles
        self.L = L
        self.mu = mu
        self.constraints = ConstraintsL2(r_x, r_y)
        self.eps = eps

        self.method = BaseSaddleMethod(
            oracle=oracles,
            z_0=z_0,
        )
        self.logger = LoggerCentralized(self.method, z_true, g_true)

    def compute_method_params(self):
        raise NotImplementedError("compute_method_params() is not implemented!")

    def update_method_params(self, **kwargs):
        """
        If parameter is None updates method's parameter using "compute_`parameter`" runner's method.
        Else sets specified parameter.
        """
        for key, value in kwargs:
            if value is None:
                func = hasattr(self, f"compute_{key}", None)
                if callable(func):
                    func()
                    setattr(self.method, key, getattr(self, key))
                else:
                    raise TypeError("compute_{key} is not callable!")
            else:
                setattr(self.method, key, value)

    def update_matrix(self, **kwargs):
        for key, value in kwargs.items():
            old_matrix = getattr(self.method, key, None)
            if old_matrix is None:
                raise AttributeError("method must has {key} attribute!")
            if old_matrix.shape != value.shape:
                raise ValueError(
                    f"wrong matrix size {value.shape}! Must be {old_matrix.shape}"
                )
            setattr(self.method, key, value)
            setattr(self, key, value)
