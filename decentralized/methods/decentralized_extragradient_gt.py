from typing import List, Optional

import numpy as np
from decentralized.loggers.logger import Logger
from decentralized.methods.base import BaseSaddleMethod
from decentralized.methods.constraints import ConstraintsL2
from decentralized.network.network import Network
from decentralized.oracles.base import (
    ArrayPair,
    BaseSmoothSaddleOracle,
    LinearCombSaddleOracle,
)


class DecentralizedExtragradientGT(BaseSaddleMethod):
    """
    Decentralized Extragradient with gradient tracking.
    (https://ieeexplore.ieee.org/document/9304470).

    Parameters
    ----------
    oracles: List[BaseSmoothSaddleOracle]
        List of oracles corresponding to network nodes.

    stepsize: float
        Stepsize of Extragradient method.

    network: Network
        Network consisting of mixing matrices.

    z_0: ArrayPair
        Initial guess (similar at each node).

    logger: Optional[Logger]
        Stores the history of the method during its iterations.

    constraints: Optional[ConstraintsL2]
        L2 constraints on problem variables.
    """

    def __init__(
        self,
        oracles: List[BaseSmoothSaddleOracle],
        stepsize: float,
        network: Network,
        z_0: ArrayPair,
        logger=Optional[Logger],
        constraints: Optional[ConstraintsL2] = None,
    ):
        self._num_nodes = len(oracles)
        oracle_sum = LinearCombSaddleOracle(
            oracles, [1 / self._num_nodes] * self._num_nodes
        )
        super().__init__(oracle_sum, z_0, None, None, logger)
        self.oracle_list = oracles
        self.stepsize = stepsize
        self.network = network
        self.constraints = constraints
        self.z_list = ArrayPair(
            np.tile(z_0.x.copy(), self._num_nodes).reshape(
                self._num_nodes, z_0.x.shape[0]
            ),
            np.tile(z_0.y.copy(), self._num_nodes).reshape(
                self._num_nodes, z_0.y.shape[0]
            ),
        )
        self.s_list = None
        self.grad_list_z = self.oracle_grad_list(self.z_list)

    def step(self):
        if self.s_list is None:
            self.s_list = self.oracle_grad_list(self.z_list)
        z_half = self.z_list - self.stepsize * self.s_list
        s_half = (
            self.s_list
            + self.oracle_grad_list(z_half)
            - self.oracle_grad_list(self.z_list)
        )
        z_new = self.mul_by_mix_mat(self.z_list) - self.stepsize * s_half
        self.s_list = (
            self.mul_by_mix_mat(self.s_list)
            + self.oracle_grad_list(z_new)
            - self.oracle_grad_list(self.z_list)
        )
        self.z_list = z_new

        self.z = ArrayPair(self.z_list.x.mean(axis=0), self.z_list.y.mean(axis=0))
        self.grad_list_z = self.oracle_grad_list(self.z_list)

    def oracle_grad_list(self, z: ArrayPair) -> ArrayPair:
        res = ArrayPair(np.empty_like(z.x), np.empty_like(z.y))
        for i in range(z.x.shape[0]):
            grad = self.oracle_list[i].grad(ArrayPair(z.x[i], z.y[i]))
            res.x[i] = grad.x
            res.y[i] = grad.y
        return res

    def mul_by_mix_mat(self, z: ArrayPair):
        self.mix_mat = self.network.__next__()[0]
        if self.logger is not None:
            self.logger.step(self)

        return ArrayPair(self.mix_mat.dot(z.x), self.mix_mat.dot(z.y))
