from typing import List, Optional

import numpy as np
from src.method.base import BaseSaddleMethod
from src.method.constraints import ConstraintsL2
from src.oracles.base import ArrayPair, BaseSmoothSaddleOracle


class DecentralizedVIPAPC(BaseSaddleMethod):
    """
    Algorithm 1 from https://arxiv.org/pdf/2202.02771.pdf.

    Parameters
    ----------
    oracles: List[BaseSmoothSaddleOracle]
        List of oracles corresponding to network nodes.

    z_0: List[ArrayPair]
        List of initial points.

    y_0: List[ArrayPair]
        List of orthogonal complement points.

    eta, theta: float
        Stepsizes.

    alpha, beta: float
        Momentums.

    gos_mat: np.ndarray
        Gossip matrix.
    """

    def __init__(
        self,
        oracles: List[BaseSmoothSaddleOracle],
        z_0: List[ArrayPair],
        y_0: List[ArrayPair],
        eta: float,
        theta: float,
        alpha: float,
        beta: float,
        gos_mat: np.ndarray,
        constraints: Optional[ConstraintsL2] = None,
    ):
        if len(z_0) != len(y_0):
            raise ValueError("Number of x^0 and y^0 should be equal!")
        self._num_nodes = len(oracles)  # M
        self.oracle_list = oracles

        self.eta = eta
        self.theta = theta
        self.alpha = alpha
        self.beta = beta

        self.gos_mat = gos_mat

        if constraints is not None:
            self.constraints = constraints
        else:
            self.constraints = ConstraintsL2(+np.inf, +np.inf)

        self.z_list = ArrayPair(
            np.array([z.x.copy() for z in z_0]),
            np.array([z.y.copy() for z in z_0]),
        )
        self.z_list_old = ArrayPair(
            np.array([z.x.copy() for z in z_0]),
            np.array([z.y.copy() for z in z_0]),
        )

        self.y_list = ArrayPair(
            np.array([z.x.copy() for z in y_0]),
            np.array([z.y.copy() for z in y_0]),
        )
        self.y_list_old = ArrayPair(
            np.array([z.x.copy() for z in y_0]),
            np.array([z.y.copy() for z in y_0]),
        )

    def step(self):
        self.grad_list_z = self.oracle_grad_list(self.z_list)
        self.grad_list_z_old = self.oracle_grad_list(self.z_list_old)
        delta_k = (
            self.grad_list_z
            + self.alpha * (self.grad_list_z - self.grad_list_z_old)
            - (self.y_list + self.alpha * (self.y_list - self.y_list_old))
        )
        self.z_list_old = self.z_list.copy()
        self.y_list_old = self.y_list.copy()
        self.z_list = self.z_list - self.eta * delta_k
        self.constraints.apply_per_row(self.z_list)
        self.y_list = self.gossip(self.z_list, self.y_list)

        self.z = ArrayPair(self.z_list.x.mean(axis=0), self.z_list.y.mean(axis=0))

    def oracle_grad_list(self, z: ArrayPair) -> ArrayPair:
        """
        Compute oracle gradients at each computational network node.

        Parameters
        ----------
        z: ArrayPair
            Point at which the gradients are computed.

        Returns
        -------
        grad: ArrayPair
        """
        res = ArrayPair(np.empty_like(z.x), np.empty_like(z.y))
        for i in range(z.x.shape[0]):
            grad = self.oracle_list[i].grad(ArrayPair(z.x[i], z.y[i]))
            res.x[i] = grad.x
            res.y[i] = grad.y
        return res

    def gossip(self, z: ArrayPair, y: ArrayPair):
        """
        Accessing gossip step.

        Parameters
        ----------
        z: ArrayPair
            Initial values at nodes.

        y: ArrayPair
            Initial values at complement points.

        Returns
        -------
        z_mixed: ArrayPair
            Values at nodes after gossip step.
        """
        z_mixed = ArrayPair(
            y.x
            - self.theta
            * (self.gos_mat @ (z.x - self.beta * (self.oracle_grad_list(z).x - y.x))),
            y.y
            - self.theta
            * (self.gos_mat @ (z.y - self.beta * (self.oracle_grad_list(z).y - y.y))),
        )
        return z_mixed
