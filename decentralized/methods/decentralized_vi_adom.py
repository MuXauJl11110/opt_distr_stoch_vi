from typing import List, Optional

import numpy as np
from decentralized.loggers.logger import Logger
from decentralized.methods.base import BaseSaddleMethod
from decentralized.methods.constraints import ConstraintsL2
from decentralized.oracles.base import (
    ArrayPair,
    BaseSmoothSaddleOracle,
    LinearCombSaddleOracle,
)


class DecentralizedVIADOM(BaseSaddleMethod):
    r"""
    Algorithm 2 from https://arxiv.org/pdf/2202.02771.pdf for one T period.

    Parameters
    ----------
    oracles: List[BaseSmoothSaddleOracle]
        List of oracles corresponding to network nodes.

    x_0: List[ArrayPair]
        Initial guess. :math:`x^0 \in  \mathcal{L}^{\perp}`.

    y_0: List[ArrayPair]
        Initial guess. :math:`y^0 \in \big(\mathbb{R}^d\big)^M`

    w_0: List[ArrayPair]
        Initial guess. :math:`w^0 \in (dom g)^M`

    eta_x, eta_y, eta_z, theta: float
        Stepsizes. :math:`\eta_x, \eta_y, \eta_z, \theta > 0`

    alpha, gamma, omega, tau: float
        Momentums.

    nu, beta: float
        Parameters.

    b: int
        Batchsize. :math:`b \in \{1, \cdots, n\}`

    p: float
        Probability. :math:`p \in (0, 1)`

    gos_mat: np.ndarray
        Gossip matrix is multiplication of T gossip matrices.

    n: int
        :math:`F_m(z) \coloneqq \frac{1}{n} \sum_{i=1}^n F_{m, i}(z)`

    logger: Optional[Logger]
        Stores the history of the method during its iterations.

    constraints: Optional[ConstraintsL2]
        Constraints.
    """

    def __init__(
        self,
        oracles: List[BaseSmoothSaddleOracle],
        x_0: List[ArrayPair],
        y_0: List[ArrayPair],
        w_0: List[ArrayPair],
        eta_x: float,
        eta_y: float,
        eta_z: float,
        theta: float,
        alpha: float,
        gamma: float,
        omega: float,
        tau: float,
        nu: float,
        beta: float,
        b: int,
        p: float,
        gos_mat: np.ndarray,
        n: int,
        logger=Optional[Logger],
        constraints: Optional[ConstraintsL2] = None,
    ):
        if len(x_0) != len(y_0) != (w_0):
            raise ValueError("Number of x^0, y^0 and w^0 should be equal!")
        self._num_nodes = len(oracles)  # M
        oracle_sum = LinearCombSaddleOracle(oracles, [1 / self._num_nodes] * self._num_nodes)
        super().__init__(oracle_sum, w_0[0], None, None, logger)
        self.oracle_list = oracles

        self.eta_x = eta_x
        self.eta_y = eta_y
        self.eta_z = eta_z
        self.theta = theta
        self.alpha = alpha
        self.gamma = gamma
        self.omega = omega
        self.tau = tau
        self.nu = nu
        self.beta = beta
        self.b = b
        self.p = p

        self.gos_mat = gos_mat
        self.n = n

        if constraints is not None:
            self.constraints = constraints
        else:
            self.constraints = ConstraintsL2(+np.inf, +np.inf)

        def init_from_list(arr: List[ArrayPair]):
            return ArrayPair(
                np.array([el.x.copy() for el in arr]),
                np.array([el.y.copy() for el in arr]),
            )

        self.z_list = init_from_list(w_0)
        self.z_list_old = init_from_list(w_0)

        self.w_list = init_from_list(w_0)
        self.w_list_old = init_from_list(w_0)

        self.y_list = init_from_list(y_0)
        self.y_list_old = init_from_list(y_0)
        self.y_list_f = init_from_list(y_0)

        self.x_list = init_from_list(x_0)
        self.x_list_old = init_from_list(x_0)
        self.x_list_f = init_from_list(x_0)

        self.m_list = ArrayPair(
            np.zeros_like(self.z_list.x),
            np.zeros_like(self.z_list.y),
        )

        self.current_step = 0

    def step(self):
        sample = np.random.choice(self.n, size=(self._num_nodes, self.b), replace=False)
        sample_half = np.random.choice(self.n, size=(self._num_nodes, self.b), replace=False)

        self.grad_list_z_batched = self.oracle_grad_list(self.z_list, sample)
        self.grad_list_z_old_batched = self.oracle_grad_list(self.z_list_old, sample)
        self.grad_list_w_old_batched = self.oracle_grad_list(self.w_list_old, sample)
        self.grad_list_w_old = self.oracle_grad_list(self.w_list_old, np.ones((self.n, 1)))

        delta = (
            (1 + self.alpha) * self.grad_list_z_batched
            - self.grad_list_w_old_batched
            - self.alpha * self.grad_list_z_old_batched
        ) / self.b + self.grad_list_w_old
        Delta_z = delta - self.nu * self.z_list - self.y_list - self.alpha * (self.y_list - self.y_list_old)

        self.z_list_old = self.z_list.copy()
        self.z_list = (1 - self.omega) * self.z_list + self.omega * self.z_list - self.eta_z * Delta_z
        self.constraints.apply_per_row(self.z_list)

        y_list_c = self.tau * self.y_list + (1 - self.tau) * self.y_list_f
        x_list_c = self.tau * self.x_list + (1 - self.tau) * self.x_list_f

        Delta_y = (
            (y_list_c + x_list_c) / self.nu
            + self.z_list
            + self.gamma * (self.y_list + self.x_list + self.nu * self.z_list_old)
        )

        self.grad_list_z_batched = self.oracle_grad_list(self.z_list, sample_half)
        self.grad_list_w_batched = self.oracle_grad_list(self.w_list, sample_half)
        self.grad_list_w = self.oracle_grad_list(self.w_list, np.ones(self.n))

        delta_half = (self.grad_list_z_batched - self.grad_list_w_batched) / self.b + self.grad_list_w

        Delta_x = (y_list_c + x_list_c) / self.nu + self.beta * (self.x_list + delta_half)
        x_mixed = ArrayPair(
            self.gos_mat @ (self.eta_x * Delta_x.x + self.m_list.x),
            self.gos_mat @ (self.eta_x * Delta_x.y + self.m_list.y),
        )

        self.y_list_old = self.y_list.copy()
        self.y_list = self.y_list - self.eta_y * Delta_y
        self.x_list_old = self.x_list.copy()
        self.x_list = self.x_list - x_mixed

        self.m_list = self.eta_x * Delta_x + self.m_list - x_mixed

        self.y_list_f = y_list_c + self.tau * (self.y_list - self.y_list_old)
        self.x_list_f = x_list_c - self.theta * ArrayPair(
            self.gos_mat @ (y_list_c.x + x_list_c.x),
            self.gos_mat @ (y_list_c.y + x_list_c.y),
        )

        self.w_list_old = self.w_list.copy()
        if np.random.random() < self.p:
            self.w_list = self.z_list_old.copy()
        else:
            self.w_list = self.w_list_old.copy()

        self.current_step += 1

    def oracle_grad_list(self, z: ArrayPair, batch: np.ndarray) -> ArrayPair:
        """
        Compute oracle gradients at each computational network node.

        Parameters
        ----------
        z: ArrayPair
            Point at which the gradients are computed.

        batch: np.ndarray
            Precomputed batches for the each computational node.

        Returns
        -------
        grad: ArrayPair
        """
        res = ArrayPair(np.empty_like(z.x), np.empty_like(z.y))
        print(batch.shape)
        for i in range(self._num_nodes):
            grad = self.oracle_list[i].grad(ArrayPair(z.x[i], z.y[i]), batch[i])
            res.x[i] = grad.x
            res.y[i] = grad.y
        return res
