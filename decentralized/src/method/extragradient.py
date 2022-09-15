from datetime import datetime
from typing import Optional

import numpy as np
from src.method.base import BaseSaddleMethod
from src.method.constraints import ConstraintsL2
from src.oracles.base import ArrayPair, BaseSmoothSaddleOracle


class Extragradient(BaseSaddleMethod):
    """
    Non-distributed Extragradient method.

    oracle: BaseSmoothSaddleOracle
        Oracle of the objective function.

    stepsize: float
        Stepsize of Extragradient method.

    z_0: ArrayPair
        Initial guess.

    tolerance: Optional[float]
        Accuracy required for stopping criteria.

    stopping_criteria: Optional[str]
        Str specifying stopping criteria. Supported values:
        "grad_rel": terminate if ||f'(x_k)||^2 / ||f'(x_0)||^2 <= eps
        "grad_abs": terminate if ||f'(x_k)||^2 <= eps

    constraints: Optional[ConstraintsL2]
        L2 constraints on problem variables.
    """

    def __init__(
        self,
        oracle: BaseSmoothSaddleOracle,
        stepsize: float,
        z_0: ArrayPair,
        tolerance: Optional[float],
        stopping_criteria: Optional[str],
        constraints: Optional[ConstraintsL2] = None,
    ):
        self.oracle = oracle
        self.z = z_0.copy()
        self.stepsize = stepsize
        self.tolerance = tolerance
        if stopping_criteria == "grad_rel":
            self.stopping_criteria = self.stopping_criteria_grad_relative
        elif stopping_criteria == "grad_abs":
            self.stopping_criteria = self.stopping_criteria_grad_absolute
        elif stopping_criteria == None:
            self.stopping_criteria = self.stopping_criteria_none
        else:
            raise ValueError(
                'Unknown stopping criteria type: "{}"'.format(stopping_criteria)
            )
        if constraints is not None:
            self.constraints = constraints
        else:
            self.constraints = ConstraintsL2(+np.inf, +np.inf)

    def step(self):
        w = self.z - self.oracle.grad(self.z) * self.stepsize
        self.constraints.apply(w)
        self.grad = self.oracle.grad(w)
        self.z = self.z - self.grad * self.stepsize
        self.constraints.apply(self.z)

    def run(self, max_iter: int, max_time: float = None):
        """
        Run the method for no more than max_iter iterations and max_time seconds.
        Parameters
        ----------
        max_iter: int
            Maximum number of iterations.
        max_time: float
            Maximum time (in seconds).
        """
        self.grad_norm_0 = self.z.norm()
        if max_time is None:
            max_time = +np.inf
        if not hasattr(self, "time"):
            self.time = 0.0

        self._absolute_time = datetime.now()
        for iter_count in range(max_iter):
            if self.time > max_time:
                break
            self._update_time()
            self.step()

    def _update_time(self):
        now = datetime.now()
        self.time += (now - self._absolute_time).total_seconds()
        self._absolute_time = now

    def stopping_criteria_grad_relative(self):
        return self.grad.dot(self.grad) <= self.tolerance * self.grad_norm_0**2

    def stopping_criteria_grad_absolute(self):
        return self.grad.dot(self.grad) <= self.tolerance

    def stopping_criteria_none(self):
        return False


def extragradient_solver(
    oracle: BaseSmoothSaddleOracle,
    stepsize: float,
    z_0: ArrayPair,
    num_iter: int,
    tolerance: Optional[float] = None,
    stopping_criteria: Optional[str] = None,
    constraints: ConstraintsL2 = None,
) -> ArrayPair:
    """
    Solve the problem with standard Extragradient method up to a desired accuracy.
    """

    method = Extragradient(
        oracle, stepsize, z_0, tolerance, stopping_criteria, constraints
    )
    method.run(max_iter=num_iter)
    return method.z
