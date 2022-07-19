import numpy as np
from decentralized.loggers import Logger
from decentralized.methods import ConstraintsL2, Extragradient
from decentralized.oracles.base import ArrayPair
from decentralized.oracles.robust_linear import (
    RobustLinearOracle,
    create_robust_linear_oracle,
)
from decentralized.oracles.saddle_simple import ScalarProdOracle


def create_random_robust_linear_oracle(n: int, d: int) -> RobustLinearOracle:
    A = np.random.randn(n, d)
    b = np.random.randn(n)
    oracle = create_robust_linear_oracle(A, b, regcoef_x=0.1, regcoef_delta=0.5, normed=True)
    return oracle


def test_extragradient_step():
    np.random.seed(0)
    n, d = 50, 8
    oracle = create_random_robust_linear_oracle(n, d)
    z_0 = ArrayPair(np.random.rand(d), np.random.rand(d))
    method = Extragradient(oracle, 0.1, z_0, tolerance=None, stopping_criteria=None, logger=None)
    method.step()


def test_extragradient_run_robust_linear():
    np.random.seed(0)
    n, d = 50, 8
    oracle = create_random_robust_linear_oracle(n, d)
    z_0 = ArrayPair(np.random.rand(d), np.random.rand(d))
    method = Extragradient(oracle, 0.1, z_0, tolerance=None, stopping_criteria=None, logger=None)
    method.run(max_iter=20)


def test_extragradient_run_scalar_prod():
    np.random.seed(0)
    d = 20
    oracle = ScalarProdOracle()
    z_0 = ArrayPair(np.random.rand(d), np.random.rand(d))
    logger = Logger(default_config_path="../tests/test_utils/config.yaml")
    method = Extragradient(oracle, 0.5, z_0, tolerance=None, stopping_criteria=None, logger=logger)
    method.run(max_iter=1000)
    z_star = logger.argument_primal_value[-1]
    assert z_star.dot(z_star) <= 0.05


def test_extragradient_run_scalar_prod_constrained():
    np.random.seed(0)
    d = 20
    oracle = ScalarProdOracle()
    z_0 = ArrayPair(np.random.rand(d), np.random.rand(d))
    logger = Logger(default_config_path="../tests/test_utils/config.yaml")
    constraints = ConstraintsL2(1.0, 2.0)
    method = Extragradient(
        oracle, 0.5, z_0, tolerance=None, stopping_criteria=None, logger=logger, constraints=constraints
    )
    method.run(max_iter=1000)
    z_star = logger.argument_primal_value[-1]
    assert z_star.dot(z_star) <= 0.05


if __name__ == "__main__":
    test_extragradient_step()
    test_extragradient_run_robust_linear()
    test_extragradient_run_scalar_prod()
    test_extragradient_run_scalar_prod_constrained()
