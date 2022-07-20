import numpy as np
from decentralized.loggers.logger import LoggerDecentralized
from decentralized.methods import DecentralizedExtragradientGT
from decentralized.oracles.base import ArrayPair
from decentralized.oracles.saddle_simple import SquareDiffOracle
from decentralized.tests.test_utils.utils import gen_mix_mat
from decentralized.utils import compute_lam_2


def test_decentralized_extragradient():
    np.random.seed(0)
    d = 20
    num_nodes = 10
    mix_mat = gen_mix_mat(num_nodes)
    lam = compute_lam_2(mix_mat)

    z_0 = ArrayPair(np.random.rand(d), np.random.rand(d))
    oracles = [SquareDiffOracle(coef_x=m / num_nodes, coef_y=1 - m / num_nodes) for m in range(1, num_nodes + 1)]
    L = 2.0
    mu = (num_nodes + 1) / num_nodes
    gamma = mu

    eta = max((1 - lam) ** 2 * gamma / (500 * L), (1 - lam) ** (4 / 3) * mu ** (1 / 3) / (40 * L ** (4 / 3)))
    eta = min(eta, (1 - lam) ** 2 / (22 * L))
    logger = LoggerDecentralized(
        default_config_path="../tests/test_utils/config_decentralized.yaml",
        z_true=ArrayPair(np.zeros(d), np.zeros(d)),
        g_true=ArrayPair(np.zeros((num_nodes, d)), np.zeros((num_nodes, d))),
    )

    method = DecentralizedExtragradientGT(
        oracles=oracles, stepsize=eta, mix_mat=mix_mat, z_0=z_0, logger=logger, constraints=None
    )
    method.run(max_iter=10000)
    assert logger.argument_primal_distance_to_opt[-1] <= 0.05
    assert logger.argument_primal_distance_to_consensus[-1] <= 0.5
    assert logger.gradient_primal_distance_to_opt[-1] <= 0.05


if __name__ == "__main__":
    test_decentralized_extragradient()
