import sys

sys.path.append("../")

import numpy as np
from loggers import Logger
from methods import DecentralizedExtragradientGT
from oracles import ArrayPair, SquareDiffOracle
from utils import compute_lam_2, ring_adj_mat


def test_decentralized_extragradient():
    np.random.seed(0)
    d = 2
    num_nodes = 10
    mix_mat = ring_adj_mat(num_nodes)
    lam = compute_lam_2(mix_mat)

    z_0 = ArrayPair(np.random.rand(d), np.random.rand(d))
    oracles = [SquareDiffOracle(coef_x=m / num_nodes, coef_y=1 - m / num_nodes) for m in range(1, num_nodes + 1)]
    L = 2.0
    mu = (num_nodes + 1) / num_nodes
    gamma = mu

    eta = max((1 - lam) ** 2 * gamma / (500 * L), (1 - lam) ** (4 / 3) * mu ** (1 / 3) / (40 * L ** (4 / 3)))
    eta = min(eta, (1 - lam) ** 2 / (22 * L))
    logger = Logger()

    method = DecentralizedExtragradientGT(
        oracles=oracles, stepsize=eta, mix_mat=mix_mat, z_0=z_0, logger=logger, constraints=None
    )
    method.run(max_iter=10000)
    z_star = logger.argument_primal_value[-1]
    assert z_star.dot(z_star) <= 0.05


if __name__ == "__main__":
    test_decentralized_extragradient()
