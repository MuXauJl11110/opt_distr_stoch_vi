import numpy as np
from decentralized.loggers.logger import LoggerDecentralized
from decentralized.methods import DecentralizedVIPAPC
from decentralized.oracles.base import ArrayPair
from decentralized.oracles.saddle_simple import SquareDiffOracle
from decentralized.utils import compute_lam, ring_gos_mat


def test_decentralized_vi_papc():
    np.random.seed(0)
    d = 20
    num_nodes = 10
    W = ring_gos_mat(num_nodes)
    lam = compute_lam(W)[0]

    oracles = [SquareDiffOracle(coef_x=m / num_nodes, coef_y=1 - m / num_nodes) for m in range(1, num_nodes + 1)]
    L = 2.0
    mu = (num_nodes + 1) / num_nodes

    beta = mu / (L ** 2)
    chi = 1 / lam
    theta = min(1 / (2 * beta), L * np.sqrt(chi) / 3)
    eta = 1 / (3 * L * np.sqrt(chi))
    alpha = 1 - min(1 / (1 + 3 * L * np.sqrt(chi) / mu), 1 / (2 * chi))
    logger = LoggerDecentralized(
        default_config_path="../tests/test_utils/config_decentralized.yaml",
        z_true=ArrayPair(np.zeros(d), np.zeros(d)),
        g_true=ArrayPair(np.zeros((num_nodes, d)), np.zeros((num_nodes, d))),
    )

    print(f"Chi:{chi}")
    print(f"Beta:{beta}")
    print(f"Theta:{theta}")
    print(f"Eta:{eta}")
    print(f"Alpha:{alpha}")

    z_0 = ArrayPair(np.random.rand(d), np.zeros(d))
    z_0_list = [z_0] * num_nodes
    y_0 = ArrayPair(np.zeros(d), np.zeros(d))
    y_0_list = [y_0] * num_nodes
    method = DecentralizedVIPAPC(
        oracles=oracles,
        z_0=z_0_list,
        y_0=y_0_list,
        eta=eta,
        theta=theta,
        alpha=alpha,
        beta=beta,
        gos_mat=W,
        logger=logger,
    )

    method.run(max_iter=1000)
    assert logger.argument_primal_distance_to_opt[-1] <= 0.05
    assert logger.argument_primal_distance_to_consensus[-1] <= 0.5
    assert logger.gradient_primal_distance_to_opt[-1] <= 0.05


if __name__ == "__main__":
    test_decentralized_vi_papc()
