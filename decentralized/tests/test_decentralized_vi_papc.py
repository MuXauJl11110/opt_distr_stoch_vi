import numpy as np
from src.loggers.logger import LoggerDecentralized
from src.methods import DecentralizedVIPAPC
from src.network.config_manager import NetworkConfigManager
from src.network.network import Network
from src.oracles.base import ArrayPair
from src.oracles.saddle_simple import SquareDiffOracle
from src.utils.compute_params import compute_lam


def test_decentralized_vi_papc():
    np.random.seed(0)
    d = 20
    num_states = 1000
    num_nodes = 10
    network = Network(
        num_states,
        num_nodes,
        "gos_mat",
        config_manager=NetworkConfigManager(
            "tests/test_utils/cycle.yaml",
        ),
    )
    lam = network.peek()[1]

    oracles = [
        SquareDiffOracle(coef_x=m / num_nodes, coef_y=1 - m / num_nodes)
        for m in range(1, num_nodes + 1)
    ]
    L = 2.0
    mu = (num_nodes + 1) / num_nodes

    beta = mu / (L**2)
    chi = 1 / lam
    theta = min(1 / (2 * beta), L * np.sqrt(chi) / 3)
    eta = 1 / (3 * L * np.sqrt(chi))
    alpha = 1 - min(1 / (1 + 3 * L * np.sqrt(chi) / mu), 1 / (2 * chi))
    logger = LoggerDecentralized(
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
        network=network,
        logger=logger,
    )

    method.run(max_iter=num_states)
    assert logger.argument_primal_distance_to_opt[-1] <= 0.05
    assert logger.argument_primal_distance_to_consensus[-1] <= 0.5
    assert logger.gradient_primal_distance_to_opt[-1] <= 0.05


if __name__ == "__main__":
    test_decentralized_vi_papc()
