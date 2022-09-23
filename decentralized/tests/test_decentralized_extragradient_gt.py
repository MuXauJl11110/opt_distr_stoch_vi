import numpy as np
from src.loggers.logger import LoggerDecentralized
from src.methods.decentralized_extragradient_gt import DecentralizedExtragradientGT
from src.network.config_manager import NetworkConfigManager
from src.network.network import Network
from src.oracles.base import ArrayPair
from src.oracles.saddle_simple import SquareDiffOracle
from src.utils.compute_params import compute_lam_2


def test_decentralized_extragradient():
    np.random.seed(0)
    d = 20
    num_states = 15000
    num_nodes = 10
    network = Network(
        num_states,
        num_nodes,
        "mix_mat",
        config_manager=NetworkConfigManager(
            "tests/test_utils/cycle.yaml",
        ),
    )
    lam = compute_lam_2(network.peek()[0])

    z_0 = ArrayPair(np.random.rand(d), np.random.rand(d))
    oracles = [
        SquareDiffOracle(coef_x=m / num_nodes, coef_y=1 - m / num_nodes)
        for m in range(1, num_nodes + 1)
    ]
    L = 2.0
    mu = (num_nodes + 1) / num_nodes
    gamma = mu

    eta = max(
        (1 - lam) ** 2 * gamma / (500 * L),
        (1 - lam) ** (4 / 3) * mu ** (1 / 3) / (40 * L ** (4 / 3)),
    )
    eta = min(eta, (1 - lam) ** 2 / (22 * L))
    logger = LoggerDecentralized(
        z_true=ArrayPair(np.zeros(d), np.zeros(d)),
        g_true=ArrayPair(np.zeros((num_nodes, d)), np.zeros((num_nodes, d))),
    )

    method = DecentralizedExtragradientGT(
        oracles=oracles,
        stepsize=eta,
        network=network,
        z_0=z_0,
        logger=logger,
        constraints=None,
    )
    method.run(max_iter=num_states)
    assert logger.argument_primal_distance_to_opt[-1] <= 0.07
    assert logger.argument_primal_distance_to_consensus[-1] <= 0.5
    assert logger.gradient_primal_distance_to_opt[-1] <= 0.05


if __name__ == "__main__":
    test_decentralized_extragradient()
