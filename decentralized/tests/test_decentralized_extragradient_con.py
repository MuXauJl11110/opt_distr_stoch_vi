import numpy as np
from src.loggers.logger import LoggerDecentralized
from src.methods.decentralized_extragradient_con import DecentralizedExtragradientCon
from src.network.config_manager import NetworkConfigManager
from src.network.network import Network
from src.oracles.base import ArrayPair
from src.oracles.saddle_simple import SquareDiffOracle
from src.utils.compute_params import compute_lam_2


def test_decentralized_extragradient():
    np.random.seed(0)
    d = 20
    num_states = 100
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

    gamma = 1 / (4 * L)
    gossip_step = (1 - np.sqrt(1 - lam**2)) / (1 + np.sqrt(1 - lam**2))
    eps = 1e-4
    eps_0 = eps * mu * gamma * (1 + gamma * L) ** 2
    con_iters = int(5 * np.sqrt(1 / (1 - lam)) * np.log(1 / eps_0))
    logger = LoggerDecentralized(
        z_true=ArrayPair(np.zeros(d), np.zeros(d)),
        g_true=ArrayPair(np.zeros((num_nodes, d)), np.zeros((num_nodes, d))),
    )

    method = DecentralizedExtragradientCon(
        oracles=oracles,
        stepsize=gamma,
        con_iters=con_iters,
        network=network,
        gossip_step=gossip_step,
        z_0=z_0,
        logger=logger,
        constraints=None,
    )
    method.run(max_iter=num_states)
    assert logger.argument_primal_distance_to_opt[-1] <= 0.05
    assert logger.argument_primal_distance_to_consensus[-1] <= 0.5
    assert logger.gradient_primal_distance_to_opt[-1] <= 0.05


if __name__ == "__main__":
    test_decentralized_extragradient()
