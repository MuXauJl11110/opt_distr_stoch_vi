import numpy as np
from src.config_managers import NetworkConfigManager
from src.network import Network
from src.oracles.base import ArrayPair
from src.oracles.saddle_simple import SquareDiffOracle
from src.runner import DecentralizedVIPAPCRunner
from src.scheduler import Scheduler
from src.utils import compute_lam


def test_decentralized_vi_papc():
    np.random.seed(0)
    d = 20
    num_nodes = 10
    num_states = 50

    oracles = [
        SquareDiffOracle(coef_x=m / num_nodes, coef_y=1 - m / num_nodes)
        for m in range(1, num_nodes + 1)
    ]
    L = 10  # 2.0
    mu = (num_nodes + 1) / num_nodes

    method_runner = DecentralizedVIPAPCRunner(L, mu, np.inf, np.inf)

    network = Network(
        num_states,
        num_nodes,
        "gos_mat",
        config_manager=NetworkConfigManager("tests/test_utils/network.yaml"),
    )
    gos_mat, _, _ = network.peek()
    z_0 = ArrayPair(np.random.rand(d), np.random.rand(d))
    y_0 = ArrayPair(np.zeros(d), np.zeros(d))
    z_true = ArrayPair(np.zeros(d), np.zeros(d))
    g_true = ArrayPair(np.zeros((num_nodes, d)), np.zeros((num_nodes, d)))

    method_runner.create_method(oracles, gos_mat, z_0, y_0, z_true, g_true)
    scheduler = Scheduler(
        method_runner=method_runner,
        network=network,
    )

    lam = compute_lam(gos_mat)[0]
    beta = mu / (L**2)
    chi = 1 / lam
    theta = min(1 / (2 * beta), L * np.sqrt(chi) / 3)
    eta = 1 / (3 * L * np.sqrt(chi))
    alpha = 1 - min(1 / (1 + 3 * L * np.sqrt(chi) / mu), 1 / (2 * chi))

    print(f"Chi:{chi}")
    print(f"Beta:{beta}")
    print(f"Theta:{theta}")
    print(f"Eta:{eta}")
    print(f"Alpha:{alpha}")

    for _ in scheduler:
        pass

    assert method_runner.logger.argument_primal_distance_to_opt[-1] <= 0.05
    assert method_runner.logger.argument_primal_distance_to_consensus[-1] <= 0.5
    assert method_runner.logger.gradient_primal_distance_to_opt[-1] <= 0.05


if __name__ == "__main__":
    test_decentralized_vi_papc()
