import numpy as np
from src.config_managers import NetworkConfigManager
from src.network import Network
from src.oracles.base import ArrayPair
from src.oracles.saddle_simple import SquareDiffOracle
from src.runner import DecentralizedVIADOMRunner
from src.scheduler import Scheduler
from src.utils import compute_lam


def test_decentralized_vi_adom():
    np.random.seed(0)
    d = 20
    num_nodes = 10
    num_states = 50

    oracles = [
        SquareDiffOracle(coef_x=m / num_nodes, coef_y=1 - m / num_nodes)
        for m in range(1, num_nodes + 1)
    ]
    L = 2.0
    b = 1
    L_avg = L
    mu = (num_nodes + 1) / num_nodes

    method_runner = DecentralizedVIADOMRunner(b, L, L_avg, mu, np.inf, np.inf)

    network = Network(
        num_states,
        num_nodes,
        "gos_mat",
        config_manager=NetworkConfigManager("tests/test_utils/network.yaml"),
    )
    gos_mat, _, _ = network.peek()
    x_0 = ArrayPair(np.zeros(d), np.zeros(d))
    y_0 = ArrayPair(np.random.rand(d), np.zeros(d))
    z_0 = ArrayPair(np.random.rand(d), np.zeros(d))
    z_true = ArrayPair(np.zeros(d), np.zeros(d))
    g_true = ArrayPair(np.zeros((num_nodes, d)), np.zeros((num_nodes, d)))

    method_runner.create_method(oracles, gos_mat, x_0, y_0, z_0, z_true, g_true)
    scheduler = Scheduler(
        method_runner=method_runner,
        network=network,
    )

    lam = compute_lam(gos_mat)[0]
    chi = 1 / lam
    omega = 1 / 16
    theta = 1 / 2
    gamma = min(
        mu / (16 * (L**2)), np.inf
    )  # , b * omega / (24 * (L_avg ** 2) * eta_z)
    beta = 5 * gamma
    nu = mu / 4
    alpha = 1 / 2
    tau = min(mu / (32 * L * chi), mu * np.sqrt(b * omega) / (32 * L_avg))
    eta_x = min(1 / (900 * chi * gamma), nu / (36 * tau * (chi**2)))
    eta_y = min(1 / (4 * gamma), nu / (8 * tau))
    eta_z = min(
        1 / (8 * L * chi), 1 / (32 * eta_y), np.sqrt(alpha * b * omega) / (8 * L_avg)
    )

    print(f"chi: {chi}")
    print(f"omega: {omega}")
    print(f"theta: {theta}")
    print(f"gamma: {gamma}")
    print(f"beta: {beta}")
    print(f"nu: {nu}")
    print(f"alpha: {alpha}")
    print(f"tau: {tau}")
    print(f"eta_x: {eta_x}")
    print(f"eta_y: {eta_y}")
    print(f"eta_z: {eta_z}")

    for _ in scheduler:
        pass

    print(method_runner.logger.argument_primal_distance_to_opt)
    assert method_runner.logger.argument_primal_distance_to_opt[-1] <= 0.05
    assert method_runner.logger.argument_primal_distance_to_consensus[-1] <= 0.5
    assert method_runner.logger.gradient_primal_distance_to_opt[-1] <= 0.05


if __name__ == "__main__":
    test_decentralized_vi_adom()
