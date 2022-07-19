import numpy as np
import pytest
from decentralized.loggers.logger import Logger
from decentralized.methods import DecentralizedVIADOM
from decentralized.oracles.base import ArrayPair
from decentralized.oracles.saddle_simple import BatchedSquareDiffOracle
from decentralized.utils import compute_lam, ring_gos_mat


@pytest.mark.tryfirst
def test_decentralized_vi_adom():
    np.random.seed(0)
    d = 20
    num_nodes = 10
    W = ring_gos_mat(num_nodes)
    lam = compute_lam(W)[0]

    oracles = [
        BatchedSquareDiffOracle(coef_x=m / num_nodes, coef_y=1 - m / num_nodes) for m in range(1, num_nodes + 1)
    ]
    L = 2.0
    b = 2
    L_avg = L
    mu = (num_nodes + 1) / num_nodes

    chi = 1 / lam
    p = 1 / 16
    omega = p
    theta = 1 / 2
    gamma = min(mu / (16 * (L ** 2)), np.inf)  # , b * omega / (24 * (L_avg ** 2) * eta_z)
    beta = 5 * gamma
    nu = mu / 4
    alpha = 1 / 2
    tau = min(mu / (32 * L * chi), mu * np.sqrt(b * p) / (32 * L_avg))
    eta_x = min(1 / (900 * chi * gamma), nu / (36 * tau * (chi ** 2)))
    eta_y = min(1 / (4 * gamma), nu / (8 * tau))
    eta_z = min(
        1 / (8 * L * chi),
        1 / (32 * eta_y),
        np.sqrt(alpha * b * omega) / (8 * L_avg),
    )
    logger = Logger(default_config_path="../tests/test_utils/config.yaml")

    print(f"chi: {chi}")
    print(f"p: {p}")
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

    x_0 = ArrayPair(np.zeros(d), np.zeros(d))
    x_0_list = [x_0] * num_nodes
    y_0 = ArrayPair(np.random.rand(d), np.zeros(d))
    y_0_list = [y_0] * num_nodes
    w_0 = ArrayPair(np.random.rand(d), np.zeros(d))
    w_0_list = [w_0] * num_nodes
    method = DecentralizedVIADOM(
        oracles=oracles,
        x_0=x_0_list,
        y_0=y_0_list,
        w_0=w_0_list,
        eta_x=eta_x,
        eta_y=eta_y,
        eta_z=eta_z,
        theta=theta,
        alpha=alpha,
        gamma=gamma,
        omega=omega,
        tau=tau,
        nu=nu,
        beta=beta,
        b=b,
        p=p,
        gos_mat=W,
        n=d,
        logger=logger,
    )

    method.run(max_iter=1000)
    z_star = logger.argument_primal_value[-1]
    assert z_star.dot(z_star) <= 0.05


if __name__ == "__main__":
    test_decentralized_vi_adom()