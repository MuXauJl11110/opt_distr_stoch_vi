# import numpy as np
# from decentralized.loggers.logger import LoggerDecentralized
# from decentralized.methods.decentralized_extragradient_con import (
#     DecentralizedExtragradientCon,
# )
# from decentralized.oracles.base import ArrayPair
# from decentralized.oracles.saddle_simple import SquareDiffOracle
# from decentralized.tests.test_utils.utils import gen_mix_mat
# from decentralized.utils.compute_params import compute_lam_2


# def test_decentralized_extragradient():
#     np.random.seed(0)
#     d = 20
#     num_nodes = 10
#     mix_mat = gen_mix_mat(num_nodes)
#     lam = compute_lam_2(mix_mat)

#     z_0 = ArrayPair(np.random.rand(d), np.random.rand(d))
#     oracles = [
#         SquareDiffOracle(coef_x=m / num_nodes, coef_y=1 - m / num_nodes)
#         for m in range(1, num_nodes + 1)
#     ]
#     L = 2.0
#     mu = (num_nodes + 1) / num_nodes

#     gamma = 1 / (4 * L)
#     gossip_step = (1 - np.sqrt(1 - lam**2)) / (1 + np.sqrt(1 - lam**2))
#     eps = 1e-4
#     eps_0 = eps * mu * gamma * (1 + gamma * L) ** 2
#     con_iters = int(5 * np.sqrt(1 / (1 - lam)) * np.log(1 / eps_0))
#     logger = LoggerDecentralized(
#         default_config_path="../tests/test_utils/config_decentralized.yaml",
#         z_true=ArrayPair(np.zeros(d), np.zeros(d)),
#         g_true=ArrayPair(np.zeros((num_nodes, d)), np.zeros((num_nodes, d))),
#     )

#     method = DecentralizedExtragradientCon(
#         oracles=oracles,
#         stepsize=gamma,
#         con_iters=con_iters,
#         mix_mat=mix_mat,
#         gossip_step=gossip_step,
#         z_0=z_0,
#         logger=logger,
#         constraints=None,
#     )
#     method.run(max_iter=100)
#     assert logger.argument_primal_distance_to_opt[-1] <= 0.05
#     assert logger.argument_primal_distance_to_consensus[-1] <= 0.5
#     assert logger.gradient_primal_distance_to_opt[-1] <= 0.05


# if __name__ == "__main__":
#     test_decentralized_extragradient()
