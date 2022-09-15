from typing import Optional

import numpy as np
from src.config_managers.runner import RunnerConfigManager
from src.layout.layout import Layout
from src.oracles.base import ArrayPair
from src.runner.containers.decentralized_extragradient_con import EG_CON_container
from src.runner.containers.decentralized_extragradient_gt import EG_GT_container
from src.runner.containers.decentralized_sliding import Sliding_container
from src.runner.containers.decentralized_vi_adom import VIADOM_container
from src.runner.decentralized_extragradient_con import (
    DecentralizedExtragradientConRunner,
)
from src.runner.decentralized_extragradient_gt import DecentralizedExtragradientGTRunner
from src.runner.decentralized_sliding import DecentralizedSaddleSlidingRunner
from src.runner.decentralized_vi_adom import DecentralizedVIADOMRunner
from src.utils.utils import call_with_redundant


class RunnerLayout(object):
    """
    Instrument for displaying runner widget.
    """

    def __init__(
        self, runner_cm: Optional[RunnerConfigManager] = RunnerConfigManager()
    ):
        """
        :param general_cm: Runner config manager.
        """
        self.runner_cm = runner_cm
        self.layout = self.initialize_layout()

    def initialize_layout(self):
        return Layout(
            "Tab",
            ["EG-CON", "EG-GT", "Sliding", "VIADOM"],
            [
                EG_CON_container,
                EG_GT_container,
                Sliding_container,
                VIADOM_container,
            ],
        )

    runner_сfg = {
        "EG-CON": (DecentralizedExtragradientConRunner, "mix_mat"),
        "EG-GT": (DecentralizedExtragradientGTRunner, "mix_mat"),
        "Sliding": (DecentralizedSaddleSlidingRunner, "mix_mat"),
        "VIADOM": (DecentralizedVIADOMRunner, "gos_mat"),
    }

    def __iter__(self):
        cfg = self.layout.value

        for method_name, method_cfg in cfg.items():
            if method_cfg["use_method"]:
                runner = RunnerLayout.runner_сfg[method_name][0]
                if method_cfg["Parameters:"]["use_precomputed"]:
                    pass

                yield (
                    method_name,
                    call_with_redundant(
                        method_cfg["Parameters:"] | self.general_cfg,
                        runner,
                    ),
                    RunnerLayout.parse_initial_guess(
                        method_cfg["Initial guess:"], self.d
                    ),
                )

    def parse_initial_guess(cfg: dict, d: int):
        def get_vector(vec_type: str, d: int):
            # if vec_type == "Random":
            #    return
            if vec_type == "Zero":
                return ArrayPair.zeros(d)
            # elif vec_type == "Simplex":
            #    return np.ones(d) / d
            else:
                raise ValueError(f"Unknown vector type: {vec_type}!")

        return {k: get_vector(v, d) for k, v in cfg.items()}

    def __call__(self, d: int, general_cfg: dict):
        self.d = d
        self.general_cfg = general_cfg
        return self
