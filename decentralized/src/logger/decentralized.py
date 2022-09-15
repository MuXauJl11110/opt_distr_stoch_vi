from typing import Optional

import numpy as np
from src.logger.centralized import LoggerCentralized
from src.config_managers.logger import LoggerConfigManager
from src.method.base import BaseSaddleMethod


class LoggerDecentralized(LoggerCentralized):
    """
    Instrument for saving method history during its iterations for decentralized methods.
    """

    def __init__(
        self,
        method: BaseSaddleMethod,
        z_true: Optional[np.ndarray] = None,
        g_true: Optional[np.ndarray] = None,
        config_manager: Optional[LoggerConfigManager] = LoggerConfigManager(
            "src/config_managers/configs/default/logger_decentralized.yaml",
            "src/config_managers/configs/general/logger_decentralized.yaml",
        ),
    ):
        """
        :param method: Method instance.
        :param z_true: Exact solution of the problem.
        :param g_true: Gradient of the exact solution of the problem.
        :param config_manager: Configuration manager.
        """
        super().__init__(
            method=method, z_true=z_true, g_true=g_true, config_manager=config_manager
        )

    def log_value(self, prefix: str, value: dict, src: object):
        if self.current_step % getattr(self, prefix + "_step") == 0:
            data = getattr(src, value["source"])
            getattr(self, prefix).append(data[value["nodes"]])
            if getattr(self, prefix + "_verbose"):
                self.output_row.append(data)

    def log_distance(self, prefix: str, cfg: dict):
        if self.current_step % getattr(self, prefix + "_step") == 0:
            source = getattr(self.method, cfg["source"])
            if cfg["target"].split("_")[-1] == "true":
                target = getattr(self, cfg["target"])
            else:
                target = getattr(self.method, cfg["target"])
            distance_config = self.config_manager.distance_functions[cfg["distance"]]
            func = getattr(eval(distance_config["object"]), distance_config["name"])
            distance = func(source, target, **distance_config["kwargs"])
            getattr(self, prefix).append(distance)
            if getattr(self, prefix + "_verbose"):
                self.output_row.append(distance)
