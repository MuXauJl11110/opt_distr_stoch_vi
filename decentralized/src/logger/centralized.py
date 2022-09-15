from typing import Optional

import numpy as np
from prettytable import PrettyTable
from src.config_managers.logger import LoggerConfigManager
from src.method.base import BaseSaddleMethod


class LoggerCentralized(object):
    """
    Instrument for saving the method's history during its iterations.
    """

    def __init__(
        self,
        method: BaseSaddleMethod,
        z_true: Optional[np.ndarray] = None,
        g_true: Optional[np.ndarray] = None,
        config_manager: Optional[LoggerConfigManager] = LoggerConfigManager(),
    ):
        """
        :param method: Method instance.
        :param z_true: Exact solution of the problem.
        :param g_true: Gradient of the exact solution of the problem.
        :param config_manager: Configuration manager.
        """
        self.current_step = 0
        self.config_manager = config_manager

        headers = []
        self.tracked_values = []
        for track_type, track_value in self.config_manager.config.items():
            for space_type, space_value in track_value.items():
                for value_type, value in space_value.items():
                    prefix = track_type + "_" + space_type + "_" + value_type
                    setattr(self, prefix, list())
                    setattr(self, prefix + "_step", value["step"])
                    setattr(self, prefix + "_verbose", value["verbose"])
                    self.tracked_values.append(prefix)
                    if value["verbose"]:
                        headers.append(prefix)

        if len(headers) > 0:
            self.output_table = PrettyTable(headers)

        self.output_row = []

        self.method = method
        self.z_true = z_true
        self.g_true = g_true

    def start(self):
        pass

    def log_value(self, prefix: str, value: dict, src: object):
        if self.current_step % getattr(self, prefix + "_step") == 0:
            data = getattr(src, value["source"])
            getattr(self, prefix).append(data)
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

    def step(self):
        if len(self.output_row) > 0:
            self.output_table.add_row(self.output_row)
            print(self.output_table)

            self.output_table.clear_rows()
            self.output_row = []

        for track_type, track_value in self.config_manager.config.items():
            for space_type, space_value in track_value.items():
                for value_type, value in space_value.items():
                    prefix = track_type + "_" + space_type + "_" + value_type
                    if value_type in self.config_manager.available_values:
                        self.log_value(prefix, value, self.method)
                    elif value_type in self.config_manager.available_distances:
                        self.log_distance(prefix, value)
                    else:
                        raise ValueError("Unknown value type!")

        self.current_step += 1

    def end(self):
        if len(self.tracked_values) > 0:
            output_str = "".join(
                [
                    el + ", " if i + 1 < len(self.tracked_values) else el
                    for i, el in enumerate(self.tracked_values)
                ]
            )
            print(output_str + " can be accessed at corresponding logger class fields.")

    @property
    def num_steps(self):
        return self.current_step
