import os
import sys
from typing import Optional

import numpy as np
import schema
import yaml
from numpy import linalg as LA
from decentralized.oracles.base import ArrayPair
from prettytable import PrettyTable
from schema import And, Or, Schema, Use


class LoggerConfigManager(object):
    """
    Instrument for managing configurations for the Logger class.

    Parameters
    ----------
    config_mode: ["default", "optional"]
        Configuration mode.

    general_config_path: Optional[str]
        Relative path to the general configuration file.

    default_config_path: Optional[str]
        Relative path to the default configuration file.

    optional_config_path: Optional[str]
        Relative path to the optional configuration file.

    config: Optional[dict]
        Dictionary containing configuration. If specified, configuration uploads to existing config.
    """

    def __init__(
        self,
        config_mode: str = "default",
        general_config_path: Optional[str] = "./configs/centralized/general_config_logger.yaml",
        default_config_path: Optional[str] = "./configs/centralized/default_config_logger.yaml",
        optional_config_path: Optional[str] = "./configs/centralized/optional_config_logger.yaml",
        config: Optional[dict] = None,
    ):
        if __name__ == "__main__":
            self.path = os.path.split(sys.argv[0])[0]
        else:
            self.path = os.path.split(__file__)[0]
            with open(os.path.join(self.path, general_config_path), "r") as cfg:
                general_cfg = yaml.safe_load(cfg)
            with open(os.path.join(self.path, default_config_path), "r") as cfg:
                default_cfg = yaml.safe_load(cfg)
            with open(os.path.join(self.path, optional_config_path), "r") as cfg:
                optional_cfg = yaml.safe_load(cfg)

        for general_key, general_value in general_cfg.items():
            if general_key == "schema":
                for key, value in general_value.items():
                    d = dict()
                    for k, v in value.items():
                        d[k] = eval(v)
                    setattr(self, key, d)
            else:
                setattr(self, general_key, general_value)

        if config_mode == "default":
            self.config = default_cfg
        elif config_mode == "optional":
            self.config = optional_cfg
        else:
            raise ValueError("Unknown config_mode!")
        self.update_schema()
        self.config_schema.validate(self.config)

        if config is not None:
            self.upload_config(config)

    def norm(self, a: np.ndarray, b: np.ndarray, **kwargs):
        return LA.norm((a - b), **kwargs)

    def pairwise_norm(self, a: np.ndarray, b: np.ndarray, **kwargs):
        dist = 0
        for i in range(a.shape[0]):
            for j in range(i):
                dist += LA.norm((a[i, :] - b[j, :]), **kwargs)

        return dist / (a.shape[0] * (a.shape[0] - 1) / 2)

    def saddle_norm2(self, a: ArrayPair, b: ArrayPair, **kwargs):
        return (a - b).dot(a - b)

    def saddle_grad_norm(self, a: ArrayPair, b: ArrayPair, **kwargs):
        x, y = (a - b).tuple()
        return LA.norm(x.sum(axis=0) / x.shape[0]) ** 2 + LA.norm(y.sum(axis=0) / y.shape[0]) ** 2

    def saddle_con(self, a: ArrayPair, b: ArrayPair, **kwargs):
        return ((a.x - b.x.mean(axis=0)) ** 2).sum() / b.x.shape[0] + (
            (a.y - b.y.mean(axis=0)) ** 2
        ).sum() / b.y.shape[0]

    def update_schema(self):
        self.config_schema = Schema(
            {
                schema.Optional(Or(*self.available_tracks)): {
                    schema.Optional(Or(*self.available_spaces)): {
                        schema.Optional(Or(*self.available_values)): self.value,
                        schema.Optional(Or(*self.available_distances)): self.distance,
                    }
                }
            }
        )

    def upload_config(self, config: dict):
        self.update_schema()
        self.config_schema.validate(config)
        for track_type, track_value in config.items():
            if track_type not in self.config:
                self.config[track_type] = dict()
            for space_type, space_value in track_value.items():
                if space_type not in self.config[track_type]:
                    self.config[track_type][space_type] = dict()
                for value_type, value in space_value.items():
                    self.config[track_type][space_type][value_type] = value

    def register_track(self, track_name: str):
        self.available_tracks.add(track_name)

    def register_space(self, space_name: str):
        self.available_spaces.add(space_name)

    def register_value(self, value_name: str):
        self.available_values.add(value_name)

    def register_distance(self, distance_name: str):
        self.available_distances.add(distance_name)

    def add_distance_function(self, distance_function_name: str, distance_function_config: dict):
        Schema(self.distance_function).validate(distance_function_config)
        self.distance_functions[distance_function_name] = distance_function_config

    def remove_track(self, track_name: str):
        self.available_tracks.remove(track_name)

    def remove_space(self, space_name: str):
        self.available_spaces.remove(space_name)

    def remove_value(self, value_name: str):
        self.available_values.remove(value_name)

    def remove_distance(self, distance_name: str):
        self.available_distances.remove(distance_name)


class Logger(object):
    """
    Instrument for saving the method history during its iterations.

    Parameters
    ----------
    config_mode: ["default", "optional"]
        Configuration mode.

    general_config_path: Optional[str]
        Relative path to the general configuration file.

    default_config_path: Optional[str]
        Relative path to the default configuration file.

    optional_config_path: Optional[str]
        Relative path to the optional configuration file.

    config: Optional[dict]
        Dictionary containing configuration. If specified, configuration uploads to existing config.

    z_true: Optional[np.ndarray]
        Exact solution of the problem/

    g_true: Optional[np.ndarray]
        Gradient of the exact solution of the problem.
    """

    def __init__(
        self,
        config_mode: str = "default",
        general_config_path: Optional[str] = "./configs/centralized/general_config_logger.yaml",
        default_config_path: Optional[str] = "./configs/centralized/default_config_logger.yaml",
        optional_config_path: Optional[str] = "./configs/centralized/optional_config_logger.yaml",
        config: Optional[dict] = None,
        z_true: Optional[np.ndarray] = None,
        g_true: Optional[np.ndarray] = None,
    ):
        self.current_step = 0
        self.config_manager = LoggerConfigManager(
            config_mode=config_mode,
            general_config_path=general_config_path,
            default_config_path=default_config_path,
            optional_config_path=optional_config_path,
            config=config,
        )

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
        self.z_true = z_true
        self.g_true = g_true

    def start(self, method):
        pass

    def log_value(self, prefix: str, value: dict, src: object):
        if self.current_step % getattr(self, prefix + "_step") == 0:
            data = getattr(src, value["source"])
            getattr(self, prefix).append(data)
            if getattr(self, prefix + "_verbose"):
                self.output_row.append(data)

    def log_distance(self, prefix: str, cfg: dict, method: object):
        if self.current_step % getattr(self, prefix + "_step") == 0:
            source = getattr(method, cfg["source"])
            if cfg["target"].split("_")[-1] == "true":
                target = getattr(self, cfg["target"])
            else:
                target = getattr(method, cfg["target"])
            distance_config = self.config_manager.distance_functions[cfg["distance"]]
            func = getattr(eval(distance_config["object"]), distance_config["name"])
            distance = func(source, target, **distance_config["kwargs"])
            getattr(self, prefix).append(distance)
            if getattr(self, prefix + "_verbose"):
                self.output_row.append(distance)

    def step(self, method):
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
                        self.log_value(prefix, value, method)
                    elif value_type in self.config_manager.available_distances:
                        self.log_distance(prefix, value, method)
                    else:
                        raise ValueError("Unknown value type!")

        self.current_step += 1

    def end(self, method):
        if len(self.tracked_values) > 0:
            output_str = "".join(
                [el + ", " if i + 1 < len(self.tracked_values) else el for i, el in enumerate(self.tracked_values)]
            )
            print(output_str + " can be accessed at corresponding logger class fields.")

    @property
    def num_steps(self):
        return self.current_step


class LoggerDecentralized(Logger):
    """
    Instrument for saving method history during its iterations for decentralized methods.
    Additionally logs distance to consensus.

    Parameters
    ----------
    config_mode: ["default", "optional"]
        Configuration mode.

    general_config_path: Optional[str]
        Relative path to the general configuration file.

    default_config_path: Optional[str]
        Relative path to the default configuration file.

    optional_config_path: Optional[str]
        Relative path to the optional configuration file.

    config: Optional[dict]
        Dictionary containing configuration. If specified, configuration uploads to existing config.

    z_true: Optional[np.ndarray]
        Exact solution of the problem. If specified, logs distance to solution.

    g_true: Optional[np.ndarray]
        Gradient of the exact solution of the problem.
    """

    def __init__(
        self,
        config_mode: str = "default",
        general_config_path: Optional[str] = "./configs/decentralized/general_config_logger.yaml",
        default_config_path: Optional[str] = "./configs/decentralized/default_config_logger.yaml",
        optional_config_path: Optional[str] = "./configs/decentralized/optional_config_logger.yaml",
        config: Optional[dict] = None,
        z_true: Optional[np.ndarray] = None,
        g_true: Optional[np.ndarray] = None,
    ):
        super().__init__(
            config_mode=config_mode,
            general_config_path=general_config_path,
            default_config_path=default_config_path,
            optional_config_path=optional_config_path,
            config=config,
            z_true=z_true,
            g_true=g_true,
        )

    def log_value(self, prefix: str, value: dict, src: object):
        if self.current_step % getattr(self, prefix + "_step") == 0:
            data = getattr(src, value["source"])
            getattr(self, prefix).append(data[value["nodes"]])
            if getattr(self, prefix + "_verbose"):
                self.output_row.append(data)

    def log_distance(self, prefix: str, cfg: dict, method: object):
        if self.current_step % getattr(self, prefix + "_step") == 0:
            source = getattr(method, cfg["source"])
            if cfg["target"].split("_")[-1] == "true":
                target = getattr(self, cfg["target"])
            else:
                target = getattr(method, cfg["target"])
            distance_config = self.config_manager.distance_functions[cfg["distance"]]
            func = getattr(eval(distance_config["object"]), distance_config["name"])
            distance = func(source, target, **distance_config["kwargs"])
            getattr(self, prefix).append(distance)
            if getattr(self, prefix + "_verbose"):
                self.output_row.append(distance)
