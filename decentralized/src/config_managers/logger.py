import os
import sys
from typing import Dict, List, Optional

import numpy as np
import schema
import yaml
from definitions import ROOT_DIR
from numpy import linalg as LA
from schema import And, Or, Schema, Use
from src.oracles.base import ArrayPair


class LoggerConfigManager(object):
    """
    Instrument for managing configurations for the Logger class.
    """

    def __init__(
        self,
        config_path: Optional[
            str
        ] = "src/config_managers/configs/default/logger_centralized.yaml",
        general_config_path: Optional[
            str
        ] = "src/config_managers/configs/general/logger_centralized.yaml",
    ):
        """
        :param config_path: Relative path to the logger configuration file.
        :param general_config_path: Relative path to the general configuration file.
        """
        self.general_config_path = os.path.join(ROOT_DIR, general_config_path)
        self.config_path = os.path.join(ROOT_DIR, config_path)
        with open(self.general_config_path, "r") as cfg:
            self.general_cfg = yaml.unsafe_load(cfg)

        for general_key, general_value in self.general_cfg.items():
            if general_key == "schema":
                for key, value in general_value.items():
                    d = dict()
                    for k, v in value.items():
                        d[k] = eval(v)
                    setattr(self, key, d)
            else:
                setattr(self, general_key, general_value)

        self.update_schema()
        self.config_schema.validate(self.config)

    @property
    def config(self):
        with open(self.config_path, "r") as cfg:
            config = yaml.unsafe_load(cfg)
        self.config_schema.validate(config)
        self._config = config
        return self._config

    @config.setter
    def config(self, new_config: Dict):
        self.config_schema.validate(new_config)
        self._config = new_config
        with open(self.config_path, "w") as cfg:
            yaml.dump(new_config, cfg)

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
        return (
            LA.norm(x.sum(axis=0) / x.shape[0]) ** 2
            + LA.norm(y.sum(axis=0) / y.shape[0]) ** 2
        )

    def saddle_con(self, a: ArrayPair, b: ArrayPair, **kwargs):
        return ((a.x - b.x.mean(axis=0)) ** 2).sum() / b.x.shape[0] + (
            (a.y - b.y.mean(axis=0)) ** 2
        ).sum() / b.y.shape[0]
