import os
from typing import Dict, Optional

import schema
import yaml
from definitions import ROOT_DIR
from schema import And, Or, Schema, Use


class NetworkConfigManager(object):
    """
    Instrument for managing configuration for the Network class.
    """

    def __init__(
        self,
        config_path: Optional[str] = "src/network/configs/default_config.yaml",
        general_config_path: Optional[str] = "src/network/configs/general_config.yaml",
    ):
        """
        :param config_path: Relative path to the network configuration file.
        :param general_config_path: Relative path to the general configuration file.
        """
        self.general_config_path = os.path.join(ROOT_DIR, general_config_path)
        self.config_path = os.path.join(ROOT_DIR, config_path)
        with open(self.general_config_path, "r") as cfg:
            self.general_cfg = yaml.unsafe_load(cfg)

        def make_schema_obj(d: dict):
            new_d = dict()
            for k, v in d.items():
                if isinstance(v, str):
                    if v in self.general_cfg["available_graph_types"]:
                        new_d[k] = v
                    else:
                        new_d[k] = eval(v)
                elif isinstance(v, dict):
                    new_d[k] = make_schema_obj(v)
                else:
                    raise ValueError("Unknown schema!")
            return new_d

        self.schema_types = [
            make_schema_obj({**_type, **self.general_cfg["schema"][0][0]})
            for _type in self.general_cfg["schema"][0][1]
        ]
        for _type in self.general_cfg["schema"][1:]:
            if isinstance(_type[1], str):
                setattr(self, "schema_" + _type[0], eval(_type[1]))
            elif isinstance(_type[1], dict):
                setattr(self, "schema_" + _type[0], make_schema_obj(_type[1]))
            else:
                raise ValueError(f"Forbidden value: {_type[1]}!")

        self.config_schema = Schema(
            {
                "default_epoch": self.schema_default_epoch,
                schema.Optional("epochs"): [self.schema_epoch],
            }
        )

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
