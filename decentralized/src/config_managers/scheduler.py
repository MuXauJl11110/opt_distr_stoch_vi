import os
from typing import Dict, Optional

import schema
import yaml
from definitions import ROOT_DIR
from schema import And, Or, Schema, Use


class SchedulerConfigManager(object):
    """
    Instrument for managing configurations for the Scheduler class.
    """

    def __init__(
        self,
        config_path: Optional[str] = "src/config_managers/configs/default/scheduler.yaml",
        general_config_path: Optional[str] = "src/config_managers/configs/general/scheduler.yaml",
    ):
        """
        :param config_path: Relative path to the scheduler configuration file.
        :param general_config_path: Relative path to the general configuration file.
        """
        self.general_config_path = os.path.join(ROOT_DIR, general_config_path)
        self.config_path = os.path.join(ROOT_DIR, config_path)

        with open(self.general_config_path, "r") as cfg:
            self.general_cfg = yaml.unsafe_load(cfg)

        for _type in self.general_cfg["schema"]:
            d = dict()
            for k, v in _type[1].items():
                d[k] = eval(v)
            setattr(self, "schema_" + _type[0], d)

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
