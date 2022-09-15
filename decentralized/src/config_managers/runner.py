import os
from typing import Dict, Optional, Union

import numpy as np
import yaml
from definitions import ROOT_DIR


class RunnerConfigManager(object):
    """
    Instrument for managing runner configuration.
    """

    def __init__(
        self,
        config_path: Optional[str] = "src/config_managers/configs/default/runner.yaml",
    ):
        """
        :param config_path: Path to the runner configuration file.
        """
        self.config_path = os.path.join(ROOT_DIR, config_path)

    @property
    def config(self):
        with open(self.config_path, "rb") as cfg:
            config = yaml.load(cfg)
        self._config = config
        return self._config

    @config.setter
    def config(
        self,
        new_config: Dict[str, Union[str, np.ndarray]],
    ):
        self._config = new_config
        with open(self.config_path, "wb") as cfg:
            yaml.dump(new_config, cfg)
