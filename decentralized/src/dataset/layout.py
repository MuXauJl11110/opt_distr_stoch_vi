from typing import Optional

from src.config_managers.dataset import DatasetConfigManager
from src.dataset.containers.synthetic import synthetic_container
from src.layout.layout import Layout


class DatasetLayout(object):
    """
    Instrument for displaying dataset widget.
    """

    def __init__(
        self, dataset_cm: Optional[DatasetConfigManager] = DatasetConfigManager()
    ):
        """
        :param general_cm: Runner config manager.
        """
        self.dataset_cm = dataset_cm
        self.layout = self.initialize_layout()

    def initialize_layout(self):
        return Layout(
            "Tab",
            ["Synthetic"],
            [synthetic_container],
        )

    def __iter__(self):
        cfg = self.layout.value
        for noise in cfg["synthetic"]["noise"]:
            self.dataset_cm.config = self.layout.value
            yield ("synthetic", cfg["synthetic"] | {"noise": float(noise)})
