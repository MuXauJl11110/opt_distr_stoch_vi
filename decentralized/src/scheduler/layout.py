import sys
from typing import Optional

from src.config_managers.scheduler import SchedulerConfigManager
from src.layout.base import BaseLayout
from src.layout.boolean import Boolean
from src.layout.container import Container
from src.layout.layout import EpochTabs, Layout, StateTabs
from src.layout.numeric import NumericText


class SchedulerLayout(object):
    """
    Instrument for displaying scheduler widget.
    """

    def __init__(
        self,
        network_general: BaseLayout,
        scheduler_cm: Optional[SchedulerConfigManager] = SchedulerConfigManager(),
    ):
        """
        :param network_general: General config of network.
        :param scheduler_cm: Scheduler config manager.
        """
        self.network_general = network_general
        self.scheduler_cm = scheduler_cm

        self.layout = self.initialize_layout()

    def initialize_layout(self):
        default_state_layout = Container(
            "VBox",
            [
                NumericText(
                    "BoundedIntText",
                    "Steps per state:",
                    1,
                    1,
                    sys.maxsize,
                    1,
                    r_label="steps_per_state",
                ),
                Boolean("Checkbox", "Verbose", False, r_label="verbose"),
            ],
            r_label="default_state",
        )
        state_layout = Container(
            "VBox",
            [
                NumericText(
                    "BoundedIntText",
                    "Steps per state:",
                    1,
                    1,
                    sys.maxsize,
                    1,
                    r_label="steps_per_state",
                ),
                Boolean("Checkbox", "Verbose", False, r_label="verbose"),
            ],
        )
        general_states_tabs_layout = Layout(
            "Tab",
            self.network_general.value["States number:"],
            [
                EpochTabs(StateTabs(default_state_layout, state_layout, int(s)), int(s))
                for s in self.network_general.value["States number:"]
            ],
        )

        def on_states_number_change(change):
            old_states = set(general_states_tabs_layout.widget.titles)
            new_states = set(change["new"])
            states = old_states.union(new_states)

            for state in states:
                if state not in old_states and state in new_states:
                    general_states_tabs_layout.insert_tab(
                        state,
                        EpochTabs(
                            StateTabs(default_state_layout, state_layout, int(state)),
                            int(state),
                        ),
                    )
                elif state in old_states and state not in new_states:
                    general_states_tabs_layout.delete_tab(state)

        self.network_general.widgets[0].widget.children[1].observe(
            on_states_number_change, names="value"
        )

        return Container(
            "VBox", [general_states_tabs_layout], "States:", r_label="scheduler"
        )

    def update_config(self, states_number: str):
        stop_list = set(["State:", "duration", "default_epoch", "default_state"])
        raw_cfg = self.layout.value[states_number]

        def parse_epoch(epoch_cfg: dict):
            if "duration" in epoch_cfg:
                duration_cfg = {"duration": epoch_cfg["duration"]}
            elif "interval" in epoch_cfg:
                duration_cfg = {"interval": epoch_cfg["interval"]}
            else:
                raise ValueError("Unknown duration parameter!")
            return {
                **duration_cfg,
                **{
                    "default_state": epoch_cfg["default_state"],
                    "states": {
                        key: value
                        for key, value in epoch_cfg.items()
                        if key not in stop_list
                    },
                },
            }

        config = {"default_epoch": parse_epoch(raw_cfg["default_epoch"]), "epochs": []}

        epochs = list(raw_cfg.keys() - stop_list)
        epochs.sort()
        for key in epochs:
            config["epochs"].append(parse_epoch(raw_cfg[key]))

        self.scheduler_cm.config = config
