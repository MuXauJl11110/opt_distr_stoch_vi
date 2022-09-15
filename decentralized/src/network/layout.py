from typing import Optional

import ipywidgets
from src.config_managers.network import NetworkConfigManager
from src.layout.boolean import Boolean
from src.layout.container import Container
from src.layout.layout import DropdownStacked, EpochTabs, Layout, StateTabs
from src.layout.numeric import NumericText
from src.layout.output import Tags
from src.layout.selection import Select


class NetworkLayout(object):
    """
    Instrument for displaying network widget.
    """

    def __init__(
        self,
        network_cm: Optional[NetworkConfigManager] = NetworkConfigManager(),
    ):
        """
        :param network_cm: Network config manager.
        """
        self.network_cm = network_cm
        self.layout = self.initialize_layout()

        self.out = ipywidgets.Output()

    def initialize_layout(self):
        states_number = Tags(
            "TagsInput",
            "States number:",
            ["50"],
            ["1", "5", "10", "25", "50", "100", "250", "500"],
        )
        nodes_number = Tags(
            "TagsInput",
            "Nodes number:",
            ["25"],
            [str(i) for i in range(10, 101, 5)],
        )
        self.general = Container(
            "VBox",
            [
                states_number,
                nodes_number,
            ],
        )
        state_widgets = [
            Boolean("Checkbox", "Fixed topology", False, r_label="fixed"),
            Boolean("Checkbox", "Directed", False, r_label="directed"),
            Boolean("Checkbox", "MST", False, r_label="MST"),
            Boolean("Checkbox", "Plot", False, r_label="plot"),
            DropdownStacked(
                ["classic", "random"],
                [
                    Container(
                        "VBox",
                        [
                            Select(
                                "ToggleButtons",
                                "Topology:",
                                self.network_cm.general_cfg["available_topologies"][
                                    "classic"
                                ][2],
                                self.network_cm.general_cfg["available_topologies"][
                                    "classic"
                                ],
                                r_label="topology",
                            ),
                            Container("VBox", [], r_label="args"),
                        ],
                    ),
                    Container(
                        "VBox",
                        [
                            Select(
                                "ToggleButtons",
                                "Topology:",
                                self.network_cm.general_cfg["available_topologies"][
                                    "random"
                                ][0],
                                self.network_cm.general_cfg["available_topologies"][
                                    "random"
                                ],
                                r_label="topology",
                            ),
                            Container(
                                "VBox",
                                [
                                    NumericText(
                                        "BoundedFloatText",
                                        "p:",
                                        0.9,
                                        0,
                                        1,
                                        0.00001,
                                        r_label="p",
                                    ),
                                    NumericText(
                                        "BoundedIntText",
                                        "seed:",
                                        30,
                                        0,
                                        100,
                                        1,
                                        r_label="seed",
                                    ),
                                ],
                                r_label="args",
                            ),
                        ],
                    ),
                ],
            ),
        ]
        default_state_layout = Container(
            "VBox", [w for w in state_widgets], r_label="default_state"
        )
        state_layout = Container(
            "VBox",
            [w for w in state_widgets],
        )

        general_states_tabs_layout = Layout(
            "Tab",
            self.general.value["States number:"],
            [
                EpochTabs(StateTabs(default_state_layout, state_layout, int(s)), int(s))
                for s in self.general.value["States number:"]
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

        states_number.widget.children[1].observe(on_states_number_change, names="value")

        return Container(
            "VBox",
            [self.general, general_states_tabs_layout],
            "States:",
            r_label="network",
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

        with self.out:
            print(config)
        self.network_cm.config = config
