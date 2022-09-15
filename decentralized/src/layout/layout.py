from bisect import bisect_left
from copy import copy
from typing import List, Optional

import ipywidgets
from ipywidgets import Dropdown, Stack, VBox
from src.layout.base import BaseLayout
from src.layout.container import Container
from src.layout.numeric import NumericText, Slider
from src.layout.selection import Select
from src.layout.string import Button


class Layout(BaseLayout):
    available_types = {"Accordion", "Tab"}

    def __init__(
        self,
        type: str,
        titles: List[str],
        widgets: List[BaseLayout],
        r_label: Optional[str] = None,
    ):
        if type not in Layout.available_types:
            raise ValueError(
                f"Unknown Layout type: {type}! Available Layout types: {Layout.available_types}."
            )

        if len(titles) != len(widgets):
            raise ValueError(
                f"Length of titles ({len(titles)}) must be equal to length of widgets ({len(widgets)})!"
            )

        self._label = r_label

        self.widgets = widgets

        layout = getattr(ipywidgets, type)

        self.widget = layout(children=[w.widget for w in widgets], titles=titles)

    @property
    def value(self):
        self._value = {}
        for widget, title in zip(self.widgets, self.widget.titles):
            if hasattr(widget, "value"):
                if widget.label is not None:
                    self._value[widget.label] = widget.value
                else:
                    self._value[title] = widget.value
        return self._value

    @property
    def label(self):
        return self._label

    def convert_str(x: str):
        try:
            return int(x)
        except:
            return float("-inf")

    def insert_tab(
        self, tab_title: str, tab: BaseLayout, replace: Optional[bool] = False
    ):
        """
        Insert tabs with integer title in sorted by titles array.
        Warning! The search doesn't work if string that can't be converted to integer comes after that can be converted.

        :param tab_title: Title of inserted tab.
        :param tab: Tab.
        :param replace: Whether inserted tab should be replaced. If not and the title is found then ValueError is raised.
        """

        insert_at = bisect_left(
            self.widget.titles, int(tab_title), key=Layout.convert_str
        )

        if (
            insert_at != len(self.widgets)
            and self.widget.titles[insert_at] == tab_title
            and not replace
        ):
            raise ValueError("If you want replace tab put `replace` flag to True!")

        self.widgets[insert_at:insert_at] = [tab]

        titles = list(self.widget.titles)
        children = list(self.widget.children)

        titles[insert_at:insert_at] = [tab_title]
        children[insert_at:insert_at] = [tab.widget]

        self.widget.children = children
        self.widget.titles = titles

    def delete_tab(self, tab_title: str):
        delete_at = bisect_left(
            self.widget.titles, int(tab_title), key=Layout.convert_str
        )
        if (
            delete_at != len(self.widgets)
            and self.widget.titles[delete_at] == tab_title
        ):
            del self.widgets[delete_at]

            titles = list(self.widget.titles)
            children = list(self.widget.children)

            del titles[delete_at]
            del children[delete_at]

            self.widget.children = children
            self.widget.titles = titles

        raise ValueError(f"Tab {tab_title} isn't it tabs!")

    def __len__(self):
        return len(self.widgets)


class DropdownStacked(BaseLayout):
    def __init__(
        self,
        options: List[str],
        widgets: List[BaseLayout],
        r_label: Optional[str] = None,
    ):
        if len(options) != len(widgets):
            raise ValueError(
                f"Length of options ({len(options)}) must be equal to length of widgets ({len(widgets)})!"
            )

        self._label = r_label
        self.widgets = widgets

        stacked = Stack([w.widget for w in widgets])
        dropdown = Dropdown(options=options)
        ipywidgets.jslink((dropdown, "index"), (stacked, "selected_index"))

        self.widget = VBox([dropdown, stacked])

    @property
    def value(self):
        selected_index = self.widget.children[1].selected_index
        self._value = self.widgets[selected_index].value
        return self._value

    @property
    def label(self):
        return self._label


class StateTabs(BaseLayout):
    def __init__(
        self,
        default_state_layout: BaseLayout,
        state_layout: BaseLayout,
        states_number: int,
        r_label: Optional[str] = None,
    ):
        self._label = r_label

        states = Layout("Tab", ["Default state"], [copy(default_state_layout)])

        states_for_configuration = [str(i) for i in range(states_number)]
        dropdown = Select(
            "Dropdown",
            "State:",
            states_for_configuration[0],
            states_for_configuration,
        )

        def add_state(b: ipywidgets.widgets.Button):
            states.insert_tab(dropdown.value, copy(state_layout))

            delete_from = bisect_left(
                states_for_configuration, int(dropdown.value), key=lambda x: int(x)
            )
            del states_for_configuration[delete_from]
            dropdown.widget.children[1].options = states_for_configuration

        button = Button("Configurate", on_click=add_state)

        self.layout = Container(
            "VBox",
            [
                states,
                Container("HBox", [dropdown, button]),
            ],
        )
        self.widget = self.layout.widget

    @property
    def value(self):
        return self.layout.value

    @property
    def label(self):
        return self._label


class EpochTabs(BaseLayout):
    def __init__(
        self,
        state_tabs: StateTabs,
        states_number: int,
        r_label: Optional[str] = None,
    ):
        self._label = r_label

        epochs = Layout(
            "Tab",
            ["Default epoch"],
            [
                Container(
                    "VBox",
                    [
                        NumericText(
                            "BoundedIntText",
                            "Duration:",
                            1,
                            1,
                            states_number,
                            1,
                            r_label="duration",
                        ),
                        copy(state_tabs),
                    ],
                    r_label="default_epoch",
                )
            ],
        )

        def add_epoch(b: ipywidgets.widgets.Button):
            def on_duration_change_max(change):
                epochs.widgets[-2].widget.children[0].children[1].max = change["new"][0]

            def on_duration_change_min(change):
                epochs.widgets[-1].widget.children[0].children[1].min = change["new"][1]

            if len(epochs) == 1:
                epochs.insert_tab(
                    str(len(epochs) - 1),
                    Container(
                        "VBox",
                        [
                            Slider(
                                "IntRangeSlider",
                                "States",
                                [1, states_number],
                                1,
                                states_number,
                                1,
                                readout_format="d",
                                r_label="interval",
                            ),
                            copy(state_tabs),
                        ],
                    ),
                )
            else:
                max_value = epochs.widgets[-1].value["interval"][1]
                if max_value < states_number:
                    epochs.insert_tab(
                        str(len(epochs) - 1),
                        Container(
                            "VBox",
                            [
                                Slider(
                                    "IntRangeSlider",
                                    "States",
                                    [max_value, states_number],
                                    max_value,
                                    states_number,
                                    1,
                                    readout_format="d",
                                    r_label="interval",
                                ),
                                copy(state_tabs),
                            ],
                        ),
                    )
                    epochs.widgets[-1].widget.children[0].children[1].observe(
                        on_duration_change_max, names="value"
                    )
                    epochs.widgets[-2].widget.children[0].children[1].observe(
                        on_duration_change_min, names="value"
                    )

        button = Button("Add epoch", on_click=add_epoch)

        self.layout = Container("VBox", [epochs, button])
        self.widget = self.layout.widget

    @property
    def value(self):
        return self.layout.value

    @property
    def label(self):
        return self._label
