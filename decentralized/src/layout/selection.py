from typing import Callable, List, Optional

import ipywidgets
from ipywidgets import HBox, Label
from src.layout.base import BaseLayout


class Select(BaseLayout):
    available_types = {"Dropdown", "RadioButtons", "Select", "ToggleButtons"}

    def __init__(
        self,
        type: str,
        label: str,
        value: str,
        options: List[str],
        disabled: Optional[bool] = False,
        readout_func: Optional[Callable[[str], object]] = None,
        r_label: Optional[str] = None,
    ):
        """
        :param type: Ipywidget class name.
        :param label: Description.
        :param value: Initial value.
        :param options: Options.
        :param disabled: Whether to disable user changes.
        :param readout_func: Function that applies to the chosen option.
        :param r_label: Returning widget label.
        """
        if type not in Select.available_types:
            raise ValueError(
                f"Unknown Select type: {type}! Available Select types: {Select.available_types}."
            )

        if r_label is not None:
            self._label = r_label
        else:
            self._label = label
        self.readout_func = readout_func

        select = getattr(ipywidgets, type)
        self.widget = HBox(
            [
                Label(value=label),
                select(
                    options=options,
                    value=value,
                    disabled=disabled,
                ),
            ]
        )

    @property
    def value(self):
        if self.readout_func is not None:
            self._value = self.readout_func(self.widget.children[1].value)
        else:
            self._value = self.widget.children[1].value
        return self._value

    @property
    def label(self):
        return self._label
