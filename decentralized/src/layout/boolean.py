from typing import Optional

import ipywidgets
from ipywidgets import HBox, Label
from src.layout.base import BaseLayout


class Boolean(BaseLayout):
    available_types = {"ToggleButton", "Checkbox"}

    def __init__(
        self,
        type: str,
        label: str,
        value: bool,
        disabled: Optional[bool] = False,
        indent: Optional[bool] = False,
        r_label: Optional[str] = None,
    ):
        if type not in Boolean.available_types:
            raise ValueError(
                f"Unknown Boolean type: {type}! Available Boolean types: {Boolean.available_types}."
            )

        if r_label is not None:
            self._label = r_label
        else:
            self._label = label

        boolean = getattr(ipywidgets, type)
        self.widget = HBox(
            [
                Label(value=label),
                boolean(value=value, disabled=disabled, indent=indent),
            ]
        )

    @property
    def value(self):
        self._value = self.widget.children[1].value
        return self._value

    @property
    def label(self):
        return self._label
