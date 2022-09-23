from typing import Callable, Optional

import ipywidgets
from ipywidgets import HBox, Label
from src.layout.base import BaseLayout


class Button:
    def __init__(
        self,
        description: str,
        disabled: Optional[bool] = False,
        button_style: Optional[
            str
        ] = "",  # 'success', 'info', 'warning', 'danger' or ''
        tooltip: Optional[str] = "Click me",
        icon: Optional[str] = "",  # (FontAwesome names without the `fa-` prefix))
        on_click: Optional[Callable] = None,
    ):
        self.widget = ipywidgets.Button(
            description=description,
            disabled=disabled,
            button_style=button_style,
            tooltip=tooltip,
            icon=icon,
        )
        if on_click is not None:
            self.widget.on_click(on_click)

    def on_click(self, func: Callable):
        self.widget.on_click(func)


class String(BaseLayout):
    available_types = {"Text", "Textarea"}

    def __init__(
        self,
        type: str,
        label: str,
        value: str,
        placeholder: str,
        disabled: Optional[bool] = False,
        r_label: Optional[str] = None,
    ):
        if type not in String.available_types:
            raise ValueError(
                f"Unknown String type: {type}! Available String types: {String.available_types}."
            )

        if r_label is not None:
            self._label = r_label
        else:
            self._label = label

        string = getattr(ipywidgets, type)
        self.widget = HBox(
            [
                Label(value=label),
                string(
                    value=value,
                    placeholder=placeholder,
                    disabled=disabled,
                ),
            ]
        )

    @property
    def value(self):
        self._value = self.widget.children[1].value
        return self._value

    @property
    def label(self):
        return self._label
