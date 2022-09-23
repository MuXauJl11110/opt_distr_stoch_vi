from typing import List, Optional

import ipywidgets
from ipywidgets import Label
from src.layout.base import BaseLayout


class Container(BaseLayout):
    available_types = {"Box", "HBox", "VBox", "GridBox"}

    def __init__(
        self,
        type: str,
        widgets: List[BaseLayout],
        label: Optional[str] = None,
        r_label: Optional[str] = None,
    ):
        if type not in Container.available_types:
            raise ValueError(
                f"Unknown Container type: {type}! Available Container types: {Container.available_types}."
            )

        if r_label is not None:
            self._label = r_label
        else:
            self._label = label

        self.widgets = widgets

        container = getattr(ipywidgets, type)

        self.widget = (
            container([Label(value=label)]) if label is not None else container()
        )
        self.widget.children += tuple([w.widget for w in widgets])

    @property
    def value(self):
        self._value = {}
        for widget in self.widgets:
            if hasattr(widget, "value"):
                if widget.label is not None:
                    self._value[widget.label] = widget.value
                else:
                    self._value = {**self._value, **widget.value}
        return self._value

    @property
    def label(self):
        return self._label
