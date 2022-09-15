from typing import List, Optional

import ipywidgets
from ipywidgets import HBox, Label
from src.layout.base import BaseLayout


class Tags(BaseLayout):
    available_types = {"TagsInput", "ColorsInput"}

    def __init__(
        self,
        type: str,
        label: str,
        value: List[str],
        allowed_tags: List[str],
        allow_duplicates: Optional[bool] = False,
        tag_style: Optional[str] = "primary",
        r_label: Optional[str] = None,
    ):
        """
        The `Tags` class is useful to for selecting/creating a list of tags.
        You can drag and drop tags to reorder them, limit them to a set of allowed values,
        or even prevent making duplicate tags.

        :param type: Ipywidget class name.
        :param label: Description.
        :param value: Initial value.
        :param allowed_tags: Allowed tags.
        :param allow_duplicates: Whether duplicates are allowed.
        :param tag_style: Tag style.
        :param r_label: Returning widget label.
        """
        if type not in Tags.available_types:
            raise ValueError(
                f"Unknown Tags type: {type}! Available Tags types: {Tags.available_types}."
            )

        if r_label is not None:
            self._label = r_label
        else:
            self._label = label
        tags = getattr(ipywidgets, type)

        self.widget = HBox(
            [
                Label(value=label),
                tags(
                    value=value,
                    allowed_tags=allowed_tags,
                    allow_duplicates=allow_duplicates,
                    tag_style=tag_style,
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
