from typing import Optional

import ipywidgets
from ipywidgets import HBox, Label
from src.layout.base import BaseLayout


class Numeric(BaseLayout):
    def __init__(
        self,
        type: str,
        label: str,
        value: int,
        min: int,
        max: int,
        step: int,
        disabled: Optional[bool] = False,
        continuous_update: Optional[bool] = False,
        orientation: Optional[str] = "horizontal",
        readout: Optional[bool] = True,
        readout_format: Optional[str] = ".2f",
        r_label: Optional[str] = None,
    ):
        """
        :param type: Ipywidget class name.
        :param label: Description.
        :param value: Initial value.
        :param min: Lower bound.
        :param max: Upper bound.
        :param step: Value can be incremented according to this parameter.
        :param disabled: Whether to disable user changes.
        :param continuous_update: Restricts executions to mouse release events.
        :param orientation: The slider's orientation is either 'horizontal' (default) or 'vertical'.
        :param readout: Displays the current value of the slider next to it.
        :param readout_format: Specifies the format function used to represent slider value. The default is '.2f'.
        :param r_label: Returning widget label.
        """
        if r_label is not None:
            self._label = r_label
        else:
            self._label = label

        numeric = getattr(ipywidgets, type)

        self.widget = HBox(
            [
                Label(value=label),
                numeric(
                    value=value,
                    min=min,
                    max=max,
                    step=step,
                    disabled=disabled,
                    continuous_update=continuous_update,
                    orientation=orientation,
                    readout=readout,
                    readout_format=readout_format,
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


class Slider(Numeric):
    available_types = {
        "IntSlider",
        "FloatSlider",
        "FloatLogSlider",
        "IntRangeSlider",
        "FloatRangeSlider",
    }

    def __init__(
        self,
        type: str,
        label: str,
        value: int,
        min: int,
        max: int,
        step: int,
        disabled: Optional[bool] = False,
        continuous_update: Optional[bool] = False,
        orientation: Optional[str] = "horizontal",
        readout: Optional[bool] = True,
        readout_format: Optional[str] = ".2f",
        r_label: Optional[str] = None,
    ):
        if type not in Slider.available_types:
            raise ValueError(
                f"Unknown Slider type: {type}! Available Slider types: {Slider.available_types}."
            )

        super().__init__(
            type=type,
            label=label,
            value=value,
            min=min,
            max=max,
            step=step,
            disabled=disabled,
            continuous_update=continuous_update,
            orientation=orientation,
            readout=readout,
            readout_format=readout_format,
            r_label=r_label,
        )


class NumericText(Numeric):
    available_types = {"BoundedIntText", "BoundedFloatText", "IntText", "FloatText"}

    def __init__(
        self,
        type: str,
        label: str,
        value: int,
        min: int,
        max: int,
        step: int,
        disabled: Optional[bool] = False,
        continuous_update: Optional[bool] = False,
        orientation: Optional[str] = "horizontal",
        readout: Optional[bool] = True,
        readout_format: Optional[str] = ".2f",
        r_label: Optional[str] = None,
    ):
        if type not in NumericText.available_types:
            raise ValueError(
                f"Unknown NumericText type: {type}! Available NumericText types: {NumericText.available_types}"
            )

        super().__init__(
            type=type,
            label=label,
            value=value,
            min=min,
            max=max,
            step=step,
            disabled=disabled,
            continuous_update=continuous_update,
            orientation=orientation,
            readout=readout,
            readout_format=readout_format,
            r_label=r_label,
        )
