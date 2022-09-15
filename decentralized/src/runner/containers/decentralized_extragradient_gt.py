from src.layout.boolean import Boolean
from src.layout.container import Container
from src.layout.numeric import NumericText
from src.layout.selection import Select

EG_GT_container = Container(
    "VBox",
    [
        Container(
            "VBox",
            [
                NumericText("BoundedFloatText", "L", 0.001, 0, 10, 0.000001),
                NumericText(
                    "BoundedFloatText", "$\mu$", 0.01, 0, 10, 0.000001, r_label="mu"
                ),
                NumericText(
                    "BoundedFloatText",
                    "$\gamma$",
                    0.01,
                    0,
                    10,
                    0.000001,
                    r_label="gamma",
                ),
                Boolean(
                    "Checkbox",
                    "Use precomputed parameters",
                    True,
                    r_label="use_precomputed",
                ),
            ],
            "Parameters:",
        ),
        Container(
            "VBox",
            [
                Select("ToggleButtons", "$z_0$", "Zero", ["Zero"], r_label="z_0"),
            ],
            "Initial guess:",
        ),
        Boolean(
            "Checkbox", "Use this method in experiment", True, r_label="use_method"
        ),
    ],
)
