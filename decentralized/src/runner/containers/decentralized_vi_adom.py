from src.layout.boolean import Boolean
from src.layout.container import Container
from src.layout.numeric import NumericText
from src.layout.selection import Select

VIADOM_container = Container(
    "VBox",
    [
        Container(
            "VBox",
            [
                NumericText(
                    "BoundedFloatText",
                    "L",
                    0.001,
                    0,
                    100000,
                    0.000001,
                ),
                NumericText(
                    "BoundedFloatText",
                    r"$\bar{L}$",
                    0.001,
                    0,
                    100000,
                    0.000001,
                    r_label="L_avg",
                ),
                NumericText(
                    "BoundedFloatText", "$\mu$", 0.01, 0, 10, 0.000001, r_label="mu"
                ),
                Boolean(
                    "Checkbox",
                    "Use precomputed parameters",
                    True,
                    r_label="use_precomputed",
                ),
                NumericText("BoundedIntText", "Batch size", 1, 1, 100, 1, r_label="b"),
            ],
            "Parameters:",
        ),
        Container(
            "VBox",
            [
                Select(
                    "ToggleButtons",
                    "$x_0$",
                    "Zero",
                    ["Zero", "Simplex", "Uniform"],
                    r_label="x_0",
                ),
                Select("ToggleButtons", "$y_0$", "Zero", ["Zero"], r_label="y_0"),
                Select("ToggleButtons", "$z_0$", "Zero", ["Zero"], r_label="z_0"),
            ],
            "Initial guess:",
        ),
        Boolean(
            "Checkbox", "Use this method in experiment", True, r_label="use_method"
        ),
    ],
)
