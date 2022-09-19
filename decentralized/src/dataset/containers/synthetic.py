from src.layout.container import Container
from src.layout.numeric import NumericText
from src.layout.output import Tags

synthetic_container = Container(
    "VBox",
    [
        NumericText(
            "BoundedIntText", "Number of nodes", 25, 1, 100, 1, r_label="num_nodes"
        ),
        NumericText("BoundedFloatText", "Mean", 5.0, 2.5, 10, 0.1, r_label="mean"),
        NumericText("BoundedFloatText", "Std", 2.0, 1.0, 10, 0.1, r_label="std"),
        NumericText("BoundedIntText", "Number of rows", 30, 1, 100, 1, r_label="l"),
        NumericText("BoundedIntText", "Ambient dimension", 20, 1, 100, 1, r_label="d"),
        Tags(
            "TagsInput",
            "Noise",
            ["0.0001"],
            [str(i) for i in [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]],
            r_label="noise",
        ),
        NumericText("BoundedIntText", "Seed", 30, 1, 100, 1, r_label="seed"),
    ],
    r_label="synthetic",
)
