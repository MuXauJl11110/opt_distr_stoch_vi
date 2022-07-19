import os
import pickle
from typing import List, Optional

from decentralized.oracles.base import ArrayPair


def save_algorithms(
    topology: str,
    num_nodes: int,
    data: str,
    methods: List[object],
    method_names: List[str],
    z_true: ArrayPair,
    dist_to_opt_type: Optional[List[str]] = ["argument", "gradient", "consensus"],
    logs_path: Optional[str] = "./logs",
    experiment_type: Optional[str] = "real",
):
    attrs = {
        "argument": "argument_primal_distance_to_opt",
        "gradient": "gradient_primal_distance_to_opt",
        "consensus": "argument_primal_distance_to_consensus",
    }

    path = os.path.join(logs_path, experiment_type, topology, f"{num_nodes}_{data}")
    os.system(f"mkdir -p {path}")

    for type_ in dist_to_opt_type:
        os.system(f"mkdir -p {os.path.join(path, type_)}")

        for method, method_name in zip(methods, method_names):
            with open(os.path.join(path, type_, method_name), "wb") as f:
                pickle.dump(getattr(method.logger, attrs[type_]), f)

    with open(os.path.join(path, "z_true"), "wb") as f:
        pickle.dump(z_true, f)
