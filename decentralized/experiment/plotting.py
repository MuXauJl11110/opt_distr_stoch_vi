import itertools
import os
import pickle
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from decentralized.oracles.base import ArrayPair
from numpy import linalg as LA


def preplot_algorithms(
    topology: str,
    num_nodes: int,
    data: str,
    labels: List[str],
    methods: List[object],
    dist_to_opt_type: Optional[str] = "argument",
    save_to=None,
):
    plt.close()
    plt.figure(figsize=(12, 6))
    plt.suptitle(f"{topology} graph, {num_nodes} nodes, {data}", fontsize=25)
    marker = itertools.cycle(("o", "v", "^", "<", ">", "s", "8", "p"))

    plt.subplot(121)
    ax = plt.gca()
    for method, label in zip(methods, labels):
        if dist_to_opt_type == "argument":
            dist_to_opt = method.logger.argument_primal_distance_to_opt
        elif dist_to_opt_type == "gradient":
            dist_to_opt = method.logger.gradient_primal_distance_to_opt
        else:
            raise ValueError(f"Unknown distance to optimum type: {dist_to_opt_type}!")
        comm_steps = np.arange(
            0,
            method.logger.comm_per_iter * len(dist_to_opt),
            method.logger.comm_per_iter,
        )
        color = next(ax._get_lines.prop_cycler)["color"]
        plt.plot(
            comm_steps,
            dist_to_opt,
            label=label,
            marker=next(marker),
            color=color,
            markevery=0.15,
            markersize=6,
        )

    ax.set_xlabel("communications", fontsize=22)
    if dist_to_opt_type == "argument":
        ax.set_ylabel("$||\overline{z} - z^*||^2$", fontsize=22)
    elif dist_to_opt_type == "gradient":
        ax.set_ylabel("$||\dfrac{1}{M} \sum_{m=1}^{M} F_m(z_m)||^2$", fontsize=22)
    else:
        raise ValueError(f"Unknown distance to optimum type: {dist_to_opt_type}!")
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.tick_params(labelsize=15)
    ax.xaxis.get_offset_text().set_size(15)
    ax.set_yscale("log")
    ax.legend(fontsize=16, loc="lower left")
    ax.grid()

    plt.subplot(122)
    ax = plt.gca()
    for method, label in zip(methods, labels):
        dist_to_con = method.logger.argument_primal_distance_to_consensus
        comm_steps = np.arange(
            0,
            method.logger.comm_per_iter * len(dist_to_con),
            method.logger.comm_per_iter,
        )
        color = next(ax._get_lines.prop_cycler)["color"]
        plt.plot(
            comm_steps,
            dist_to_con,
            label=label,
            marker=next(marker),
            color=color,
            markevery=0.15,
            markersize=6,
        )

    ax.set_xlabel("communications", fontsize=22)
    ax.set_ylabel("dist. to consensus", fontsize=22)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.tick_params(labelsize=15)
    ax.xaxis.get_offset_text().set_size(15)
    ax.set_yscale("log")
    ax.legend(fontsize=16, loc="lower right")
    ax.grid()

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if save_to is not None:
        plt.savefig(save_to)
        plt.close()
    else:
        plt.show()


def plot_algorithms(
    topology: str,
    num_nodes: int,
    data: str,
    labels: List[str],
    method_names: List[str],
    comm_budget_experiment: int,
    dist_types: Optional[List[str]] = ["argument", "gradient", "consensus"],
    logs_path: Optional[str] = "./logs",
    plots_path: Optional[str] = "./plots",
    experiment_type: Optional[str] = "real",
    save_folder: Optional[str] = None,
    save_to: Optional[str] = None,
):
    def saddle_grad_norm(a: ArrayPair, b: ArrayPair, **kwargs):
        x, y = (a - b).tuple()
        return LA.norm(x.sum(axis=0) / x.shape[0]) ** 2 + LA.norm(y.sum(axis=0) / y.shape[0]) ** 2

    for type_ in dist_types:
        plt.figure(figsize=(12, 6))
        plt.title(f"{topology} graph, {num_nodes} nodes, {data}", fontsize=25)
        marker = itertools.cycle(("o", "v", "^", "<", ">", "s", "8", "p"))

        ax = plt.gca()
        min_comm = np.inf
        flag = True
        if type_ == "argument":
            plt.ylabel("$||\overline{z} - z^*||^2$", fontsize=22)
        elif type_ == "gradient":
            plt.ylabel("$||\dfrac{1}{M} \sum_{m=1}^{M} F_m(z_m)||^2$", fontsize=22)
        elif type_ == "consensus":
            plt.ylabel("dist. to consensus", fontsize=22)
            flag = False
        else:
            raise ValueError(f"Unknown distance to optimum type: {type_}!")
        for method_name, label in zip(method_names, labels):
            path = os.path.join(logs_path, experiment_type, topology, f"{num_nodes}_{data}")
            os.system(f"mkdir -p {os.path.join(path, type_)}")
            with open(os.path.join(path, type_, f"{method_name}.pkl"), "rb") as f:
                dist = pickle.load(f)
            with open(os.path.join(path, "z_true"), "rb") as f:
                z_true = pickle.load(f)
                z_0 = ArrayPair.zeros(z_true.x.shape[0])
            comm_steps = np.arange(len(dist)) * (comm_budget_experiment / len(dist))
            color = next(ax._get_lines.prop_cycler)["color"]
            if flag:
                i = np.where(dist == np.amin(dist))[0][0]
                if dist[i] < min_comm:
                    min_comm = comm_steps[i]

                if type_ == "argument":
                    dist /= (z_0 - z_true).dot(z_0 - z_true)
                elif type_ == "gradient":
                    dist /= saddle_grad_norm(z_0, z_true)
            plt.plot(
                comm_steps,
                dist,
                label=label,
                marker=next(marker),
                color=color,
                markevery=0.15,
                markersize=6,
            )

        plt.xlabel("communications", fontsize=22)
        if flag:
            plt.xlim((0, min_comm))
        plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        plt.tick_params(labelsize=15)
        plt.yscale("log")
        plt.legend(fontsize=16, loc="lower left")
        plt.grid()
        plt.tight_layout(rect=[0, 0, 1, 0.93])

        if save_to is not None:
            if save_folder is not None:
                path = os.path.join(plots_path, experiment_type, type_, topology, save_folder)
            else:
                path = os.path.join(plots_path, experiment_type, type_, topology)
            os.system(f"mkdir -p {path}")
            plt.savefig(os.path.join(path, f"{save_to}.pdf"))
            plt.close()
        else:
            plt.show()
