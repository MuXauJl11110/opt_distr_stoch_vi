import os
import sys
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from definitions import ROOT_DIR
from prettytable import PrettyTable
from src.network.network import Network
from src.runner.base import BaseRunner
from src.config_managers.scheduler import SchedulerConfigManager


class Scheduler(object):
    """
    Class for scheduling method`s parameters.
    """

    def __init__(
        self,
        method_runner: BaseRunner,
        network: Network,
        config_manager: Optional[SchedulerConfigManager] = SchedulerConfigManager(),
    ):
        """
        :param method_runner: Method runner.
        :param network: Network.
        :param config_manager: Configuration manager.
        """
        self.method_runner = method_runner
        self.network = network
        self.config_manager = config_manager
        self.num_states = network.num_states

        self.default_epoch_cfg = self.config_manager.config["default_epoch"]

        self.current_state = 0
        self.current_epoch = 0
        self.next_epoch_cnt = 0

        self.current_epoch_start = 0
        self.current_epoch_end = self.default_epoch_cfg["duration"]
        self.current_epoch_cfg = self.config_manager.config["default_epoch"]
        self.eval_next_epoch()

        self.time = 0.0

        self.output_table = PrettyTable(["epoch", "steps performed", "elapsed time"])

        self.peeked = deque()

    def get_epoch_config(self):
        if self.current_state == self.next_epoch_start:
            self.current_epoch_start = min(self.next_epoch_start, self.num_states)
            self.current_epoch_end = min(self.next_epoch_end, self.num_states)
            self.current_epoch_cfg = self.next_epoch_cfg
            self.eval_next_epoch()
        elif (
            self.current_state + self.default_epoch_cfg["duration"]
            >= self.next_epoch_start
        ):
            self.current_epoch_start = min(self.current_state, self.num_states)
            self.current_epoch_end = min(self.next_epoch_start, self.num_states)
            self.current_epoch_cfg = self.default_epoch_cfg
        else:
            self.current_epoch_start = min(self.current_state, self.num_states)
            self.current_epoch_end = min(
                self.current_epoch_end + self.default_epoch_cfg["duration"],
                self.num_states,
            )
            self.current_epoch_cfg = self.default_epoch_cfg

    def eval_next_epoch(self):
        if len(self.config_manager.config["epochs"]) == self.next_epoch_cnt:
            self.next_epoch_cfg = self.default_epoch_cfg
            self.next_epoch_start = sys.maxsize
            self.next_epoch_end = sys.maxsize
        else:
            self.next_epoch_cfg = self.config_manager.config["epochs"][
                self.next_epoch_cnt
            ]
            self.next_epoch_start, self.next_epoch_end = self.next_epoch_cfg["interval"]
            self.next_epoch_cnt += 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_state >= self.num_states:
            # next(self.network)
            raise StopIteration

        self.get_epoch_config()
        self.current_epoch += 1
        start = datetime.now()
        steps_performed = 0
        # print(f"Current epoch: {self.current_epoch_start, self.current_epoch_end}")
        # print(f"Next epoch: {self.next_epoch_start, self.next_epoch_end}")
        for state in range(self.current_epoch_start, self.current_epoch_end):
            if self.current_state >= self.num_states:
                end = datetime.now()
                elapsed_time = (end - start).total_seconds()
                # if self.current_epoch_cfg["verbose"]:
                #     self.output_table.add_row([self.current_epoch, steps_performed, elapsed_time])
                #     print(self.output_table)
                #     self.output_table.clear_rows()
                self.method_runner.logger.step()
                self.method_runner.logger.end()
                next(self.network)
                raise StopIteration
            else:
                W, lambda_min, lambda_max = next(self.network)
                self.method_runner.compute_method_params()
                # self.compute_method_params(lambda_min, lambda_max)
                # self.compute_method_params(min(self.network.min_lambdas), max(self.network.max_lambdas))
                self.method_runner.update_matrix(W)
                if (
                    "duration" in self.current_epoch_cfg
                    and (state - self.current_epoch_start)
                    in self.current_epoch_cfg["states"]
                ):
                    state_cfg = self.current_epoch_cfg["states"][
                        state - self.current_epoch_start
                    ]
                elif (
                    "interval" in self.current_epoch_cfg
                    and state in self.current_epoch_cfg["states"]
                ):
                    state_cfg = self.current_epoch_cfg["states"][state]
                else:
                    state_cfg = self.current_epoch_cfg["default_state"]
                for _ in range(state_cfg["steps_per_state"]):
                    self.method_runner.method.step()
                    self.method_runner.logger.step()
                    if state_cfg["verbose"]:
                        end = datetime.now()
                        elapsed_time = (end - start).total_seconds()
                        self.output_table.add_row(
                            [self.current_epoch, steps_performed, elapsed_time]
                        )
                        print(self.output_table)
                        self.output_table.clear_rows()
                self.current_state += 1
                steps_performed += state_cfg["steps_per_state"]

        end = datetime.now()
        elapsed_time = (end - start).total_seconds()
        # if self.current_epoch_cfg["verbose"]:
        #     self.output_table.add_row([self.current_epoch, steps_performed, elapsed_time])
        #     print(self.output_table)
        #     self.output_table.clear_rows()

        return elapsed_time

    def peek(self, ahead: Optional[int] = 0) -> Tuple[np.ndarray, float, float]:
        while len(self.peeked) <= ahead:
            self.peeked.append(self.__next__())
        return self.peeked[ahead]
