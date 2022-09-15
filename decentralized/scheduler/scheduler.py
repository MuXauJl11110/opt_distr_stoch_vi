import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

import schema
import yaml
from decentralized.loggers import LoggerDecentralized
from decentralized.network import Network
from decentralized.runners.base import BaseRunner
from prettytable import PrettyTable
from schema import And, Or, Schema, Use


class SchedulerConfigManager(object):
    """
    Instrument for managing configurations for the Scheduler class.
    """

    def __init__(
        self,
        config_path: Optional[str] = "./configs/config_scheduler.yaml",
        general_config_path: Optional[str] = "./configs/general_config_scheduler.yaml",
    ):
        """
        :param config_path: Relative path to the scheduler configuration file.
        :param general_config_path: Relative path to the general configuration file.
        """
        if __name__ == "__main__":
            self.path = os.path.split(sys.argv[0])[0]
        else:
            self.path = os.path.split(__file__)[0]
        self.general_config_path = os.path.join(self.path, general_config_path)
        self.config_path = os.path.join(self.path, config_path)

        with open(self.general_config_path, "r") as cfg:
            self.general_cfg = yaml.unsafe_load(cfg)

        for _type in self.general_cfg["schema"]:
            d = dict()
            for k, v in _type[1].items():
                d[k] = eval(v)
            setattr(self, "schema_" + _type[0], d)

        self.config_schema = Schema(
            {
                "default_epoch": self.schema_default_epoch,
                schema.Optional("epochs"): [self.schema_epoch],
            }
        )
        self.config_schema.validate(self.config)

    @property
    def config(self):
        with open(self.config_path, "r") as cfg:
            config = yaml.unsafe_load(cfg)
        self.config_schema.validate(config)
        self._config = config
        return self._config

    @config.setter
    def config(self, new_config: Dict):
        self._config = new_config
        with open(self.config_path, "w") as cfg:
            yaml.dump(new_config, cfg)


class Scheduler(object):
    """
    Class for scheduling methods's parameters.
    """

    def __init__(
        self,
        method_runner: BaseRunner,
        network: Network,
        logger: LoggerDecentralized,
        config_manager: Optional[SchedulerConfigManager] = SchedulerConfigManager(),
    ):
        """
        Parameters
        ----------
        method_runner : BaseSaddleMethod
            Method runner.
        network : Network
            Network class instance.
        logger : LoggerDecentralized
             Logger class instance.
        config_manager : Optional[SchedulerConfigManager], optional
            Configuration manager., by default SchedulerConfigManager()
        """
        self.network = network
        self.logger = logger
        self.method_runner = method_runner
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

    def get_epoch_config(self):
        if self.current_state == self.next_epoch_start:
            self.current_epoch_start = min(self.next_epoch_start, self.num_states)
            self.current_epoch_end = min(self.next_epoch_end, self.num_states)
            self.current_epoch_cfg = self.next_epoch_cfg
            self.eval_next_epoch()
        elif self.current_state + self.default_epoch_cfg["duration"] >= self.next_epoch_start:
            self.current_epoch_start = min(self.current_state, self.num_states)
            self.current_epoch_end = min(self.next_epoch_start, self.num_states)
            self.current_epoch_cfg = self.default_epoch_cfg
        else:
            self.current_epoch_start = min(self.current_state, self.num_states)
            self.current_epoch_end = min(self.current_epoch_end + self.default_epoch_cfg["duration"], self.num_states)
            self.current_epoch_cfg = self.default_epoch_cfg

    def eval_next_epoch(self):
        if len(self.config_manager.config["epochs"]) == self.next_epoch_cnt:
            self.next_epoch_cfg = self.default_epoch_cfg
            self.next_epoch_start = sys.maxsize
            self.next_epoch_end = sys.maxsize
        else:
            self.next_epoch_cfg = self.config_manager.config["epochs"][self.next_epoch_cnt]
            self.next_epoch_start, self.next_epoch_end = self.next_epoch_cfg["interval"]
            self.next_epoch_cnt += 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_state >= self.num_states:
            raise StopIteration

        self.get_epoch_config()
        self.current_epoch += 1
        start = datetime.now()
        steps_performed = 0
        for state in range(self.current_epoch_start, self.current_epoch_end):
            if self.current_state >= self.num_states:
                end = datetime.now()
                elapsed_time = (end - start).total_seconds()
                if self.current_epoch_cfg["verbose"]:
                    self.output_table.add_row([self.current_epoch, steps_performed, elapsed_time])
                    print(self.output_table)
                    self.output_table.clear_rows()
                self.logger.step(self.method_runner.method)
                self.logger.end(self.method_runner.method)
                raise StopIteration
            else:
                W, lambda_min, lambda_max = next(self.network)
                if not hasattr(self.method_runner.method, "gos_mat"):
                    self.method_runner.method.gos_mat = W
                elif not hasattr(self.method_runner.method, "mix_mat"):
                    self.method_runner.method.mix_mat = W
                self.method_runner.compute_method_params()
                if (
                    "duration" in self.current_epoch_cfg
                    and (state - self.current_epoch_start) in self.current_epoch_cfg["states"]
                ):
                    state_cfg = self.current_epoch_cfg["states"][state - self.current_epoch_start]
                elif "interval" in self.current_epoch_cfg and state in self.current_epoch_cfg["states"]:
                    state_cfg = self.current_epoch_cfg["states"][state]
                else:
                    state_cfg = self.current_epoch_cfg["default_state"]
                for _ in range(state_cfg["steps_per_state"]):
                    self.method_runner.method.step()
                    self.logger.step(self.method_runner.method)
                self.current_state += 1
                steps_performed += state_cfg["steps_per_state"]

        end = datetime.now()
        elapsed_time = (end - start).total_seconds()
        if self.current_epoch_cfg["verbose"]:
            self.output_table.add_row([self.current_epoch, steps_performed, elapsed_time])
            print(self.output_table)
            self.output_table.clear_rows()

        return elapsed_time
