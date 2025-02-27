import torch
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field
import pandas as pd
import seaborn as sns
import pickle
from datetime import datetime
from typing import Type, TypeVar


def partial_class(class_init, **kwargs):
    def initialize_class(**add_kwargs):
        kwargs.update(add_kwargs)
        return class_init(**kwargs)

    return initialize_class


class Metric(BaseModel):
    name: str
    repetitions: list[float] = []

    def update(self, value: float) -> None:
        self.repetitions.append(value)


class ShotLog(BaseModel):
    shot: int
    metrics: list[Metric] = []

    def update(self, metric_update: Metric) -> None:
        updated_existing = False
        for metric in self.metrics:
            if metric.name == metric_update.name:
                metric.update(metric_update.repetitions[0])  # single element update
                updated_existing = True
                break
        if not updated_existing:
            self.metrics.append(metric_update)


class NShotLog(BaseModel):
    model_name: str
    shot_logs: list[ShotLog] = []

    def update(self, shot_log_update: ShotLog) -> None:
        self.shot_logs.append(shot_log_update)


class NShotLogger:
    def __init__(self, experiment_name: str) -> None:
        self.experiment_name = experiment_name
        self.log: list[NShotLog] = []

    def update(self, nshot_log_update: NShotLog) -> None:
        self.log.append(nshot_log_update)

    def to_dataframe(self, include_experiment_name: bool = False) -> pd.DataFrame:
        rows = []
        for nshot_log in self.log:
            for shot_log in nshot_log.shot_logs:
                repetition_count = len(shot_log.metrics[0].repetitions)
                for repetition in range(repetition_count):
                    row_dict = {"model": nshot_log.model_name, "shot": shot_log.shot}
                    if include_experiment_name:
                        row_dict["experiment_name"] = self.experiment_name
                    for metric in shot_log.metrics:
                        row_dict[metric.name] = metric.repetitions[repetition]
                    rows.append(row_dict)
        return pd.DataFrame(rows)

    def plot(self) -> None:
        df = self.to_dataframe()
        available_metrics = [metric.name for metric in self.log[0].shot_logs[0].metrics]
        fig, axs = plt.subplots(1, len(available_metrics), figsize=(len(available_metrics) * 5, 5))
        for i, available_metric in enumerate(available_metrics):
            sns.lineplot(df, ax=axs[i], x="shot", y=available_metric, hue="model")
        plt.show()

    def save(self) -> None:
        timestamp = datetime.now().strftime("%y%m%d%H%M")
        with open(f"train_logs/{self.experiment_name}{timestamp}.pkl", "wb") as f:
            pickle.dump(self, f)


class ExperimentLogger:
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.loggers: list[NShotLogger] = []

    def update(self, logger: NShotLogger) -> None:
        self.loggers.append(logger)

    def save(self) -> None:
        timestamp = datetime.now().strftime("%y%m%d%H%M")
        with open(f"train_logs/{self.experiment_name}{timestamp}.pkl", "wb") as f:
            pickle.dump(self, f)

    def to_dataframe(self, aggregation_metric: str) -> pd.DataFrame:
        all_experiments = pd.concat([log.to_dataframe(include_experiment_name=True) for log in self.loggers], axis=0)
        df_agg = (
            all_experiments.groupby(["experiment_name", "shot", "model"]).agg(["mean", "std"])[aggregation_metric] * 100
        ).round(1)
        df_agg["mean ± std"] = "$" + df_agg["mean"].astype(str) + "_{" + df_agg["std"].astype(str) + "}$"
        df_agg = df_agg.drop(columns=["mean", "std"])
        df_agg = df_agg.reset_index()
        df_agg = df_agg.pivot(index=["experiment_name", "shot"], columns="model", values="mean ± std")
        return df_agg

    def to_raw_dataframe(self, aggregation_metric: str) -> pd.DataFrame:
        all_experiments = pd.concat([log.to_dataframe(include_experiment_name=True) for log in self.loggers], axis=0)
        return all_experiments

    def to_aggregable_dataframe(self, aggregation_metric: str) -> pd.DataFrame:
        all_experiments = pd.concat([log.to_dataframe(include_experiment_name=True) for log in self.loggers], axis=0)
        df_agg = (
            all_experiments.groupby(["experiment_name", "shot", "model"]).agg(["mean", "std"])[aggregation_metric] * 100
        ).round(2)
        df_agg = df_agg.reset_index()
        df_agg = df_agg.pivot(index=["experiment_name", "shot"], columns="model", values=["mean", "std"])
        return df_agg


class MetricLogger:
    def __init__(self, exp_name: str):
        self.exp_name = exp_name
        self.log = {}

    def __call__(self, new_log: dict):
        for key, item in new_log.items():
            key = f"{self.exp_name}_{key}"
            if key not in self.log:
                self.log[key] = []
            self.log[key].append(item)

    def __add__(self, other):
        if not isinstance(other, MetricLogger):
            raise ValueError("add only metricloggers")

        return {**self.log, **other.log}

    def accumulate(self) -> dict:
        return {key: torch.tensor(value).mean() for key, value in self.log.items()}

    def plot(self, disperse: bool = False):
        if not disperse:
            for key, items in self.log.items():
                plt.plot(items, label=key)
            plt.legend()
            plt.show()
        else:
            key_set = set()
            for key in self.log.keys():
                key_key = key.split("_")[-1]
                key_set.add(key_key)
            key_set_list = list(key_set)
            fig, axs = plt.subplots(1, len(key_set_list), figsize=(len(key_set_list) * 5, 5))
            for i, key_key in enumerate(key_set_list):
                for key, items in self.log.items():
                    if key.endswith(key_key):
                        axs[i].plot(items, label=key)
                axs[i].legend()
                # axs[i].set_ylim(0, 1)
                axs[i].set_title(f"{key_key}")
            plt.tight_layout()
            plt.show()

    def bar(self, nexperiments=2):
        color_swatch = ["blue", "red", "orange", "violet", "pink"]
        nmetrics = len(list(self.log)) // nexperiments
        colors = []
        for i in range(nexperiments):
            colors += [color_swatch[i]] * nmetrics
        plt.bar(list(self.log.keys()), list(self.log.values()), color=colors)
        plt.tight_layout()
        plt.show()
