import numpy as np
from abc import ABC, abstractmethod
from typing import Callable
from pydantic import BaseModel

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score


class LAATMetric(ABC, BaseModel):
    metric_name: str
    metric_function: Callable

    @abstractmethod
    def __call__(self, y_pred: np.array, y_proba: np.array, y_true: np.array) -> float:
        pass


class Accuracy(LAATMetric):
    metric_name: str = "accuracy"
    metric_function: Callable = accuracy_score

    def __call__(self, y_pred: np.array, y_proba: np.array, y_true: np.array) -> float:
        return self.metric_function(y_true=y_true, y_pred=y_pred)


class F1(LAATMetric):
    metric_name: str = "f1"
    metric_function: Callable = f1_score

    def __call__(self, y_pred: np.array, y_proba: np.array, y_true: np.array) -> float:
        return self.metric_function(y_true=y_true, y_pred=y_pred)


class MacroF1(LAATMetric):
    metric_name: str = "f1"
    metric_function: Callable = f1_score

    def __call__(self, y_pred: np.array, y_proba: np.array, y_true: np.array) -> float:
        print(y_pred.shape, y_proba.shape)
        return self.metric_function(y_true=y_true, y_pred=y_pred, average="macro")


class ROCAUC(LAATMetric):
    metric_name: str = "roc_auc"
    metric_function: Callable = roc_auc_score

    def __call__(self, y_pred: np.array, y_proba: np.array, y_true: np.array) -> float:
        return self.metric_function(y_true=y_true, y_score=y_proba)


class OVOROCAUC(LAATMetric):
    metric_name: str = "roc_auc"
    metric_function: Callable = roc_auc_score

    def __call__(self, y_pred: np.array, y_proba: np.array, y_true: np.array) -> float:
        print(y_pred.shape, y_proba.shape)
        return self.metric_function(y_true=y_true, y_score=y_proba, multi_class="ovo", labels=[0, 1, 2])
