import torch
import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.base import BaseEstimator
from typing import Optional, Callable, Type
from abc import ABC


class TrainRunInfo(BaseModel):
    """Used for passing training info to the model init"""

    shot: int
    repetition: int
    kwargs: Optional[dict] = None

    def __str__(self):
        return f"shot_{self.shot}_repetition_{self.repetition}"


class LAATModel(ABC):
    def __init__(
        self,
        model_name: str,
        model_class: Type[BaseEstimator],
        pandas_to_numpy_mapper: Callable,
        scaler_class: Type[BaseEstimator] = MinMaxScaler,
    ):
        self.model_name = model_name
        self.model_class = model_class
        self._pandas_to_numpy_mapper = pandas_to_numpy_mapper
        self._scaler_class = scaler_class

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        train_run_info: TrainRunInfo,
        X_validation: Optional[pd.DataFrame] = None,
        y_validation: Optional[pd.DataFrame] = None,
    ) -> None:
        self.model = self._init_model(train_run_info)
        X_train, y_train = self._pandas_to_numpy_mapper(X_train, y_train)
        X_train = self._preprocess(X_train, train=True)
        self.model.fit(X_train, y_train)

    def _init_model(self, train_run_info: TrainRunInfo) -> BaseEstimator:
        return self.model_class()

    def _preprocess(self, X: np.array, train: bool) -> np.array:
        if train:
            self.scaler = self._scaler_class()
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)

    def predict(self, X: pd.DataFrame) -> np.array:
        X = self._pandas_to_numpy_mapper(X)
        X = self._preprocess(X, train=False)
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.array:
        X = self._pandas_to_numpy_mapper(X)
        X = self._preprocess(X, train=False)
        return self.model.predict_proba(X)

    def clear(self) -> None:
        """Delete model and free up memory"""
        del self.model
        del self.scaler
        torch.cuda.empty_cache()
