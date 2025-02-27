from laat.models.base import LAATModel, TrainRunInfo
from laat.datasets import LAATDataset
from sklearn.base import BaseEstimator
from typing import Type, Callable
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from typing import Optional
import torch


class KNNLAATModel(LAATModel):
    def _init_model(self, train_run_info: TrainRunInfo) -> BaseEstimator:
        return self.model_class(n_neighbors=min(5, train_run_info.shot))


class GridSearchCVLAATModel(LAATModel):
    def __init__(
        self,
        model_name: str,
        model_class: Type[BaseEstimator],
        pandas_to_numpy_mapper: Callable,
        scaler_class: Type[BaseEstimator] = MinMaxScaler,
        param_grid: dict = {},
        scoring: str = "roc_auc",
    ):
        super().__init__(
            model_name=model_name,
            model_class=model_class,
            pandas_to_numpy_mapper=pandas_to_numpy_mapper,
            scaler_class=scaler_class,
        )

        self._param_grid = param_grid
        self._scoring = scoring

    def _init_model(self, train_run_info: TrainRunInfo) -> BaseEstimator:
        # if one-shot, no grid search
        if train_run_info.shot == 1:
            return self.model_class()
        # do grid search otherwise
        return GridSearchCV(
            cv=min(5, train_run_info.shot),
            estimator=self.model_class(),
            param_grid=self._param_grid,
            scoring=self._scoring,
        )


class CatBoostLAATModel(LAATModel):
    def __init__(
        self,
        model_name: str,
        model_class: Type[BaseEstimator],
        pandas_to_numpy_mapper: Callable,
        dataset: LAATDataset,
        param_grid: dict,
        scaler_class: Type[BaseEstimator] = MinMaxScaler,
    ):
        super().__init__(
            model_name=model_name,
            model_class=model_class,
            pandas_to_numpy_mapper=pandas_to_numpy_mapper,
            scaler_class=scaler_class,
        )
        self.dataset = dataset
        self.param_grid = param_grid

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        train_run_info: TrainRunInfo,
        X_validation: Optional[pd.DataFrame] = None,
        y_validation: Optional[pd.DataFrame] = None,
    ) -> None:
        self.model = self._init_model(train_run_info)
        X_train = X_train.to_numpy()
        y_train = self._pandas_to_numpy_mapper(y=y_train)
        X_train = self._preprocess(X_train, train=True)

        if train_run_info.shot == 1:
            self.model.fit(X_train, y_train, verbose=False)
        else:
            self.model.grid_search(self.param_grid, X_train, y_train, cv=min(3, train_run_info.shot), verbose=False)

    def _init_model(self, train_run_info: TrainRunInfo) -> BaseEstimator:
        cat_columns = [self.dataset.df.columns.tolist().index(k) for k in self.dataset.categorical_columns]
        return self.model_class(cat_features=cat_columns, logging_level="Silent")

    def _preprocess(self, X: np.array, train: bool) -> np.array:
        return X

    def predict(self, X: pd.DataFrame) -> np.array:
        X = self._preprocess(X, train=False)
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.array:
        X = self._preprocess(X, train=False)
        return self.model.predict_proba(X)

    def clear(self) -> None:
        """Delete model and free up memory"""
        del self.model
        torch.cuda.empty_cache()
