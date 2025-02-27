import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from laat.models.base import LAATModel, TrainRunInfo
from typing import Optional


class TabPFNLAATModel(LAATModel):
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
        # if crosses upper limit of TabPFN, sample 10000
        if X_train.shape[0] > 10000:
            random_sample = np.random.permutation(X_train.shape[0])[:10000]
            X_train = X_train[random_sample]
            y_train = y_train[random_sample]
        X_train = self._preprocess(X_train, train=True)
        self.model.fit(X_train, y_train)

    def _preprocess(self, X: pd.DataFrame, train: bool) -> np.array:
        # tabpfn handles preprocessing
        X = torch.from_numpy(X).float()
        return X

    def predict(self, X: pd.DataFrame) -> np.array:
        X = self._pandas_to_numpy_mapper(X)
        X = self._preprocess(X, train=False)
        # tabfpn batching
        return np.concatenate(
            [self.model.predict(Xb[0]) for Xb in DataLoader(TensorDataset(X), batch_size=10000, shuffle=False)], axis=0
        )

    def predict_proba(self, X: pd.DataFrame) -> np.array:
        X = self._pandas_to_numpy_mapper(X)
        X = self._preprocess(X, train=False)
        return np.concatenate(
            [self.model.predict_proba(Xb[0]) for Xb in DataLoader(TensorDataset(X), batch_size=10000, shuffle=False)],
            axis=0,
        )

    def clear(self) -> None:
        """Delete model and free up memory"""
        del self.model
        torch.cuda.empty_cache()
