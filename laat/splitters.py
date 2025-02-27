import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class NShotSplitter:
    @staticmethod
    def split(
        X: pd.DataFrame, y: pd.DataFrame, shot: int
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        all_train_label_indices, all_test_label_indices = [], []
        for label in y.iloc[:, 0].unique().tolist():
            label_indices = np.argwhere(y.iloc[:, 0] == label).reshape(-1)
            label_indices = np.random.permutation(label_indices)
            train_label_indices, test_label_indices = label_indices[:shot], label_indices[shot:]
            all_train_label_indices.append(train_label_indices)
            all_test_label_indices.append(test_label_indices)
        all_train_label_indices = np.concatenate(all_train_label_indices)
        all_test_label_indices = np.concatenate(all_test_label_indices)

        return (
            X.iloc[all_train_label_indices],
            X.iloc[all_test_label_indices],
            y.iloc[all_train_label_indices],
            y.iloc[all_test_label_indices],
        )


class SkewedSplitter:
    @staticmethod
    def split(
        dataset_name: str,
        X: pd.DataFrame,
        y: pd.DataFrame,
        delta: int,
        train_size: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)

        # found in sklearn datasets ipynb
        if dataset_name == "diabetes":
            # not amazing, but it does the job
            condition = (X_train["Age"] < 50) & (y_train["Outcome"] == "yes")
            # condition = (X_train["Pregnancies"] > 1) & (y_train["Outcome"] == "yes")
        elif dataset_name == "breast-ljubljana":
            # good
            condition = (X_train["Age"] < 5) & (y_train["Class"] == "yes")
        elif dataset_name == "myocardial":
            condition = (X_train["SEX"] == "female") & (y_train["ZSN"] == "yes")
        elif dataset_name == "adult":
            # not working
            condition = ((X_train["workclass"] != "Private") & (y_train["income"] == "yes")) | (
                X_train["workclass"] == "Private"
            ) & (y_train["income"] == "no")
        elif dataset_name == "cdc-diabetes":
            condition = (X_train["Sex"] == 0) & (y_train["Diabetes_binary"] == "no") | (X_train["Sex"] == 1) & (
                y_train["Diabetes_binary"] == "yes"
            )
        else:
            raise NotImplementedError(f"skewness condition for {dataset_name} has not been implemented")

        condition_indices = condition.to_numpy().nonzero()[0]
        num_condition = len(condition_indices)
        delta_condition_indices = np.random.permutation(condition_indices)[: int(num_condition * (1 - delta))]
        biased_selector = np.ones_like(condition).astype(bool)
        biased_selector[delta_condition_indices] = False

        return X_train[biased_selector], X_test, y_train[biased_selector], y_test


# import torch
# from abc import ABC, abstractmethod


# class Splitter(ABC):
#     @abstractmethod
#     def split(X: torch.tensor, y: torch.tensor) -> tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
#         pass


# class StratifiedSplitter(Splitter):
#     @staticmethod
#     def split(X: torch.tensor, y: torch.tensor) -> tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
#         pass


# class NShotSplitter(Splitter):
#     @staticmethod
#     def split(
#         X: torch.tensor, y: torch.tensor, shot: int
#     ) -> tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
#         n_classes = len(y.unique())
#         all_chosen_indices = []
#         for class_index in range(n_classes):
#             class_k = torch.where(y == class_index)[0]
#             if len(class_k) <= shot:
#                 raise ValueError(f"There must be at least {shot} + 1 examples in the dataset")
#             permutation = torch.randperm(len(class_k))
#             chosen_indices = class_k[permutation[:shot]]
#             all_chosen_indices.append(chosen_indices)
#         all_chosen_indices = torch.cat(all_chosen_indices)
#         test_indices = torch.ones(y.shape[0]).bool()
#         test_indices[all_chosen_indices] = False

#         X_train, y_train, X_test, y_test = (
#             X[all_chosen_indices],
#             y[all_chosen_indices],
#             X[test_indices],
#             y[test_indices],
#         )

#         return X_train, y_train, X_test, y_test
