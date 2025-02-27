# stratified experiment
from laat.datasets import (
    HeartDiseaseDataset,  # works
    DiabetesDataset,  # works
    BankMarketingDataset,
    AdultDataset,  # works
    PhishingDataset,
    ThyroidCancerDataset,  # works
)
from laat.models import LAATClassifier
from laat.splitters import NShotSplitter
from sklearn.neural_network import MLPClassifier
from laat.utils import MetricLogger, partial_class, NShotLogger, NShotLog, ShotLog, Metric
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


import torch
from torch import nn
from sklearn.model_selection import StratifiedShuffleSplit, KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler
from tqdm.auto import tqdm
from skorch.callbacks import EarlyStopping
from hpsklearn import HyperoptEstimator, mlp_regressor, mlp_classifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import numpy as np


class TorchLogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.LazyLinear(1)

    def forward(self, x):
        return self.model(x)


class TorchMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.LazyLinear(100), nn.ReLU(), nn.LazyLinear(1))

    def forward(self, x):
        return self.model(x)


device = "cuda"
attribution_method = "ixg"
nrepetitions = 30
upto_dataset_percent = 0.1
n_shots = 5
nsteps = 200
seed = 69
dataset = HeartDiseaseDataset.load()

np.random.seed(seed)
torch.manual_seed(seed)

model_kwargs = {
    # "module": PModel,
    "llm_ratings": dataset.llm_ratings,
    "lr": 0.1,
    "max_epochs": nsteps,
    # "train_split": None,
    "optimizer": torch.optim.SGD,
    "optimizer__momentum": 0.9,
    "optimizer__weight_decay": 1e-4,
    "verbose": False,
    "device": device,
}

gammas = [1.0, 1e1, 1e2, 1e3]  # [0, 1e1, 1e2, 1e3]
models = {
    f"laat_mlp_{gamma}": partial_class(
        LAATClassifier, module=TorchMLP, gamma=gamma, is_mse=True, train_split=None, **model_kwargs
    )
    for gamma in gammas
}
# models.update(
#     {
#         f"laat_mlp_{gamma}_cosine": partial_class(
#             LAATClassifier, module=TorchMLP, gamma=gamma, is_mse=False, train_split=None, **model_kwargs
#         )
#         for gamma in gammas
#     }
# )

models.update(
    {
        f"laat_lr_{gamma}": partial_class(
            LAATClassifier, module=TorchLogisticRegression, gamma=gamma, is_mse=True, train_split=None, **model_kwargs
        )
        for gamma in gammas
    }
)

# models.update(
#     {
#         f"laat_lr_{gamma}_cosine": partial_class(
#             LAATClassifier, module=TorchLogisticRegression, gamma=gamma, is_mse=False, train_split=None, **model_kwargs
#         )
#         for gamma in gammas
#     }
# )

additional_models = {
    "catboost": partial_class(CatBoostClassifier, verbose=False),
    "xgb": XGBClassifier,
    "lr": LogisticRegression,
    "mlp": partial_class(MLPClassifier, max_iter=500),
    "knn": KNeighborsClassifier,
    "random_forest": RandomForestClassifier,
}
models.update(additional_models)

metrics = {"acc": accuracy_score, "f1": f1_score, "roc": roc_auc_score}

# train original model on full dataset first
# X_train, X_test, y_train, y_test = train_test_split(dataset.X, dataset.y, test_size=0.80)
# scaler = QuantileTransformer(n_quantiles=min(X_train.shape[0], 1000))
# X_train = torch.from_numpy(scaler.fit_transform(X_train)).float()
# X_test = torch.from_numpy(scaler.transform(X_test)).float()
# model = LAATClassifier(
#     gamma=0,
#     **model_kwargs,
#     callbacks=[("es", EarlyStopping(patience=16))],
# )
# model.fit(X_train, y_train)
# preds = model.predict(X_test)
# probas = model.predict_proba(X_test)[:, 1]

# orig_acc = accuracy_score(y_test, preds)
# orig_f1 = f1_score(y_test, preds)
# orig_roc = roc_auc_score(y_test, probas)

# better generalization with small amounts of data
meta_logger = NShotLogger(f"{dataset.__class__.__name__}_{attribution_method}")
for model_name, model_init in models.items():
    print(model_name)
    model_logger = NShotLog(model_name=model_name)
    for shot in range(1, n_shots + 1):
        model = model_init()
        if model_name == "knn":
            model = model_init(n_neighbors=shot)
        shot_logger = ShotLog(shot=shot)
        for repetition in tqdm(range(nrepetitions)):
            X_train, y_train, X_test, y_test = NShotSplitter.split(dataset.X, dataset.y, shot)

            # scaler = QuantileTransformer(n_quantiles=min(X_train.shape[0], 1000))
            scaler = MinMaxScaler()
            # scaler = StandardScaler()
            X_train = np.array(scaler.fit_transform(X_train), dtype=np.float32)
            X_test = np.array(scaler.transform(X_test), dtype=np.float32)

            y_train, y_test = np.array(y_train, dtype=np.float32), np.array(y_test, dtype=np.float32)

            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probas = model.predict_proba(X_test)[:, 1]

            for metric_name, metric_function in metrics.items():
                value = metric_function(y_test, preds)  # / (orig_acc + 1e-9)
                shot_logger.update(metric_update=Metric(name=metric_name, repetitions=[value]))
        model_logger.update(shot_logger)
    meta_logger.update(nshot_log_update=model_logger)
meta_logger.plot()
meta_logger.save()


# meta_logger = NShotLogger(f"{dataset.__class__.__name__}_{attribution_method}_biased")
# for model_name, model_init in models.items():
#     print(model_name)
#     model_logger = NShotLog(model_name=model_name)
#     for shot in [1]:
#         model = model_init()
#         shot_logger = ShotLog(shot=shot)
#         sss = StratifiedShuffleSplit(n_splits=nrepetitions, train_size=0.8)
#         for i, (train_index, test_index) in enumerate(tqdm(sss.split(dataset.X, dataset.y))):
#             X_train, y_train, X_test, y_test = (
#                 dataset.X[train_index],
#                 dataset.y[train_index],
#                 dataset.X[test_index],
#                 dataset.y[test_index],
#             )

#             # men - heart disease, women - no heart disease
#             biased_indices = ((y_train == 1) & (X_train[:, 1] == 0)) | ((y_train == 0) & (X_train[:, 1] == 1))
#             X_train = X_train[biased_indices]
#             y_train = y_train[biased_indices]
#             # scaler = QuantileTransformer(n_quantiles=min(X_train.shape[0], 1000))
#             scaler = MinMaxScaler()
#             # scaler = StandardScaler()
#             X_train = np.array(scaler.fit_transform(X_train), dtype=np.float32)
#             X_test = np.array(scaler.transform(X_test), dtype=np.float32)

#             y_train, y_test = np.array(y_train, dtype=np.float32), np.array(y_test, dtype=np.float32)

#             model.fit(X_train, y_train)
#             preds = model.predict(X_test)
#             probas = model.predict_proba(X_test)[:, 1]

#             for metric_name, metric_function in metrics.items():
#                 value = metric_function(y_test, preds)  # / (orig_acc + 1e-9)
#                 shot_logger.update(metric_update=Metric(name=metric_name, repetitions=[value]))
#         model_logger.update(shot_logger)
#     meta_logger.update(nshot_log_update=model_logger)
# # meta_logger.plot()
# meta_logger.save()


# regular ass training
# metric_loggers = {}
# gammas = [0, 1]
# for gamma in gammas:
#     model = LAATClassifier(
#         gamma=gamma,
#         **model_kwargs,
#         # callbacks=[("es", EarlyStopping(patience=50))],
#     )
#     model = HyperoptEstimator(classifier=mlp_classifier(model))

#     metric_logger = MetricLogger(f"stratified_{gamma}")
#     sss = KFold(n_splits=nrepetitions, shuffle=True, random_state=seed)
#     for i, (train_index, test_index) in enumerate(tqdm(sss.split(dataset.X, dataset.y))):
#         X_train, y_train, X_test, y_test = (
#             dataset.X[train_index],
#             dataset.y[train_index],
#             dataset.X[test_index],
#             dataset.y[test_index],
#         )

#         scaler = MinMaxScaler()
#         X_train = torch.from_numpy(scaler.fit_transform(X_train)).float()
#         X_test = torch.from_numpy(scaler.transform(X_test)).float()

#         model.fit(X_train, y_train)
#         preds = model.predict(X_test)
#         # probas = model.predict_proba(X_test)[:, 1]

#         acc = accuracy_score(y_test, preds)
#         f1 = f1_score(y_test, preds)
#         # roc = roc_auc_score(y_test, probas)

#         metric_logger({"acc": acc, "f1": f1})
#     metric_loggers.update(metric_logger.accumulate())
# meta_metric_logger = MetricLogger("meta")
# meta_metric_logger.log = metric_loggers
# meta_metric_logger.bar(nexperiments=len(gammas))
