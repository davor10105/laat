# stratified experiment
from laat.datasets import (
    HeartDiseaseDataset,  # works
    DiabetesDataset,  # works
    BankMarketingDataset,
    AdultDataset,  # works
    PhishingDataset,
    ThyroidCancerDataset,  # works
    BloodDataset,
)
from laat.models import LAATClassifier
from sklearn.neural_network import MLPClassifier
from laat.utils import MetricLogger


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


class PModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.LazyLinear(1)
        # self.model = nn.Sequential(nn.LazyLinear(100), nn.ReLU(), nn.LazyLinear(1))

    def forward(self, x):
        return self.model(x)


device = "cuda"
nrepetitions = 10
upto_dataset_percent = 0.1
n_dataset_steps = 5
nsteps = 500
seed = 69
dataset = BloodDataset.load()

np.random.seed(seed)
torch.manual_seed(seed)

dataset_disbalance_factor = int((1 / dataset.y.mean()).round())
max_train_size_multiplier = int(dataset.X.shape[0] * upto_dataset_percent) // dataset_disbalance_factor
train_size_step = max_train_size_multiplier // n_dataset_steps

model_kwargs = {
    "module": PModel,
    "llm_ratings": dataset.llm_ratings,
    "lr": 0.1,
    "max_epochs": nsteps,
    # "train_split": None,
    "optimizer": torch.optim.SGD,
    "optimizer__momentum": 0.9,
    "optimizer__weight_decay": 1e-4,
    "verbose": False,
    "device": device,
    "is_mse": True,
}

# train original model on full dataset first
X_train, X_test, y_train, y_test = train_test_split(dataset.X, dataset.y, test_size=0.80)
scaler = QuantileTransformer(n_quantiles=min(X_train.shape[0], 1000))
X_train = torch.from_numpy(scaler.fit_transform(X_train)).float()
X_test = torch.from_numpy(scaler.transform(X_test)).float()
model = LAATClassifier(
    gamma=0,
    **model_kwargs,
    callbacks=[("es", EarlyStopping(patience=16))],
)
model.fit(X_train, y_train)
preds = model.predict(X_test)
probas = model.predict_proba(X_test)[:, 1]

orig_acc = accuracy_score(y_test, preds)
orig_f1 = f1_score(y_test, preds)
orig_roc = roc_auc_score(y_test, probas)

# better generalization with small amounts of data
metric_loggers = {}
gammas = [1e2]
for train_size_multiplier in range(1, min(max_train_size_multiplier, n_dataset_steps)):
    # for train_size_multiplier in range(1, max_train_size_multiplier + 1, train_size_step):
    train_size = train_size_multiplier * dataset_disbalance_factor
    print(train_size)
    for gamma in gammas:
        model = LAATClassifier(
            gamma=gamma,
            train_split=None,  # no validation
            **model_kwargs,
            # callbacks=[("es", EarlyStopping(patience=50))],
        )
        # model = HyperoptEstimator(classifier=mlp_classifier(model))
        metric_logger = MetricLogger(f"stratified_{gamma}")
        sss = StratifiedShuffleSplit(n_splits=nrepetitions, train_size=train_size)
        for i, (train_index, test_index) in enumerate(tqdm(sss.split(dataset.X, dataset.y))):
            X_train, y_train, X_test, y_test = (
                dataset.X[train_index],
                dataset.y[train_index],
                dataset.X[test_index],
                dataset.y[test_index],
            )

            # scaler = QuantileTransformer(n_quantiles=min(X_train.shape[0], 1000))
            # scaler = MinMaxScaler()
            scaler = StandardScaler()
            X_train = torch.from_numpy(scaler.fit_transform(X_train)).float()
            X_test = torch.from_numpy(scaler.transform(X_test)).float()

            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probas = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, preds)  # / (orig_acc + 1e-9)
            f1 = f1_score(y_test, preds)  # / (orig_f1 + 1e-9)
            roc = roc_auc_score(y_test, probas)  # / (orig_roc + 1e-9)

            metric_logger({"acc": acc, "f1": f1, "roc": roc})
        for key, item in metric_logger.accumulate().items():
            if key not in metric_loggers:
                metric_loggers[key] = []
            metric_loggers[key].append(item)
meta_metric_logger = MetricLogger("meta")
meta_metric_logger.log = metric_loggers
meta_metric_logger.plot(disperse=True)

# better generalization with no data in certain categories
# metric_loggers = {}
# gammas = [0, 1e-1, 1, 1e1, 1e2]
# for gamma in gammas:
#     model = LAATClassifier(
#         gamma=gamma,
#         **model_kwargs,
#         # callbacks=[("es", EarlyStopping(patience=50))],
#     )
#     # model = HyperoptEstimator(classifier=mlp_classifier(model))
#     metric_logger = MetricLogger(f"stratified_{gamma}")
#     sss = StratifiedShuffleSplit(n_splits=nrepetitions, train_size=0.75, random_state=seed)
#     for i, (train_index, test_index) in enumerate(tqdm(sss.split(dataset.X, dataset.y))):
#         X_train, y_train, X_test, y_test = (
#             dataset.X[train_index],
#             dataset.y[train_index],
#             dataset.X[test_index],
#             dataset.y[test_index],
#         )

#         # men - heart disease, women - no heart disease
#         biased_indices = ((y_train == 1) & (X_train[:, 1] == 0)) | ((y_train == 0) & (X_train[:, 1] == 1))
#         X_train = X_train[biased_indices]
#         y_train = y_train[biased_indices]

#         scaler = MinMaxScaler()
#         X_train = torch.from_numpy(scaler.fit_transform(X_train)).float()
#         X_test = torch.from_numpy(scaler.transform(X_test)).float()

#         model.fit(X_train, y_train)
#         preds = model.predict(X_test)
#         probas = model.predict_proba(X_test)[:, 1]

#         acc = accuracy_score(y_test, preds)
#         f1 = f1_score(y_test, preds)
#         roc = roc_auc_score(y_test, probas)

#         metric_logger({"acc": acc, "f1": f1, "roc": roc})
#     metric_loggers.update(metric_logger.accumulate())
# meta_metric_logger = MetricLogger("meta")
# meta_metric_logger.log = metric_loggers
# meta_metric_logger.bar(nexperiments=len(gammas))


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
