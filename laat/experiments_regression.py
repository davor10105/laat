# stratified experiment
from laat.datasets import HeartDiseaseDataset, DiabetesDataset, BankMarketingDataset, CaliforniaHousingDataset
from laat.models import LAATRegressor
from sklearn.neural_network import MLPClassifier
from laat.utils import MetricLogger


import torch
from torch import nn
from sklearn.model_selection import StratifiedShuffleSplit, KFold
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler
from tqdm.auto import tqdm
from skorch.callbacks import EarlyStopping
from hpsklearn import HyperoptEstimator, mlp_regressor, mlp_classifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import hyperopt as hp


class PModel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.model = nn.LazyLinear(1)
        self.model = nn.Sequential(nn.LazyLinear(100), nn.ReLU(), nn.LazyLinear(1))

    def forward(self, x):
        return self.model(x)


train_size_multiplier = 0.01  # how many of the positive class examples are there
nrepetitions = 5
nsteps = 1000
seed = 69
dataset = CaliforniaHousingDataset.load()

model_kwargs = {
    "module": PModel,
    "llm_ratings": dataset.llm_ratings,
    "lr": 0.01,
    "max_epochs": nsteps,
    "train_split": None,
    "optimizer": torch.optim.SGD,
    "optimizer__momentum": 0.9,
    "optimizer__weight_decay": 1e-4,
    "verbose": False,
    # callbacks=[("es", EarlyStopping(patience=16))],
}

# better generalization with small amounts of data
metric_loggers = {}
gammas = [0, 1e-1, 1]
for gamma in gammas:
    model = LAATRegressor(
        gamma=gamma,
        **model_kwargs,
    )
    # model = HyperoptEstimator(regressor=mlp_regressor(model, learning_rate_init=hp.uniform(name, 1e-4, 0.1)))
    metric_logger = MetricLogger(f"stratified_{gamma}")
    np.random.seed(seed)
    for _ in range(nrepetitions):
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.X, dataset.y, train_size=train_size_multiplier, shuffle=True
        )

        scaler = QuantileTransformer(n_quantiles=min(X_train.shape[0], 1000))
        X_train = torch.from_numpy(scaler.fit_transform(X_train)).float()
        X_test = torch.from_numpy(scaler.transform(X_test)).float()

        y_mean, y_std = y_train.mean(), y_train.std()
        y_train = (y_train - y_mean) / y_std
        y_test = (y_test - y_mean) / y_std

        model.fit(X_train, y_train.view(-1, 1))

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)

        metric_logger({"r2": r2, "rmse": rmse})
    metric_loggers.update(metric_logger.accumulate())
meta_metric_logger = MetricLogger("meta")
meta_metric_logger.log = metric_loggers
meta_metric_logger.bar(nexperiments=len(gammas))

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
# gammas = [0, 1e-1]
# for gamma in gammas:
#     model = LAATRegressor(
#         gamma=gamma,
#         **model_kwargs,
#         # callbacks=[("es", EarlyStopping(patience=5))],
#     )
#     model = HyperoptEstimator(regressor=mlp_regressor(model))

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

#         model.fit(X_train, y_train.flatten())

#         y_pred = model.predict(X_test)
#         r2 = r2_score(y_test, y_pred)
#         rmse = root_mean_squared_error(y_test, y_pred)

#         metric_logger({"r2": r2, "rmse": rmse})
#     metric_loggers.update(metric_logger.accumulate())
# meta_metric_logger = MetricLogger("meta")
# meta_metric_logger.log = metric_loggers
# meta_metric_logger.bar(nexperiments=len(gammas))
