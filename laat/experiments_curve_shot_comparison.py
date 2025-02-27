# stratified experiment
from laat.datasets import (
    HeartDiseaseDataset,
    DiabetesDataset,  # works
    BankMarketingDataset,
    AdultDataset,  # works
    PhishingDataset,
    ThyroidCancerDataset,
    BloodDataset,
    CreditGDataset,
    MyocardialDataset,
    PC1Dataset,
    WineDataset,
    KC2Dataset,
    CDCDiabetesDataset,
    ViralPneumoniaDataset,
    BreastLjubljanaDataset,
    Rival10Dataset,
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
from tabpfn import TabPFNClassifier
from lightgbm import LGBMClassifier
from laat.other_models import SAINT
from torch.utils.data import TensorDataset, DataLoader
from types import SimpleNamespace


import torch
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit, KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler
from tqdm.auto import tqdm
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, loguniform
from skorch.callbacks import EarlyStopping
from hpsklearn import HyperoptEstimator, mlp_regressor, mlp_classifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import numpy as np
from functools import partial

from skorch.dataset import unpack_data
from skorch.classifier import NeuralNetBinaryClassifier


class LAATComparisonClassifier(NeuralNetBinaryClassifier):
    def __init__(self, *args, gamma, is_mse, llm_ratings, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.is_mse = is_mse
        self.llm_ratings = llm_ratings
        self.step_counter = 0

    def __str__(self):
        return f"{self.__class__.__name__}_{self.gamma}"

    def _get_attribution_loss(self, Xi, yi, y_pred, cls_loss):
        # (ne, tupane!) ###### DODAJ GRADIJENT NA LOSS, NE NA SUM!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ##### POGLEDAJ AKO MOZES RANDOM SAMPLING POZICIJA?
        attributions = torch.autograd.grad(y_pred.mean(), Xi, create_graph=True, retain_graph=True)[0] * Xi
        llm_attributions = F.normalize(self.llm_ratings.to(attributions.device) * Xi, dim=-1) * attributions.norm(
            dim=-1, keepdim=True
        )

        att_loss = nn.MSELoss()(F.normalize(attributions, dim=-1), F.normalize(llm_attributions, dim=-1).detach())
        return att_loss

    def train_step_single(self, batch, **fit_params):
        self._set_training(True)
        Xi, yi = unpack_data(batch)
        Xi.requires_grad = True
        y_pred = self.infer(Xi, **fit_params)
        loss = self.get_loss(y_pred, yi, X=Xi, training=True)

        # calculate attribution loss
        # epsilon-LRP
        # epsilon = 0.1
        # output_relevance = torch.ones_like(y_pred) / (y_pred + epsilon)
        # attributions = torch.autograd.grad(y_pred, Xi, output_relevance, create_graph=True, retain_graph=True)[0] * Xi

        # integrated gradients
        # ig_iters = 20
        # batch_blank = torch.zeros_like(Xi)
        # mean_grad = torch.zeros_like(Xi)
        # for i in range(1, ig_iters + 1):
        #     x = (batch_blank + i / ig_iters * (Xi - batch_blank)).detach()
        #     x.requires_grad = True
        #     y = self.infer(x, **fit_params)
        #     grad = torch.autograd.grad(y.mean(), x)[0]
        #     mean_grad += grad / ig_iters
        # attributions = (Xi - batch_blank) * mean_grad
        # print(Xi.shape)
        # attributions = torch.autograd.grad(y_pred.mean(), Xi, create_graph=True, retain_graph=True)[0] * Xi
        # llm_attributions = F.normalize(self.llm_ratings.to(attributions.device) * Xi, dim=-1) * attributions.norm(
        #     dim=-1, keepdim=True
        # )

        # max to one
        # attributions = attributions / (attributions.abs().max(-1, keepdim=True)[0] + 1e-9)
        # llm_attributions = llm_attributions / (llm_attributions.abs().max(-1, keepdim=True)[0] + 1e-9)

        # att_loss = nn.MSELoss()(attributions, llm_attributions.detach())
        # att_loss = nn.MSELoss()(F.normalize(attributions, dim=-1), F.normalize(llm_attributions, dim=-1).detach())
        # if self.is_mse:
        #     att_loss = nn.MSELoss()(F.normalize(attributions, dim=-1), F.normalize(llm_attributions, dim=-1).detach())
        # else:
        #     att_loss = (1 - (F.cosine_similarity(attributions, llm_attributions.detach()) * 2 - 1)).mean()
        att_loss = self._get_attribution_loss(Xi, yi, y_pred, loss)
        # loss += self.gamma * (nsteps - self.step_counter) / nsteps * att_loss
        attribution_factor = att_loss.item() / (att_loss.item() + loss.item())
        loss += self.gamma / (attribution_factor + 1e-9) * att_loss
        # loss += self.gamma * att_loss
        loss.backward()

        self.step_counter += 1

        return {
            "loss": loss,
            "y_pred": y_pred,
        }

    def validation_step(self, batch, **fit_params):
        self._set_training(False)
        Xi, yi = unpack_data(batch)
        Xi.requires_grad = True
        y_pred = self.infer(Xi, **fit_params)
        loss = self.get_loss(y_pred, yi, X=Xi, training=False)

        # calculate attribution loss
        # attributions = torch.autograd.grad(y_pred.mean(), Xi, create_graph=True, retain_graph=True)[0] * Xi
        # llm_attributions = F.normalize(self.llm_ratings.to(attributions.device), dim=-1) * attributions.norm(
        #     dim=-1, keepdim=True
        # )

        # att_loss = nn.MSELoss()(attributions, llm_attributions.detach())
        # att_loss = (1 - (F.cosine_similarity(attributions, llm_attributions.detach()) * 2 - 1)).mean()
        # att_loss = nn.MSELoss()(F.normalize(attributions, dim=-1), F.normalize(llm_attributions, dim=-1).detach())
        att_loss = self._get_attribution_loss(Xi, yi, y_pred, loss)
        loss += self.gamma * att_loss

        return {
            "loss": loss,
            "y_pred": y_pred,
        }


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
shots = [1, 5, 10]
nsteps = 100
seed = 69
dataset = DiabetesDataset.load()

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

distributions = dict(optimizer__weight_decay=uniform(1e-1, 2e0), lr=uniform(1e-1, 2.0))


def split(X: torch.tensor, y: torch.tensor) -> tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    print(X)
    sample_size = X.shape[0]
    train_size = sample_size // 2
    return X_train[:train_size], y_train[:train_size], X_test[train_size:], y_test[train_size:]


gammas = [1e2]
gammas_fuckery = [1.0]
models = {}
models = {
    f"laat_lr_{gamma}_compare": partial_class(
        LAATComparisonClassifier,
        module=TorchLogisticRegression,
        gamma=gamma,
        is_mse=True,
        train_split=None,
        **model_kwargs,
    )
    for gamma in gammas_fuckery
}
# models.update(
#     {
#         f"laat_lr_{gamma}": partial_class(
#             LAATClassifier,
#             module=TorchLogisticRegression,
#             gamma=gamma,
#             is_mse=True,
#             train_split=None,
#             **model_kwargs,
#         )
#         for gamma in gammas
#     }
# )
# models.update(
#     {
#         f"laat_mlp_{gamma}_cosine": partial_class(
#             LAATComparisonClassifier, module=TorchMLP, gamma=gamma, is_mse=True, train_split=None, **model_kwargs
#         )
#         for gamma in gammas
#     }
# )

# models.update(
#     {
#         f"laat_lr_{gamma}": partial_class(
#             LAATClassifier,
#             module=TorchLogisticRegression,
#             gamma=gamma,
#             is_mse=True,
#             train_split=None,
#             **model_kwargs,
#         )
#         for gamma in gammas
#     }
# )

# models.update(
#     {
#         f"laat_lr_{gamma}_cosine": partial_class(
#             LAATClassifier, module=TorchLogisticRegression, gamma=gamma, is_mse=False, train_split=None, **model_kwargs
#         )
#         for gamma in gammas
#     }
# )

nclasses = dataset.y.max().int().item()
additional_models = {
    # "catboost": partial_class(CatBoostClassifier, verbose=False),
    # "xgb": XGBClassifier,
    # "saint": partial_class(
    #     SAINT,
    #     params=dict(dim=64, depth=6, heads=4, dropout=0.1),
    #     args=SimpleNamespace(
    #         use_gpu=True,
    #         batch_size=128,
    #         val_batch_size=256,
    #         data_parallel=False,
    #         model_id="saint",
    #         cat_idx=[],
    #         num_features=dataset.X.shape[-1],
    #         num_classes=nclasses,
    #         objective="binary" if nclasses == 1 else "classification",
    #         lr=0.00003,
    #         epochs=100,
    #     ),
    # ),
    # "tabfpn": partial_class(TabPFNClassifier, device="cuda"),
    # "lgbm": LGBMClassifier,
    "lr": LogisticRegression,
    # "mlp": partial_class(MLPClassifier, max_iter=500),
    # "knn": KNeighborsClassifier,
    # "random_forest": RandomForestClassifier,
}
models.update(additional_models)

metrics = {"acc": accuracy_score, "f1": f1_score, "roc": roc_auc_score}

# train original model on full dataset first
X_train, X_test, y_train, y_test = train_test_split(dataset.X, dataset.y, test_size=0.80)
scaler = MinMaxScaler()
X_train = torch.from_numpy(scaler.fit_transform(X_train)).float()
X_test = torch.from_numpy(scaler.transform(X_test)).float()
model = XGBClassifier()
model.fit(X_train, y_train)
preds = model.predict(X_test)
probas = model.predict_proba(X_test)[:, 1]

orig_metrics = {}
for metric_name, metric in metrics.items():
    orig_metrics[metric_name] = metric(y_test, preds)

# better generalization with small amounts of data
meta_logger = NShotLogger(f"{dataset.__class__.__name__}_{attribution_method}")
for model_name, model_init in models.items():
    print(model_name)
    model_logger = NShotLog(model_name=model_name)
    np.random.seed(seed)
    torch.manual_seed(seed)
    for shot in shots:
        shot_logger = ShotLog(shot=shot)
        for repetition in tqdm(range(nrepetitions)):
            model = model_init()
            if model_name == "knn":
                model = model_init(n_neighbors=shot)
            # if "laat" in model_name:
            #     model = RandomizedSearchCV(model, distributions, random_state=seed)
            X_train, y_train, X_test, y_test = NShotSplitter.split(dataset.X, dataset.y, shot)

            if model_name != "tabfpn":
                # preprocess
                # scaler = QuantileTransformer(n_quantiles=min(X_train.shape[0], 1000))
                scaler = MinMaxScaler()
                # scaler = StandardScaler()
                X_train = np.array(scaler.fit_transform(X_train), dtype=np.float32)
                X_test = np.array(scaler.transform(X_test), dtype=np.float32)

                y_train, y_test = np.array(y_train, dtype=np.float32), np.array(y_test, dtype=np.float32)

            model.fit(X_train, y_train)

            if model_name != "tabfpn":
                all_probas = model.predict_proba(X_test)
                preds = all_probas.argmax(-1)
                probas = all_probas[:, 1]
            else:
                # tabfpn batching
                all_probas = []
                for X_test_batch, y_test_batch in DataLoader(
                    TensorDataset(X_test, y_test), batch_size=10000, shuffle=False
                ):
                    probas_all = model.predict_proba(X_test_batch)
                    all_probas.append(probas_all)
                all_probas = np.concatenate(all_probas, axis=0)
                preds = all_probas.argmax(-1)
                probas = all_probas[:, 1]
                del model
                torch.cuda.empty_cache()

            # if "laat" in model_name:
            #     print(model.best_params_)

            for metric_name, metric_function in metrics.items():
                value = metric_function(y_test, preds)  #  / (orig_metrics[metric_name] + 1e-9)
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
