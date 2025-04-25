# stratified experiment
from laat.splitters import NShotSplitter, SkewedSplitter
from laat.utils import MetricLogger, partial_class, NShotLogger, NShotLog, ShotLog, Metric, ExperimentLogger
from laat.models.skorch import SkorchLAATModel
from laat.datasets import LAATDataset
from laat.models.base import TrainRunInfo
from laat.models.original_featllm_old import OriginalFeatLLMLAATModel
from laat.models.tabpfn import TabPFNLAATModel
from laat.models.saint import SAINT
from laat.models.sklearn import GridSearchCVLAATModel, CatBoostLAATModel
from catboost import CatBoostClassifier
from tabpfn import TabPFNClassifier
from laat.models.laat import (
    LAATLAATModel,
    LAATEnsembleLAATModel,
    LAATComparisonClassifier,
    TorchLogisticRegression,
    TorchMLP,
    LAATUnweightedClassifier,
    UpdateLAATClassifier,
    LAATFewshotLAATModel,
)
from langchain_google_genai import ChatGoogleGenerativeAI

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import pickle


import torch
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit, KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler
from tqdm.auto import tqdm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.stats import uniform, loguniform
from skorch.callbacks import EarlyStopping
from skorch.classifier import NeuralNetBinaryClassifier
from hpsklearn import HyperoptEstimator, mlp_regressor, mlp_classifier
from laat.metrics import Accuracy, F1, ROCAUC, DemographicParityRatio
import numpy as np
from functools import partial
import random
from types import SimpleNamespace

from skorch.dataset import unpack_data
from skorch.classifier import NeuralNetBinaryClassifier

# featllm fail - "breast-ljubljana" "myocardial",
# dataset_names = ["diabetes", "adult"]
# dataset_names = ["electricity", "bank", "cdc-diabetes"]
dataset_names = [
    # "diabetes",
    # "breast-ljubljana",
    # "myocardial",
    # "cdc-diabetes",
    # "contraceptive",
    # "bodyfat",
    "indian_liver",
]  # ["diabetes", "breast-ljubljana", "myocardial", "cdc-diabetes"]
# dataset_names = ["cdc-diabetes"]
n_classes = 1
nrepetitions = 20
shots = [0]
metrics = [Accuracy(), F1(), ROCAUC()]  # , DemographicParityRatio()]
seed = 69

# model_kwargs = {
#     # "module": TorchLogisticRegression,
#     "lr": 0.1,
#     "max_epochs": 20,
#     "train_split": None,
#     "optimizer": torch.optim.SGD,
#     "optimizer__momentum": 0.9,
#     "optimizer__weight_decay": 0,
#     "verbose": False,
#     "device": "cuda",
# }

model_kwargs = {
    # "module": TorchLogisticRegression,
    "lr": 1e-2,
    "max_epochs": 20,
    "train_split": None,
    "optimizer": torch.optim.Adam,
    "verbose": False,
    "device": "cuda",
}

experiment_logger = ExperimentLogger("skewness_sweep")
for dataset_name in dataset_names:
    print(dataset_name)
    dataset = LAATDataset.load(dataset_name, "laat/data")

    models = [
        TabPFNLAATModel(
            model_name="tabfpn",
            model_class=partial_class(TabPFNClassifier, device="cuda"),
            pandas_to_numpy_mapper=dataset.to_numpy,
        ),
        OriginalFeatLLMLAATModel(
            model_name="featllm_gemini-2.0-flash-lite-preview-02-05",
            model_class=partial_class(LogisticRegression, max_iter=1000),
            pandas_to_numpy_mapper=dataset.to_numpy,
            dataset=dataset,
            reasoning_llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite-preview-02-05", temperature=0.5),
            parsing_llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.0),
            n_estimators=20,
            save_path="./skewness_original_featllm_functions",
        ),
        OriginalFeatLLMLAATModel(
            model_name="featllm_gpt-4o-mini",
            model_class=partial_class(LogisticRegression, max_iter=1000),
            pandas_to_numpy_mapper=dataset.to_numpy,
            dataset=dataset,
            reasoning_llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.5),
            parsing_llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.0),
            n_estimators=20,
            save_path="./skewness_original_featllm_functions",
        ),
        LAATLAATModel(
            model_name=f"laat_llama-3.3-70b-versatile_mlp",
            model_class=partial_class(UpdateLAATClassifier, module=TorchMLP, **model_kwargs),
            pandas_to_numpy_mapper=dataset.to_numpy,
            dataset=dataset,
            reasoning_llm=ChatGroq(model="llama-3.3-70b-versatile"),
            parsing_llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.0),
            gamma=100.0,
            n_estimates=5,
        ),
        LAATLAATModel(
            model_name=f"laat_gemini-2.0-flash-lite-preview-02-05_mlp",
            model_class=partial_class(UpdateLAATClassifier, module=TorchMLP, **model_kwargs),
            pandas_to_numpy_mapper=dataset.to_numpy,
            dataset=dataset,
            reasoning_llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite-preview-02-05"),
            parsing_llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.0),
            gamma=100.0,
            n_estimates=5,
        ),
        LAATLAATModel(
            model_name=f"laat_gpt-4o-mini_mlp",
            model_class=partial_class(UpdateLAATClassifier, module=TorchMLP, **model_kwargs),
            pandas_to_numpy_mapper=dataset.to_numpy,
            dataset=dataset,
            reasoning_llm=ChatOpenAI(model="gpt-4o-mini"),
            parsing_llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.0),
            gamma=100.0,
            n_estimates=5,
        ),
        LAATLAATModel(
            model_name=f"laat_llama-3.3-70b-versatile_lr",
            model_class=partial_class(UpdateLAATClassifier, module=TorchLogisticRegression, **model_kwargs),
            pandas_to_numpy_mapper=dataset.to_numpy,
            dataset=dataset,
            reasoning_llm=ChatGroq(model="llama-3.3-70b-versatile"),
            parsing_llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.0),
            gamma=100.0,
            n_estimates=5,
        ),
        LAATLAATModel(
            model_name=f"laat_gemini-2.0-flash-lite-preview-02-05_lr",
            model_class=partial_class(UpdateLAATClassifier, module=TorchLogisticRegression, **model_kwargs),
            pandas_to_numpy_mapper=dataset.to_numpy,
            dataset=dataset,
            reasoning_llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite-preview-02-05"),
            parsing_llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.0),
            gamma=100.0,
            n_estimates=5,
        ),
        LAATLAATModel(
            model_name=f"laat_gpt-4o-mini_lr",
            model_class=partial_class(UpdateLAATClassifier, module=TorchLogisticRegression, **model_kwargs),
            pandas_to_numpy_mapper=dataset.to_numpy,
            dataset=dataset,
            reasoning_llm=ChatOpenAI(model="gpt-4o-mini"),
            parsing_llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.0),
            gamma=100.0,
            n_estimates=5,
        ),
        GridSearchCVLAATModel(
            model_name="lr",
            model_class=LogisticRegression,
            param_grid={"C": [100, 10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5], "penalty": ["l2"]},
            scoring="roc_auc",
            pandas_to_numpy_mapper=dataset.to_numpy,
        ),
        GridSearchCVLAATModel(
            model_name="mlp",
            model_class=partial_class(MLPClassifier, max_iter=1000),
            param_grid={
                "alpha": [0.001, 0.01, 0.1, 1, 10],
                "learning_rate_init": [0.1, 0.01, 0.001, 0.0001],
            },
            scoring="roc_auc",
            pandas_to_numpy_mapper=dataset.to_numpy,
        ),
        GridSearchCVLAATModel(
            model_name="xgb",
            model_class=XGBClassifier,
            param_grid={
                "max_depth": [2, 4, 6, 8, 10],
                "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
                "lambda": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
                "eta": [0.01, 0.03, 0.1, 0.3],
            },
            scoring="roc_auc",
            pandas_to_numpy_mapper=dataset.to_numpy,
        ),
        GridSearchCVLAATModel(
            model_name="rf",
            model_class=RandomForestClassifier,
            param_grid={
                "bootstrap": [True, False],
                "max_depth": [2, 4, 6, 8, 10],  # , 12],
                "n_estimators": [2, 4, 8, 16, 32, 64],  # ], 128, 256],
            },
            scoring="roc_auc",
            pandas_to_numpy_mapper=dataset.to_numpy,
        ),
        CatBoostLAATModel(
            model_name="catboost",
            model_class=CatBoostClassifier,
            pandas_to_numpy_mapper=dataset.to_numpy,
            dataset=dataset,
            param_grid={
                "colsample_bylevel": [0.01, 0.03, 0.06, 0.1],
                "boosting_type": ["Ordered", "Plain"],
                "depth": [2, 4, 6, 8, 10],
            },
        ),
    ]
    # better generalization with small amounts of data
    # with open("FeatLLM/train_logs/diabetes2502211835.pkl", "rb") as f:
    #     meta_logger = pickle.load(f)
    meta_logger = NShotLogger(f"{dataset.dataset_name}")
    for model in models:
        print(model.model_name)
        model_logger = NShotLog(model_name=model.model_name)
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        shot = int(0.8 * len(dataset.X))
        shot_logger = ShotLog(shot=shot)
        for repetition in tqdm(range(nrepetitions)):
            X_train, X_test, y_train, y_test, sensitive_features = SkewedSplitter.split(
                dataset_name=dataset_name, X=dataset.X, y=dataset.y, delta=0, train_size=0.8
            )
            model.train(
                X_train,
                y_train,
                TrainRunInfo(
                    shot=shot,
                    repetition=repetition,
                    kwargs={
                        "X_test": X_test,
                        "y_test": y_test,
                    },
                ),
            )

            all_probas = model.predict_proba(X_test)
            preds = all_probas.argmax(-1)
            probas = all_probas[:, 1]
            for metric in metrics:
                if isinstance(metric, DemographicParityRatio):
                    value = metric(
                        y_pred=preds,
                        y_proba=probas,
                        y_true=dataset.to_numpy(y=y_test),
                        sensitive_features=sensitive_features,
                    )
                else:
                    value = metric(
                        y_pred=preds, y_proba=probas, y_true=dataset.to_numpy(y=y_test)
                    )  #  / (orig_metrics[metric_name] + 1e-9)
                # print(metric, value)
                shot_logger.update(metric_update=Metric(name=metric.metric_name, repetitions=[value]))
            model.clear()
        model_logger.update(shot_logger)
        meta_logger.update(nshot_log_update=model_logger)
    # meta_logger.plot()
    meta_logger.save()

    experiment_logger.update(meta_logger)
    experiment_logger.save()
