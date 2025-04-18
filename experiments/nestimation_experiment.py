# stratified experiment
from laat.splitters import NShotSplitter
from laat.utils import MetricLogger, partial_class, NShotLogger, NShotLog, ShotLog, Metric, ExperimentLogger
from laat.models.skorch import SkorchLAATModel
from laat.datasets import LAATDataset
from laat.models.base import TrainRunInfo
from laat.models.featllm import FeatLLMLAATModel
from laat.models.tabpfn import TabPFNLAATModel
from laat.models.saint import SAINT
from laat.models.sklearn import GridSearchCVLAATModel
from tabpfn import TabPFNClassifier
from laat.models.laat import (
    LAATLAATModel,
    LAATComparisonClassifier,
    TorchLogisticRegression,
    TorchMLP,
    LAATUnweightedClassifier,
    LAATFewshotLAATModel,
)

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq


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
from laat.metrics import Accuracy, F1, ROCAUC
import numpy as np
from functools import partial
import random
from types import SimpleNamespace

from skorch.dataset import unpack_data
from skorch.classifier import NeuralNetBinaryClassifier

dataset_names = ["adult", "diabetes", "breast-ljubljana", "myocardial", "electricity", "bank", "cdc-diabetes"]
n_estimatess = [1, 2, 3, 4, 5, 6, 7, 8]
n_classes = 1
nrepetitions = 3
shots = [1, 5, 10]
metrics = [Accuracy(), F1(), ROCAUC()]
seed = 69

model_kwargs = {
    # "module": TorchLogisticRegression,
    "lr": 1e-2,
    "max_epochs": 200,
    "train_split": None,
    "optimizer": torch.optim.Adam,
    "verbose": False,
    "device": "cuda",
}

model_kwargs = {
    # "module": TorchLogisticRegression,
    "lr": 1e-2,
    "max_epochs": 200,
    "train_split": None,
    "optimizer": torch.optim.SGD,
    "optimizer__momentum": 0.9,
    "optimizer__nesterov": True,
    "verbose": False,
    "device": "cuda",
}

experiment_logger = ExperimentLogger("nestimation_experiment")
for dataset_name in dataset_names:
    dataset = LAATDataset.load(dataset_name, "laat/data")

    models = [
        partial_class(
            LAATLAATModel,
            model_name=f"laat_gpt-4o-mini_lr_{n_estimates}",
            model_class=partial_class(LAATUnweightedClassifier, module=TorchLogisticRegression, **model_kwargs),
            pandas_to_numpy_mapper=dataset.to_numpy,
            dataset=dataset,
            reasoning_llm=ChatOpenAI(model="gpt-4o-mini"),
            parsing_llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.0),
            gamma=100.0,
            n_estimates=n_estimates,
        )
        for n_estimates in n_estimatess
    ]
    # better generalization with small amounts of data
    meta_logger = NShotLogger(f"{dataset.dataset_name}")
    for model_class in models:
        model = model_class()
        print(model.model_name)
        model_logger = NShotLog(model_name=model.model_name)
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        for shot in shots:
            shot_logger = ShotLog(shot=shot)
            for repetition in tqdm(range(nrepetitions)):
                model = model_class(save_path=f"n_estimation_importances/{repetition}")
                X_train, X_test, y_train, y_test = NShotSplitter.split(X=dataset.X, y=dataset.y, shot=shot)
                model.train(X_train, y_train, TrainRunInfo(shot=shot, repetition=repetition))

                all_probas = model.predict_proba(X_test)
                preds = all_probas.argmax(-1)
                probas = all_probas[:, 1]
                for metric in metrics:
                    value = metric(
                        y_pred=preds, y_proba=probas, y_true=dataset.to_numpy(y=y_test)
                    )  #  / (orig_metrics[metric_name] + 1e-9)
                    shot_logger.update(metric_update=Metric(name=metric.metric_name, repetitions=[value]))
                model.clear()
            model_logger.update(shot_logger)
        meta_logger.update(nshot_log_update=model_logger)
    meta_logger.plot()
    meta_logger.save()

    experiment_logger.update(meta_logger)
    experiment_logger.save()
