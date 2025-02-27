# stratified experiment
from laat.splitters import NShotSplitter
from laat.utils import MetricLogger, partial_class, NShotLogger, NShotLog, ShotLog, Metric
from laat.models.skorch import SkorchLAATModel
from laat.datasets import LAATDataset
from laat.models.base import TrainRunInfo
from laat.models.featllm import FeatLLMLAATModel
from laat.models.laat import (
    LAATLAATModel,
    LAATComparisonClassifier,
    TorchLogisticRegression,
    LAATUnweightedClassifier,
    LAATMulticlassClassifier,
)

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from langchain_openai import ChatOpenAI


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
from laat.metrics import Accuracy, F1, ROCAUC, MacroF1, OVOROCAUC
import numpy as np
from functools import partial
import random

from skorch.dataset import unpack_data
from skorch.classifier import NeuralNetBinaryClassifier

dataset_name = "contraceptive-multiclass"
n_classes = 3

nrepetitions = 2
shots = [1, 5, 10]
seed = 69
dataset = LAATDataset.load(dataset_name, "laat/data")

model_kwargs = {
    "module": partial_class(TorchLogisticRegression, n_classes=n_classes),
    "lr": 0.1,
    "max_epochs": 100,
    "train_split": None,
    "optimizer": torch.optim.SGD,
    "optimizer__momentum": 0.9,
    "optimizer__weight_decay": 1e-4,
    "verbose": False,
    "device": "cuda",
    "criterion": nn.CrossEntropyLoss,
}

models = [
    LAATLAATModel(
        model_name="laat",
        model_class=partial_class(LAATMulticlassClassifier, **model_kwargs),
        pandas_to_numpy_mapper=dataset.to_numpy,
        dataset=dataset,
        reasoning_llm=ChatOpenAI(model="gpt-4o-mini"),
        parsing_llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.0),
        gamma=1.0,
        n_estimates=5,
    ),
    LAATLAATModel(
        model_name="laat_0",
        model_class=partial_class(LAATMulticlassClassifier, **model_kwargs),
        pandas_to_numpy_mapper=dataset.to_numpy,
        dataset=dataset,
        reasoning_llm=ChatOpenAI(model="gpt-4o-mini"),
        parsing_llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.0),
        gamma=0.0,
        n_estimates=5,
    ),
    FeatLLMLAATModel(
        model_name="featllm",
        model_class=partial_class(LogisticRegression, max_iter=1000),
        pandas_to_numpy_mapper=dataset.to_numpy,
        dataset=dataset,
        reasoning_llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.5),
        parsing_llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.0),
        n_estimators=2,
    ),
    SkorchLAATModel(model_name="lr", model_class=LogisticRegression, pandas_to_numpy_mapper=dataset.to_numpy),
    SkorchLAATModel(
        model_name="mlp",
        model_class=partial_class(MLPClassifier, max_iter=1000),
        pandas_to_numpy_mapper=dataset.to_numpy,
    ),
]

metrics = [Accuracy(), MacroF1(), OVOROCAUC()]
# better generalization with small amounts of data
meta_logger = NShotLogger(f"{dataset.dataset_name}")
for model in models:
    print(model.model_name)
    model_logger = NShotLog(model_name=model.model_name)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    for shot in shots:
        shot_logger = ShotLog(shot=shot)
        for repetition in tqdm(range(nrepetitions)):
            X_train, X_test, y_train, y_test = NShotSplitter.split(X=dataset.X, y=dataset.y, shot=shot)
            model.train(X_train, y_train, TrainRunInfo(shot=shot, repetition=repetition))

            print(X_train.to_numpy().shape)

            all_probas = model.predict_proba(X_test)
            preds = all_probas.argmax(-1)
            probas = all_probas
            if model.model_name == "laat":
                print(model.model.module_.model.weight)
            if model.model_name == "lr":
                print(model.model.coef_)
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
