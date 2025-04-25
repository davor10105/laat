# stratified experiment
from laat.splitters import NShotSplitter
from laat.utils import MetricLogger, partial_class, NShotLogger, NShotLog, ShotLog, Metric, ExperimentLogger
from laat.models.skorch import SkorchLAATModel
from laat.datasets import LAATDataset
from laat.models.base import TrainRunInfo
from laat.models.original_featllm_old import OriginalFeatLLMLAATModel
from laat.models.tabpfn import TabPFNLAATModel
from laat.models.sklearn import GridSearchCVLAATModel, CatBoostLAATModel
from tabpfn import TabPFNClassifier
from catboost import CatBoostClassifier
from laat.models.laat import (
    LAATLAATModel,
    LAATEnsembleLAATModel,
    LAATComparisonClassifier,
    TorchLogisticRegression,
    TorchMLP,
    TorchMLPSiLU,
    LAATUnweightedClassifier,
    LAATFewshotLAATModel,
    UpdateLAATClassifier,
    GammaDecay,
)

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
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
from laat.metrics import Accuracy, F1, ROCAUC, Precision, Recall
import numpy as np
from functools import partial
import random
from types import SimpleNamespace

from skorch.dataset import unpack_data
from skorch.classifier import NeuralNetBinaryClassifier

# dataset_names = ["adult", "diabetes", "breast-ljubljana", "myocardial", "electricity", "bank", "cdc-diabetes"]
# dataset_names = ["adult", "diabetes", "breast-ljubljana", "electricity", "bank", "cdc-diabetes", "myocardial"]
# dataset_names = ["diabetes", "breast-ljubljana", "myocardial", "electricity", "bank", "cdc-diabetes"]
dataset_names = [
    "adult",
    "diabetes",
    "breast-ljubljana",
    "myocardial",
    "electricity",
    "bank",
    "cdc-diabetes",
    "contraceptive",
    "bodyfat",
    "indian_liver",
]
n_classes = 1
nrepetitions = 20
shots = [1, 5, 10]
noise_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
metrics = [Accuracy(), F1(), ROCAUC(), Precision(), Recall()]
seed = 69

# model_kwargs = {
#     # "module": TorchLogisticRegression,
#     "lr": 0.1,
#     "max_epochs": 500,
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
    "max_epochs": 200,
    "train_split": None,
    "optimizer": torch.optim.Adam,
    "verbose": False,
    "device": "cuda",
}

experiment_logger = ExperimentLogger("noising_sweep")
for dataset_name in dataset_names:
    print(dataset_name)
    dataset = LAATDataset.load(dataset_name, "laat/data")

    models = [
        LAATLAATModel(
            model_name=f"laat_gpt-4o-mini_mlp_noise_{noise_ratio}",
            model_class=partial_class(UpdateLAATClassifier, module=TorchMLP, **model_kwargs),
            pandas_to_numpy_mapper=dataset.to_numpy,
            dataset=dataset,
            reasoning_llm=ChatOpenAI(model="gpt-4o-mini"),
            parsing_llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.0),
            gamma=100.0,
            n_estimates=5,
            noise_ratio=noise_ratio,
        )
        for noise_ratio in noise_ratios
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
        for shot in shots:
            shot_logger = ShotLog(shot=shot)
            for repetition in tqdm(
                range(nrepetitions)  # if "featllm" not in model.model_name else 10)
            ):  # 10 repetitions for featllm because of cost
                X_train, X_test, y_train, y_test = NShotSplitter.split(X=dataset.X, y=dataset.y, shot=shot)

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
                    value = metric(
                        y_pred=preds, y_proba=probas, y_true=dataset.to_numpy(y=y_test)
                    )  #  / (orig_metrics[metric_name] + 1e-9)
                    # print(metric.metric_name, value, end=" ")
                    shot_logger.update(metric_update=Metric(name=metric.metric_name, repetitions=[value]))
                model.clear()
            model_logger.update(shot_logger)
        meta_logger.update(nshot_log_update=model_logger)
    # meta_logger.plot()
    meta_logger.save()

    experiment_logger.update(meta_logger)
    experiment_logger.save()
