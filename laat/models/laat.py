import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.base import BaseEstimator
from typing import Optional, Callable, Type, Any
from abc import ABC
from laat.models.base import LAATModel, TrainRunInfo
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.base import RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from laat.prompts.laat import (
    FEATURE_IMPORTANCE_PROMPT,
    FEATURE_IMPORTANCE_FEWSHOT_PROMPT,
    FEATURE_IMPORTANCE_FORMATTING_PROMPT,
)
from laat.datasets import LAATDataset
from pathlib import Path
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
import random

from skorch.dataset import unpack_data
from skorch.classifier import NeuralNetBinaryClassifier, NeuralNetClassifier
from skorch.callbacks import Callback

from tqdm.auto import tqdm


class LAATComparisonClassifier(NeuralNetBinaryClassifier):
    def __init__(self, importance_scores: np.array, gamma: float, **kwargs):
        super().__init__(**kwargs)

        self.importance_scores = torch.from_numpy(importance_scores).float()
        self.gamma = gamma

    def __str__(self):
        return f"{self.__class__.__name__}_{self.gamma}"

    def _get_attribution_loss(self, Xi, yi, y_pred, cls_loss):
        attributions = torch.autograd.grad(y_pred.mean(), Xi, create_graph=True, retain_graph=True)[0] * Xi
        llm_importance_scores = F.normalize(
            self.importance_scores.to(attributions.device) * Xi, dim=-1
        ) * attributions.norm(dim=-1, keepdim=True)
        att_loss = nn.MSELoss()(F.normalize(attributions, dim=-1), F.normalize(llm_importance_scores, dim=-1).detach())
        return att_loss

    def train_step_single(self, batch, **fit_params):
        self._set_training(True)
        Xi, yi = unpack_data(batch)
        Xi.requires_grad = True
        y_pred = self.infer(Xi, **fit_params)
        loss = self.get_loss(y_pred, yi, X=Xi, training=True)
        att_loss = self._get_attribution_loss(Xi, yi, y_pred, loss)
        attribution_factor = att_loss.item() / (att_loss.item() + loss.item())
        loss += self.gamma / (attribution_factor + 1e-9) * att_loss
        # loss += self.gamma * att_loss
        loss.backward()

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
        att_loss = self._get_attribution_loss(Xi, yi, y_pred, loss)
        attribution_factor = att_loss.item() / (att_loss.item() + loss.item())
        loss += self.gamma / (attribution_factor + 1e-9) * att_loss
        # loss += self.gamma * att_loss

        return {
            "loss": loss,
            "y_pred": y_pred,
        }


class LAATUnweightedClassifier(NeuralNetBinaryClassifier):
    def __init__(self, importance_scores: np.array, gamma: float, attribution_method: str = "inputxgradient", **kwargs):
        super().__init__(**kwargs)

        self.importance_scores = torch.from_numpy(importance_scores).float()
        self.gamma = gamma
        self.attribution_method = attribution_method

    def __str__(self):
        return f"{self.__class__.__name__}_{self.gamma}"

    def _get_attribution_loss(self, Xi, yi, y_pred, cls_loss, fit_params):
        if self.attribution_method == "inputxgradient":
            attributions = torch.autograd.grad(y_pred.mean(), Xi, create_graph=True, retain_graph=True)[0] * Xi
        elif self.attribution_method == "integratedgradients":
            # integrated gradients
            ig_iters = 10
            batch_blank = torch.zeros_like(Xi)
            mean_grad = torch.zeros_like(Xi)
            for i in range(1, ig_iters + 1):
                x = (batch_blank + i / ig_iters * (Xi - batch_blank)).detach()
                x.requires_grad = True
                y = self.infer(x, **fit_params)
                grad = torch.autograd.grad(y.mean(), x)[0]
                mean_grad += grad / ig_iters
            attributions = (Xi - batch_blank) * mean_grad

        importance_scores = self.importance_scores.to(attributions.device)
        llm_importance_scores = importance_scores.abs().exp() * importance_scores.sign() * Xi
        # llm_importance_scores = importance_scores * Xi
        att_loss = nn.MSELoss()(F.normalize(attributions, dim=-1), F.normalize(llm_importance_scores, dim=-1).detach())
        return att_loss

    def train_step_single(self, batch, **fit_params):
        self._set_training(True)
        Xi, yi = unpack_data(batch)
        Xi.requires_grad = True
        y_pred = self.infer(Xi, **fit_params)
        loss = self.get_loss(y_pred, yi, X=Xi, training=True)
        att_loss = self._get_attribution_loss(Xi, yi, y_pred, loss, fit_params=fit_params)

        loss += self.gamma * att_loss
        loss.backward()

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
        att_loss = self._get_attribution_loss(Xi, yi, y_pred, loss, fit_params=fit_params)
        loss += self.gamma * att_loss

        return {
            "loss": loss,
            "y_pred": y_pred,
        }


class GammaDecay(Callback):
    def __init__(self, decay_factor: float = 0.95) -> None:
        self.decay_factor = decay_factor

    # This runs after every epoch
    def on_epoch_end(self, net, **kwargs):
        net.gamma *= self.decay_factor


class UpdateLAATClassifier(NeuralNetBinaryClassifier):
    def __init__(self, importance_scores: np.array, gamma: float, attribution_method: str = "inputxgradient", **kwargs):
        super().__init__(**kwargs)

        self.importance_scores = torch.from_numpy(importance_scores).float()
        self.gamma = gamma
        self.attribution_method = attribution_method

    def __str__(self):
        return f"{self.__class__.__name__}_{self.gamma}"

    def _get_attribution_loss(self, Xi, yi, y_pred, cls_loss, fit_params):
        if self.attribution_method == "inputxgradient":
            attributions = torch.autograd.grad(y_pred.mean(), Xi, create_graph=True, retain_graph=True)[0]
        elif self.attribution_method == "integratedgradients":
            # integrated gradients
            ig_iters = 10
            batch_blank = torch.zeros_like(Xi)
            mean_grad = torch.zeros_like(Xi)
            for i in range(1, ig_iters + 1):
                x = (batch_blank + i / ig_iters * (Xi - batch_blank)).detach()
                x.requires_grad = True
                y = self.infer(x, **fit_params)
                grad = torch.autograd.grad(y.mean(), x)[0]
                mean_grad += grad / ig_iters
            attributions = (Xi - batch_blank) * mean_grad

        importance_scores = self.importance_scores.to(attributions.device)
        llm_importance_scores = importance_scores.abs().exp() * importance_scores.sign()
        # llm_importance_scores = importance_scores * Xi
        att_loss = nn.MSELoss()(
            F.normalize(attributions.mean(0, keepdim=True), dim=-1), F.normalize(llm_importance_scores, dim=-1).detach()
        )
        return att_loss

    def train_step_single(self, batch, **fit_params):
        self._set_training(True)
        Xi, yi = unpack_data(batch)
        Xi.requires_grad = True
        y_pred = self.infer(Xi, **fit_params)
        loss = self.get_loss(y_pred, yi, X=Xi, training=True)
        att_loss = self._get_attribution_loss(Xi, yi, y_pred, loss, fit_params=fit_params)

        loss += self.gamma * att_loss
        loss.backward()

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
        att_loss = self._get_attribution_loss(Xi, yi, y_pred, loss, fit_params=fit_params)
        loss += self.gamma * att_loss

        return {
            "loss": loss,
            "y_pred": y_pred,
        }


class TorchLogisticRegression(nn.Module):
    def __init__(self, n_classes: int = 1):
        super().__init__()
        self.model = nn.LazyLinear(n_classes)

    def forward(self, x):
        return self.model(x)


class TorchMLP(nn.Module):
    def __init__(self, n_classes: int = 1):
        super().__init__()
        self.model = nn.Sequential(nn.LazyLinear(100), nn.ReLU(), nn.LazyLinear(n_classes))

    def forward(self, x):
        return self.model(x)


class TorchMLPSiLU(nn.Module):
    def __init__(self, n_classes: int = 1):
        super().__init__()
        self.model = nn.Sequential(
            nn.LazyLinear(100), nn.ReLU(), nn.LazyLinear(100), nn.ReLU(), nn.LazyLinear(n_classes)
        )

    def forward(self, x):
        return self.model(x)


class FeatureScore(BaseModel):
    feature_name: str = Field(description="the name of the extracted feature")
    importance_score: int = Field(description="the importance score of the extracted feature")


class ImportanceScores(BaseModel):
    importance_scores: list[FeatureScore] = Field(
        description="a list of feature importance scores related to a class found in the text, in order of their appearance"
    )


class LAATUtilities:
    @staticmethod
    def convert_categoricals_to_dummy(X: pd.DataFrame) -> pd.DataFrame:
        return pd.get_dummies(X)

    @staticmethod
    def dataset_to_prompt(dataset: LAATDataset) -> tuple[str, str]:
        """Encode a dataset as a LAAT prompt.

        Args:
            dataset (LAATDataset): Dataset to encode.
            column_names (Optional[list[str]]): column_names to use (in case of feature bagging). Defaults to None.

        Returns:
            tuple[str, str]: Dataset converted to prompt and used features (in feature bagging scenarios)
        """
        prompt = f"Task: {dataset.feature_descriptions['#DatasetTask#']}\nFeatures:\n"
        feature_template = "{index}. {feature_name}: {feature_description}"
        feature_prompts = []
        # check if dataset is in right format
        assert (
            len(list(filter(lambda name: len(name.split("_")) > 1, dataset.X.columns.tolist()))) == 0
        ), "No dataset features can contain '_' as a part of their name"
        # encode categorical features as binary flags
        X_dummy = LAATUtilities.convert_categoricals_to_dummy(dataset.X)
        column_names = X_dummy.columns.tolist()
        for i, column_name in enumerate(column_names):
            feature_name = column_name
            feature_description = dataset.feature_descriptions[column_name]
            feature_prompts.append(
                feature_template.format(index=i + 1, feature_name=feature_name, feature_description=feature_description)
            )
        return prompt + "\n".join(feature_prompts), column_names

    @staticmethod
    def _feature_value_cannonizer(value: Any) -> Any:
        try:
            # round to 3 digits
            value = round(value, 3)
        except:
            pass
        return value

    @staticmethod
    def class_importance_prompts(dataset: LAATDataset) -> list[str]:
        labels_to_query = dataset.feature_descriptions["#DatasetClasses#"]
        # if it's binary classification, take the last class
        if len(labels_to_query) == 2:
            labels_to_query = labels_to_query[-1:]
        label_prompts = []
        for label in labels_to_query:
            label_prompts.append(
                FEATURE_IMPORTANCE_PROMPT.format(
                    dataset_prompt=LAATUtilities.dataset_to_prompt(dataset=dataset)[0], label=label
                )
            )
        return label_prompts


class LAATFewshotUtilities(LAATUtilities):
    @staticmethod
    def _feature_value_cannonizer(value: Any) -> Any:
        try:
            # round to 3 digits
            value = round(value, 3)
        except:
            pass
        return value

    @staticmethod
    def examples_to_prompt(features: pd.DataFrame, labels: pd.DataFrame, column_names: list[str]) -> str:
        """Encode a dataframe of examples into a prompt for few-shot.

        Args:
            examples (pd.DataFrame): Examples to encode.
            column_names (list[str]): Column names to use for encoding examples.

        Returns:
            str: Encoded examples.
        """
        feature_template = "{feature_name} is {feature_value}"
        label_template = "Ground truth is {label}"
        feature_label_prompts = []
        for (_, row), label in zip(features.iterrows(), labels.iloc[:, 0]):
            feature_prompt = " ".join(
                [
                    feature_template.format(
                        feature_name=column_name,
                        feature_value=LAATFewshotUtilities._feature_value_cannonizer(row[column_name]),
                    )
                    for column_name in column_names
                ]
            )
            label_prompt = label_template.format(label=label)
            feature_label_prompts.append(f"{feature_prompt}, {label_prompt}")
        # example shuffling
        random.shuffle(feature_label_prompts)
        prompt = "\n".join(feature_label_prompts)
        return prompt

    @staticmethod
    def _feature_value_cannonizer(value: Any) -> Any:
        try:
            # round to 3 digits
            value = round(value, 3)
        except:
            pass
        return value

    @staticmethod
    def class_importance_prompts(dataset: LAATDataset, features: pd.DataFrame, labels: pd.DataFrame) -> list[str]:
        labels_to_query = dataset.feature_descriptions["#DatasetClasses#"]
        if len(labels_to_query) == 2:
            labels_to_query = labels_to_query[-1:]

        label_prompts = []
        for label in labels_to_query:
            dataset_prompt, column_names = LAATFewshotUtilities.dataset_to_prompt(dataset=dataset)
            examples_prompt = LAATFewshotUtilities.examples_to_prompt(
                features=features, labels=labels, column_names=dataset.X.columns.tolist()
            )
            label_prompts.append(
                FEATURE_IMPORTANCE_FEWSHOT_PROMPT.format(
                    dataset_prompt=dataset_prompt,
                    examples_prompt=examples_prompt,
                    label=label,
                )
            )
        return label_prompts


class LAATLAATModel(LAATModel):
    def __init__(
        self,
        model_name: str,
        model_class: Type[BaseEstimator],
        pandas_to_numpy_mapper: Callable,
        dataset: LAATDataset,
        reasoning_llm: BaseChatModel,
        parsing_llm: BaseChatModel,
        gamma: float = 1.0,  # strength of LLM regularization
        n_estimates: int = 1,  # how many times to query the LLM and mean the resulting importance scores
        save_path: str = "./laat_importances",  # path for saving importance scores strings so that they don't have to be regenerated every time
        num_retries: int = 3,  # number of retries to generate importance scores before error
        scaler_class: Type[BaseEstimator] = MinMaxScaler,
    ):
        super().__init__(
            model_name=model_name,
            model_class=model_class,
            pandas_to_numpy_mapper=pandas_to_numpy_mapper,
            scaler_class=scaler_class,
        )
        self.model_name = model_name
        self.model_class = model_class
        self._pandas_to_numpy_mapper = pandas_to_numpy_mapper
        self.dataset = dataset
        self.reasoning_llm = reasoning_llm
        self.parsing_llm = parsing_llm
        self.gamma = gamma
        self.n_estimates = n_estimates
        self.save_path = save_path
        self.num_retries = num_retries

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        train_run_info: TrainRunInfo,
        X_validation: Optional[pd.DataFrame] = None,
        y_validation: Optional[pd.DataFrame] = None,
    ) -> None:

        # load pre-existing importance scores if they exist
        self.importance_scores = self._load(train_run_info=train_run_info)
        # if they do not exist, generate them
        if self.importance_scores is None:
            # give num_retries number of tries to generate
            for retry_index in range(self.num_retries):
                try:
                    self.importance_scores = self._generate_importance_scores()
                    # break and save if successfully generated importance scores
                    break
                except Exception as e:
                    # if last retry throw error
                    if retry_index == self.num_retries - 1:
                        raise ValueError("Importance scores could not be generated in the specified amount of retries")
            # save generated importance scores
            self._save(train_run_info=train_run_info)
        # initialize the model and fit to data
        self.model = self._init_model(train_run_info)
        X_train, y_train = self._pandas_to_numpy_mapper(X_train, y_train)
        # preprocess data into torch.tensors
        X_train = self._preprocess(X_train, train=True)
        if self.importance_scores.shape[0] > 1:
            y_train = torch.from_numpy(y_train).long()
        else:
            y_train = torch.from_numpy(y_train).float()
        # fit the model
        self.model.fit(X_train, y_train)

    def _init_model(self, train_run_info: TrainRunInfo) -> BaseEstimator:
        return self.model_class(importance_scores=self.importance_scores, gamma=self.gamma)

    def _preprocess(self, X: np.array, train: bool) -> torch.tensor:
        X = super()._preprocess(X=X, train=train)
        X = torch.from_numpy(X).float()
        return X

    def _generate_importance_scores(self) -> np.array:
        # create a LAAT pipeline
        all_importance_scores = []
        label_prompts = LAATUtilities.class_importance_prompts(dataset=self.dataset)
        laat_pipeline = RunnableParallel(
            {
                f"{i}": (
                    PromptTemplate.from_template(label_prompt)
                    | self.reasoning_llm
                    | StrOutputParser()
                    | dummy_print
                    | FEATURE_IMPORTANCE_FORMATTING_PROMPT
                    | self.parsing_llm.with_structured_output(ImportanceScores)
                )
                for i, label_prompt in enumerate(label_prompts)
            }
        )
        # infer LLM n_estimates times and mean the resulting importance scores
        for _ in tqdm(range(self.n_estimates)):
            importance_scores = laat_pipeline.invoke({})
            importance_scores = [
                [feature_score.importance_score for feature_score in importance_scores[f"{i}"].importance_scores]
                for i in range(len(label_prompts))
            ]
            print(importance_scores)
            all_importance_scores.append(importance_scores)
        all_importance_scores = np.array(all_importance_scores).mean(0)
        return all_importance_scores

    def _save(self, train_run_info: TrainRunInfo) -> None:
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(self.save_path, f"{self}.pickle"), "wb") as f:
            pickle.dump(self.importance_scores, f)

    def _load(self, train_run_info: TrainRunInfo) -> Optional[list[list[int]]]:
        importance_scores = None
        try:
            with open(os.path.join(self.save_path, f"{self}.pickle"), "rb") as f:
                importance_scores = pickle.load(f)
        except:
            pass
        return importance_scores

    def __str__(self) -> str:
        if isinstance(self.reasoning_llm, ChatGoogleGenerativeAI):
            model_name = self.reasoning_llm.model.split("/")[-1]
        else:
            model_name = self.reasoning_llm.model_name
        return f"{self.dataset.dataset_name}_{model_name}_{self.n_estimates}"

    def clear(self) -> None:
        """Delete model and free up memory"""
        del self.model
        del self.importance_scores
        torch.cuda.empty_cache()


class LAATEnsembleLAATModel(LAATModel):
    def __init__(
        self,
        model_name: str,
        model_class: Type[BaseEstimator],
        pandas_to_numpy_mapper: Callable,
        dataset: LAATDataset,
        reasoning_llm: BaseChatModel,
        parsing_llm: BaseChatModel,
        gamma: float = 1.0,  # strength of LLM regularization
        n_estimates: int = 1,  # how many times to query the LLM and mean the resulting importance scores
        save_path: str = "./ensemblelaat_importances",  # path for saving importance scores strings so that they don't have to be regenerated every time
        num_retries: int = 3,  # number of retries to generate importance scores before error
        scaler_class: Type[BaseEstimator] = MinMaxScaler,
    ):
        super().__init__(
            model_name=model_name,
            model_class=model_class,
            pandas_to_numpy_mapper=pandas_to_numpy_mapper,
            scaler_class=scaler_class,
        )
        self.model_name = model_name
        self.model_class = model_class
        self._pandas_to_numpy_mapper = pandas_to_numpy_mapper
        self.dataset = dataset
        self.reasoning_llm = reasoning_llm
        self.parsing_llm = parsing_llm
        self.gamma = gamma
        self.n_estimates = n_estimates
        self.save_path = save_path
        self.num_retries = num_retries

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        train_run_info: TrainRunInfo,
        X_validation: Optional[pd.DataFrame] = None,
        y_validation: Optional[pd.DataFrame] = None,
    ) -> None:

        # load pre-existing importance scores if they exist
        self.importance_scores = self._load(train_run_info=train_run_info)
        # if they do not exist, generate them
        if self.importance_scores is None:
            # give num_retries number of tries to generate
            for retry_index in range(self.num_retries):
                try:
                    self.importance_scores = self._generate_importance_scores()
                    # break and save if successfully generated importance scores
                    break
                except Exception as e:
                    # if last retry throw error
                    if retry_index == self.num_retries - 1:
                        raise ValueError("Importance scores could not be generated in the specified amount of retries")
            # save generated importance scores
            self._save(train_run_info=train_run_info)
        # initialize the model and fit to data
        self.models = [
            self._init_model(self.importance_scores[importance_i], train_run_info)
            for importance_i in range(self.n_estimates)
        ]
        X_train, y_train = self._pandas_to_numpy_mapper(X_train, y_train)
        # preprocess data into torch.tensors
        X_train = self._preprocess(X_train, train=True)
        if len(self.dataset.label_list) > 2:
            y_train = torch.from_numpy(y_train).long()
        else:
            y_train = torch.from_numpy(y_train).float()
        # fit the model
        for model in self.models:
            model.fit(X_train, y_train)

    def _init_model(self, importance_scores: np.array, train_run_info: TrainRunInfo) -> BaseEstimator:
        return self.model_class(importance_scores=importance_scores, gamma=self.gamma)

    def _preprocess(self, X: np.array, train: bool) -> torch.tensor:
        X = super()._preprocess(X=X, train=train)
        X = torch.from_numpy(X).float()
        return X

    def predict(self, X: pd.DataFrame) -> np.array:
        raise NotImplementedError("predict for laat ensemble not implemented")

    def predict_proba(self, X: pd.DataFrame) -> np.array:
        X = self._pandas_to_numpy_mapper(X)
        X = self._preprocess(X, train=False)
        predict_probas = []
        for model in self.models:
            predict_probas.append(model.predict_proba(X))
        predict_probas = np.stack(predict_probas, axis=0).mean(0)
        return predict_probas

    def _generate_importance_scores(self) -> np.array:
        # create a LAAT pipeline
        all_importance_scores = []
        label_prompts = LAATUtilities.class_importance_prompts(dataset=self.dataset)
        laat_pipeline = RunnableParallel(
            {
                f"{i}": (
                    PromptTemplate.from_template(label_prompt)
                    | self.reasoning_llm
                    | StrOutputParser()
                    | dummy_print
                    | FEATURE_IMPORTANCE_FORMATTING_PROMPT
                    | self.parsing_llm.with_structured_output(ImportanceScores)
                )
                for i, label_prompt in enumerate(label_prompts)
            }
        )
        # infer LLM n_estimates times and mean the resulting importance scores
        for _ in range(self.n_estimates):
            importance_scores = laat_pipeline.invoke({})
            importance_scores = [
                [feature_score.importance_score for feature_score in importance_scores[f"{i}"].importance_scores]
                for i in range(len(label_prompts))
            ]
            print(importance_scores)
            all_importance_scores.append(importance_scores)
        all_importance_scores = np.array(all_importance_scores)
        return all_importance_scores

    def _save(self, train_run_info: TrainRunInfo) -> None:
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(self.save_path, f"{self}.pickle"), "wb") as f:
            pickle.dump(self.importance_scores, f)

    def _load(self, train_run_info: TrainRunInfo) -> Optional[list[list[int]]]:
        importance_scores = None
        try:
            with open(os.path.join(self.save_path, f"{self}.pickle"), "rb") as f:
                importance_scores = pickle.load(f)
        except:
            pass
        return importance_scores

    def __str__(self) -> str:
        return f"{self.dataset.dataset_name}_{self.reasoning_llm.model_name}_{self.n_estimates}"

    def clear(self) -> None:
        """Delete model and free up memory"""
        del self.models
        del self.importance_scores
        torch.cuda.empty_cache()


class LAATFewshotLAATModel(LAATLAATModel):
    def __init__(
        self,
        model_name: str,
        model_class: Type[BaseEstimator],
        pandas_to_numpy_mapper: Callable,
        dataset: LAATDataset,
        reasoning_llm: BaseChatModel,
        parsing_llm: BaseChatModel,
        gamma: float = 1.0,  # strength of LLM regularization
        n_estimates: int = 1,  # how many times to query the LLM and mean the resulting importance scores
        save_path: str = "./laat_fewshot_importances",  # path for saving importance scores strings so that they don't have to be regenerated every time
        scaler_class: Type[BaseEstimator] = MinMaxScaler,
    ):
        super().__init__(
            model_name=model_name,
            model_class=model_class,
            pandas_to_numpy_mapper=pandas_to_numpy_mapper,
            scaler_class=scaler_class,
        )
        self.model_name = model_name
        self.model_class = model_class
        self._pandas_to_numpy_mapper = pandas_to_numpy_mapper
        self.dataset = dataset
        self.reasoning_llm = reasoning_llm
        self.parsing_llm = parsing_llm
        self.gamma = gamma
        self.n_estimates = n_estimates
        self.save_path = save_path

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        train_run_info: TrainRunInfo,
        X_validation: Optional[pd.DataFrame] = None,
        y_validation: Optional[pd.DataFrame] = None,
    ) -> None:

        # load pre-existing importance scores if they exist
        self.importance_scores = self._load(train_run_info=train_run_info)
        # if they do not exist, generate them
        if self.importance_scores is None:
            self.importance_scores = self._generate_importance_scores(X=X_train, y=y_train)
            self._save(train_run_info=train_run_info)
        # initialize the model and fit to data
        self.model = self._init_model(train_run_info)
        X_train, y_train = self._pandas_to_numpy_mapper(X_train, y_train)
        # preprocess data into torch.tensors
        X_train = self._preprocess(X_train, train=True)
        if self.importance_scores.shape[0] > 1:
            y_train = torch.from_numpy(y_train).long()
        else:
            y_train = torch.from_numpy(y_train).float()
        # fit the model
        self.model.fit(X_train, y_train)

    def _generate_importance_scores(self, X: pd.DataFrame, y: pd.DataFrame) -> np.array:
        # create a LAAT pipeline
        all_importance_scores = []
        label_prompts = LAATFewshotUtilities.class_importance_prompts(dataset=self.dataset, features=X, labels=y)
        laat_pipeline = RunnableParallel(
            {
                f"{i}": (
                    PromptTemplate.from_template(label_prompt)
                    | self.reasoning_llm
                    | StrOutputParser()
                    | dummy_print
                    | FEATURE_IMPORTANCE_FORMATTING_PROMPT
                    | self.parsing_llm.with_structured_output(ImportanceScores)
                )
                for i, label_prompt in enumerate(label_prompts)
            }
        )
        # infer LLM n_estimates times and mean the resulting importance scores
        for _ in range(self.n_estimates):
            importance_scores = laat_pipeline.invoke({})
            importance_scores = [
                [feature_score.importance_score for feature_score in importance_scores[f"{i}"].importance_scores]
                for i in range(len(label_prompts))
            ]
            print(importance_scores)
            all_importance_scores.append(importance_scores)
        all_importance_scores = np.array(all_importance_scores).mean(0)
        return all_importance_scores

    def _save(self, train_run_info: TrainRunInfo) -> None:
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(self.save_path, f"{self.dataset.dataset_name}_{train_run_info}.pickle"), "wb") as f:
            pickle.dump(self.importance_scores, f)

    def _load(self, train_run_info: TrainRunInfo) -> Optional[list[list[int]]]:
        importance_scores = None
        try:
            with open(os.path.join(self.save_path, f"{self.dataset.dataset_name}_{train_run_info}.pickle"), "rb") as f:
                importance_scores = pickle.load(f)
        except:
            pass
        return importance_scores


def dummy_print(t):
    print(t)
    return t
