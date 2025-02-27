import torch
from torch import nn
import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator
from typing import Optional, Callable, Type, Any
from abc import ABC
from laat.models.base import LAATModel, TrainRunInfo
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.prompts import PromptTemplate
from laat.prompts.featllm import CONDITION_EXTRACTION_PROMPT, CONDITION_FORMATTING_PROMPT, FUNCTION_GENERATION_PROMPT
from laat.datasets import LAATDataset
import random
from pathlib import Path
import os
import pickle


class Conditions(BaseModel):
    conditions: list[str] = Field(
        description="all conditions related to a class found in the text, in order of their appearance"
    )


class FeatLLMUtilities:
    @staticmethod
    def dataset_to_prompt(dataset: LAATDataset, column_names: Optional[list[str]] = None) -> tuple[str, str]:
        """Encode a dataset as a FeatLLM prompt.

        Args:
            dataset (LAATDataset): Dataset to encode.
            column_names (Optional[list[str]]): column_names to use (in case of feature bagging). Defaults to None.

        Returns:
            tuple[str, str]: Dataset converted to prompt and used features (in feature bagging scenarios)
        """
        prompt = f"Task: {dataset.feature_descriptions['#DatasetTask#']}\nFeatures:\n"
        feature_template = "{index}. {feature_name}: {feature_description}"
        feature_prompts = []
        if column_names is None:
            column_names = dataset.X.columns.tolist()
        if len(column_names) >= 20:
            # feature bagging threshold
            random.shuffle(column_names)
            column_names = column_names[:10]  # featllm uses 10 features per bag
        for i, column_name in enumerate(column_names):
            feature_name = column_name
            feature_description = dataset.feature_descriptions[column_name]
            if column_name in dataset.categorical_columns:
                categories = dataset.X[column_name].cat.categories.tolist()
                feature_description += f" (categorical variable with the following categories: {', '.join(categories)})"
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
                        feature_value=FeatLLMUtilities._feature_value_cannonizer(row[column_name]),
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
    def condition_extraction_prompt(
        dataset: LAATDataset, features: pd.DataFrame, labels: pd.DataFrame
    ) -> tuple[str, str]:
        dataset_prompt, column_names = FeatLLMUtilities.dataset_to_prompt(dataset=dataset)
        examples_prompt = FeatLLMUtilities.examples_to_prompt(
            features=features, labels=labels, column_names=column_names
        )
        condition_prompt = """10 different conditions for class "{label}":
- [Condition]
..."""
        conditions_prompt = "\n\n".join(
            [condition_prompt.format(label=label) for label in dataset.y.iloc[:, 0].cat.categories.tolist()]
        )

        return (
            CONDITION_EXTRACTION_PROMPT.format(
                dataset_prompt=dataset_prompt, examples_prompt=examples_prompt, conditions_prompt=conditions_prompt
            ),
            column_names,
        )

    @staticmethod
    def sanitize_output(text: str):
        _, after = text.split("```python")
        return after.split("```")[0]


class FeatLLMLAATModel(LAATModel):
    def __init__(
        self,
        model_name: str,
        model_class: Type[BaseEstimator],
        pandas_to_numpy_mapper: Callable,
        dataset: LAATDataset,
        reasoning_llm: BaseChatModel,
        parsing_llm: BaseChatModel,
        n_estimators: int = 20,  # number of repetitions of LLM feature engineering and small model training
        save_path: str = "./featllm_functions",  # path for saving function strings so that they don't have to be regenerated every time
        num_retries: int = 3,  # number of retries to generate preprocessing function before error
    ):
        self.model_name = model_name
        self.model_class = model_class
        self._pandas_to_numpy_mapper = pandas_to_numpy_mapper
        self.dataset = dataset
        self.reasoning_llm = reasoning_llm
        self.parsing_llm = parsing_llm
        self.n_estimators = n_estimators
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
        # feature engineering via llm
        self.models = [self._init_model(train_run_info) for _ in range(self.n_estimators)]
        # load pre-existing functions if they exist
        self.featllm_preprocess_functions, loaded_functions = self._load(train_run_info=train_run_info)
        # if they do not exist, generate them
        if not loaded_functions:
            # give num_retries number of tries to generate
            for retry_index in range(self.num_retries):
                try:
                    self.featllm_preprocess_functions = self._generate_functions(X_train=X_train, y_train=y_train)
                    # break and save if successfully generated functions
                    break
                except Exception as e:
                    # if last retry throw error
                    if retry_index == self.num_retries - 1:
                        raise ValueError("Functions could not be generated in the specified amount of retries")
            # save generated functions
            self._save(train_run_info=train_run_info)
        # utilize the functions for preprocessing and fit the models
        for i in range(self.n_estimators):
            preprocess_function = self.featllm_preprocess_functions[i]
            exec(preprocess_function)
            changed_X = locals().get("add_new_features")(X_train.copy())
            pre_X_train, pre_y_train = self._pandas_to_numpy_mapper(changed_X, y_train)
            self.models[i].fit(pre_X_train, pre_y_train)

    def _generate_functions(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> list[str]:
        featllm_preprocess_functions = []
        encoded_dataset, column_names = FeatLLMUtilities.condition_extraction_prompt(
            dataset=self.dataset, features=X_train, labels=y_train
        )
        # batch LLM invokation over n_estimators
        runnable_dict = {}
        for i in range(self.n_estimators):
            runnable_dict[f"{i}"] = (
                PromptTemplate.from_template(encoded_dataset)
                | self.reasoning_llm
                | StrOutputParser()
                | CONDITION_FORMATTING_PROMPT
                | self.parsing_llm.with_structured_output(Conditions)
                | RunnableLambda(
                    lambda conditions: FUNCTION_GENERATION_PROMPT.format(
                        features_prompt=FeatLLMUtilities.dataset_to_prompt(
                            dataset=self.dataset, column_names=column_names
                        )[0],
                        conditions_prompt="\n".join(
                            [f"{i+1}. {condition}" for i, condition in enumerate(conditions.conditions)]
                        ),
                    )
                )
                | self.reasoning_llm
                | StrOutputParser()
                | FeatLLMUtilities.sanitize_output
            )
        runnable_batch = RunnableParallel(runnable_dict)
        runnable_output = runnable_batch.invoke({})
        featllm_preprocess_functions = [runnable_output[f"{i}"] for i in range(self.n_estimators)]
        # test out the functions to make sure they work
        for preprocess_function in featllm_preprocess_functions:
            exec(preprocess_function)
            locals().get("add_new_features")(X_train.copy())

        return featllm_preprocess_functions

    def _save(self, train_run_info: TrainRunInfo) -> None:
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(self.save_path, f"{self}_{train_run_info}.pickle"), "wb") as f:
            pickle.dump(self.featllm_preprocess_functions, f)

    def _load(self, train_run_info: TrainRunInfo) -> tuple[list[str], bool]:
        featllm_preprocess_functions = []
        try:
            with open(os.path.join(self.save_path, f"{self}_{train_run_info}.pickle"), "rb") as f:
                featllm_preprocess_functions = pickle.load(f)
        except:
            pass
        return featllm_preprocess_functions, len(featllm_preprocess_functions) == self.n_estimators

    def __str__(self) -> str:
        return f"{self.dataset.dataset_name}_{self.reasoning_llm.model_name}_{self.n_estimators}"

    def predict(self, X: pd.DataFrame) -> np.array:
        pred_probas = []
        for i in range(self.n_estimators):
            exec(self.featllm_preprocess_functions[i])
            changed_X = locals().get("add_new_features")(X.copy())
            pre_X_train = self._pandas_to_numpy_mapper(changed_X)
            pred_probas.append(self.models[i].predict_proba(pre_X_train))
        pred_probas = np.stack(pred_probas, axis=0).mean(0)
        return pred_probas.argmax(axis=-1)

    def predict_proba(self, X: pd.DataFrame) -> np.array:
        pred_probas = []
        for i in range(self.n_estimators):
            exec(self.featllm_preprocess_functions[i])
            changed_X = locals().get("add_new_features")(X.copy())
            pre_X_train = self._pandas_to_numpy_mapper(changed_X)
            pred_probas.append(self.models[i].predict_proba(pre_X_train))
        return np.stack(pred_probas, axis=0).mean(0)

    def clear(self) -> None:
        """Delete model and free up memory"""
        del self.models
        del self.featllm_preprocess_functions
        torch.cuda.empty_cache()
