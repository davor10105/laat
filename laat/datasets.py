import numpy as np
import pandas as pd
from typing import Type, TypeVar, Optional, Union
import os
import json
from laat.prompts.laat import METADATA_GENERATION_PROMPT
from pydantic import BaseModel, Field
from langchain.chat_models.base import BaseChatModel

T = TypeVar("T", bound="LAATDataset")


class SingleFeatureDescription(BaseModel):
    name: str = Field(description="name of the feature")
    description: str = Field(description="a brief description of the feature")


class FeatureDescriptions(BaseModel):
    feature_descriptions: list[SingleFeatureDescription] = Field(description="a list of all feature descriptions")


class LAATDataset:
    def __init__(self, dataset_name: str, data_root: str) -> None:
        self.dataset_name = dataset_name
        self._load_dataset(dataset_name, data_root)
        # set feature and label count for convenience
        np_X = self.to_numpy(self.X)
        self.n_features = np_X.shape[1]
        n_classes = len(self.feature_descriptions["#DatasetClasses#"])
        if n_classes == 2:
            n_classes -= 1
        self.n_classes = n_classes
        # set other attributes for featllm convenience
        self.file_name = os.path.join(data_root, f"{dataset_name}.csv")
        self.meta_file_name = os.path.join(data_root, f"{dataset_name}-metadata.json")

    def _load_dataset(self, dataset_name: str, data_root: str) -> None:
        # load features from the CSV
        df = pd.read_csv(os.path.join(data_root, f"{dataset_name}.csv"))
        # featllm *sigh*
        self.df = df
        self.target_attr = df.columns[-1]
        self.categorical_indicator = [
            True if (dt == np.dtype("O") or pd.api.types.is_string_dtype(dt)) else False for dt in df.dtypes.tolist()
        ][:-1]
        # convert categorical variables to pd.Categorical
        for column_name in df.columns:
            if df[column_name].dtype == "object":
                df[column_name] = pd.Categorical(df[column_name])
        # split into data and labels
        X, y = df.iloc[:, :-1], df.iloc[:, -1:]

        # set categoricals
        categorical_columns = [col for col in X.columns if X[col].dtype == "category"]
        self.X, self.y, self.categorical_columns = X, y, categorical_columns

        # load descriptions for the JSON
        with open(os.path.join(data_root, f"{dataset_name}-metadata.json"), "r") as f:
            self.feature_descriptions = json.load(f)
            assert (
                "#DatasetTask#" in self.feature_descriptions and "#DatasetClasses#" in self.feature_descriptions
            ), "You must include #DatasetTask# and #DatasetClasses# into dataset metadata."

        # save labels for convenience
        self.label_list = self.feature_descriptions["#DatasetClasses#"]  # y.iloc[:, -1].unique().tolist()

    def to_numpy(
        self, X: Optional[pd.DataFrame] = None, y: Optional[pd.DataFrame] = None
    ) -> Union[np.array, tuple[np.array, np.array]]:
        assert X is not None or y is not None, "At least one of (X, y) must be not None"
        retvals = []
        if X is not None:
            # encode categoricals as flags and convert to numpy
            X = pd.get_dummies(X)
            X = X.to_numpy().astype(np.float32)
            retvals.append(X)
        if y is not None:
            y = (
                y.iloc[:, 0]
                .map({label: i for i, label in enumerate(self.feature_descriptions["#DatasetClasses#"])})
                .to_numpy()
            )
            retvals.append(y)
        if len(retvals) == 1:
            return retvals[0]
        return tuple(retvals)

    @classmethod
    def load(cls: Type[T], dataset_name: str, data_root: str = "./data") -> T:
        return cls(dataset_name=dataset_name, data_root=data_root)

    @staticmethod
    def generate_metadata(
        dataset_name: str, dataset_task_description: str, model: BaseChatModel, data_root: str = "./data"
    ) -> None:
        """Generate the metadata file for a dataset.

        Args:
            dataset_name (str): Name of the .csv file in the data_root directory.
            dataset_description (str): Description of the dataset.
            data_root (str, optional): Root folder for the datasets. Defaults to "./data".
        """
        df = pd.read_csv(os.path.join(data_root, f"{dataset_name}.csv"))
        target_attr = df.columns[-1]
        categorical_indicator = [
            True if (dt == np.dtype("O") or pd.api.types.is_string_dtype(dt)) else False for dt in df.dtypes.tolist()
        ][:-1]
        # convert categorical variables to pd.Categorical
        for column_name in df.columns:
            if df[column_name].dtype == "object":
                df[column_name] = pd.Categorical(df[column_name])
        # split into data and labels
        X, y = df.iloc[:, :-1], df.iloc[:, -1:]

        # set categoricals
        categorical_columns = [col for col in X.columns if X[col].dtype == "category"]
        assert (
            len(list(filter(lambda name: len(name.split("_")) > 1, X.columns.tolist()))) == 0
        ), "No dataset features can contain '_' as a part of their name"
        # encode categorical features as binary flags
        X_dummy = pd.get_dummies(X)
        column_names = X_dummy.columns.tolist()
        # combine original and dummy features
        all_features_names = list(set(X.columns.tolist()) | set(column_names))
        # prepare a list for all features (might need more than one iteration if any of the features are missed)
        all_feature_descriptions = []
        # give 5 tries to generate all descriptions
        metadata_chain = METADATA_GENERATION_PROMPT | model.with_structured_output(FeatureDescriptions)
        for _ in range(5):
            # get currently needed feature names
            needed_feature_names = set(all_features_names) - set([feature.name for feature in all_feature_descriptions])
            # generate the descriptions
            feature_descriptions = metadata_chain.invoke(
                {
                    "dataset_description": dataset_task_description,
                    "features": needed_feature_names,
                }
            ).feature_descriptions
            all_feature_descriptions += feature_descriptions
            # if all generated, break
            current_feature_names = set([feature.name for feature in all_feature_descriptions])
            if set(all_features_names) == current_feature_names:
                break
        # sort the classes
        dataset_classes = sorted(y[y.columns[0]].unique().tolist())
        # construct the metadata dict
        metadata_dict = {
            "#DatasetTask#": dataset_task_description,
            "#DatasetClasses#": dataset_classes,
        }
        for feature_description in sorted(all_feature_descriptions, key=lambda k: k.name):
            metadata_dict[feature_description.name] = feature_description.description
        # save the metadata
        with open(os.path.join(data_root, f"{dataset_name}-metadata.json"), "w") as f:
            json.dump(metadata_dict, f, indent=2)
