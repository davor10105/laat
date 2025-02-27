import numpy as np
import pandas as pd
from typing import Type, TypeVar, Optional, Union
import os
import json

T = TypeVar("T", bound="LAATDataset")


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


# import torch
# from ucimlrepo import fetch_ucirepo
# from openml.datasets import get_dataset
# from sklearn.datasets import fetch_california_housing
# import pandas as pd
# import numpy as np
# from pydantic import BaseModel, Field

# from typing import Type, TypeVar
# from abc import ABC

# T = TypeVar("T", bound="LAATDataset")

# LLM_PROMPT = """You are an expert at assigning importance scores to features used for a classification task. For each feature, output an integer importance score between -10 and 10. Positive scores indicate a positive causal effect between the feature and the class probability, while negative scores indicate a negative causal effect between the feature and the class probability.
# Task: Patient diabetes prediction
# Class: The patient has diabetes
# Features:"""


# def get_feat_llmdataset(data_name, shot, seed):
#     file_name = f"~/research/llm-xai-guidance/FeatLLM/data/{data_name}.csv"
#     df = pd.read_csv(file_name)
#     default_target_attribute = df.columns[-1]

#     categorical_indicator = [
#         True if (dt == np.dtype("O") or pd.api.types.is_string_dtype(dt)) else False for dt in df.dtypes.tolist()
#     ][:-1]
#     attribute_names = df.columns[:-1].tolist()

#     X = df.convert_dtypes()
#     X = X.drop(columns=[default_target_attribute])
#     y = df[default_target_attribute].to_numpy()
#     label_list = np.unique(y).tolist()
#     return X, y, default_target_attribute, label_list, categorical_indicator


# class LAATDataset(ABC):
#     @classmethod
#     def load(cls: Type[T], *args, **kwargs) -> T:
#         return cls(*args, **kwargs)


# class HeartDiseaseDataset(LAATDataset):
#     """NOT GREAT, GAMMA 1 SLIGHT IMPROVEMENT, BUT ICKY
#         model_kwargs = {
#         # "module": PModel,
#         "llm_ratings": dataset.llm_ratings,
#         "lr": 0.1,
#         "max_epochs": nsteps,
#         # "train_split": None,
#         "optimizer": torch.optim.SGD,
#         "optimizer__momentum": 0.9,
#         "optimizer__weight_decay": 1,
#         "verbose": False,
#         "device": device,
#     }"""

#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__()

#         X, y, default_target_attribute, label_list, categorical_indicator = get_feat_llmdataset("heart", 2, 69)
#         for i, label in enumerate(label_list):
#             y[y == label] = i
#         X = pd.get_dummies(X)

#         self.X = torch.from_numpy(X.astype(float).to_numpy()).float()
#         self.y = torch.from_numpy(y.astype(float)).flatten().float()

#         self.llm_ratings = torch.tensor([[7.0, 5, 4, 6, -6, 8, -2, 3, 9, -4, -2, -3, 5, -4, 3, -6, 7, 8, 6, -5]])


# class DiabetesDataset(LAATDataset):
#     def __init__(self, *args, **kwargs) -> None:
#         """WORKS, DO NOT TOUCH
#         MinMaxScaler
#         nsteps = 100
#         model_kwargs = {
#             # "module": PModel,
#             "llm_ratings": dataset.llm_ratings,
#             "lr": 0.1,
#             "max_epochs": nsteps,
#             # "train_split": None,
#             "optimizer": torch.optim.SGD,
#             "optimizer__momentum": 0.9,
#             "optimizer__weight_decay": 1e-4,
#             "verbose": False,
#             "device": device,
#         }
#         """
#         super().__init__()

#         X, y, default_target_attribute, label_list, categorical_indicator = get_feat_llmdataset("diabetes", 2, 69)
#         for i, label in enumerate(label_list):
#             y[y == label] = i
#         # X = pd.get_dummies(X)

#         self.X = torch.from_numpy(X.astype(float).to_numpy()).float()
#         self.y = torch.from_numpy(y.astype(float)).flatten().float()

#         self.llm_ratings = torch.tensor([[3.0, 10, 2, 4, 7, 8, 6, 5]])


# class BankMarketingDataset(LAATDataset):
#     """WORKS!!! NE DIRAJ!!!
#         model_kwargs = {
#         # "module": PModel,
#         "llm_ratings": dataset.llm_ratings,
#         "lr": 0.1,
#         "max_epochs": nsteps,
#         # "train_split": None,
#         "optimizer": torch.optim.SGD,
#         "optimizer__momentum": 0.9,
#         "optimizer__weight_decay": 1e-4,
#         "verbose": False,
#         "device": device,
#     }"""

#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__()

#         X, y, default_target_attribute, label_list, categorical_indicator = get_feat_llmdataset("bank", 2, 69)
#         for i, label in enumerate(label_list):
#             y[y == label] = i
#         X = pd.get_dummies(X)

#         self.X = torch.from_numpy(X.astype(float).to_numpy()).float()
#         self.y = torch.from_numpy(y.astype(float)).flatten().float()

#         self.llm_ratings = torch.tensor(
#             [
#                 [
#                     2.0,
#                     6,
#                     0,
#                     10,
#                     -4,
#                     -3,
#                     5,
#                     1,
#                     -3,
#                     -2,
#                     -1,
#                     2,
#                     6,
#                     -1,
#                     -2,
#                     7,
#                     1,
#                     5,
#                     0,
#                     1,
#                     -2,
#                     3,
#                     -3,
#                     -1,
#                     4,
#                     0,
#                     3,
#                     -6,
#                     2,
#                     -2,
#                     3,
#                     -3,
#                     7,
#                     -2,
#                     -4,
#                     -2,
#                     1,
#                     2,
#                     5,
#                     -3,
#                     -2,
#                     3,
#                     6,
#                     -5,
#                     -4,
#                     1,
#                     2,
#                     -5,
#                     1,
#                     9,
#                     -3,
#                 ]
#             ]
#         )


# class CaliforniaHousingDataset(LAATDataset):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__()

#         dataset = fetch_california_housing()
#         X, y = dataset.data, dataset.target

#         self.X = torch.from_numpy(X).float()
#         self.y = torch.from_numpy(y).flatten().float()[:, None]

#         self.feature_metadata = [
#             {
#                 "name": "MedInc",
#                 "description": "median income in block group, scaled to 0 - 1 interval",
#                 "unit": "",
#                 "preprocessor": lambda k: int(k),
#             },
#             {
#                 "name": "HouseAge",
#                 "description": "median house age in block group, scaled to 0 - 1 interval",
#                 "unit": "",
#                 "preprocessor": lambda k: int(k),
#             },
#             {
#                 "name": "AveRooms",
#                 "description": "average number of rooms per household, scaled to 0 - 1 interval",
#                 "unit": "",
#                 "preprocessor": lambda k: int(k),
#             },
#             {
#                 "name": "AveBedrms",
#                 "description": "average number of bedrooms per household, scaled to 0 - 1 interval",
#                 "unit": "",
#                 "preprocessor": lambda k: int(k),
#             },
#             {
#                 "name": "Population",
#                 "description": "block group population, scaled to 0 - 1 interval",
#                 "unit": "",
#                 "preprocessor": lambda k: int(k),
#             },
#             {
#                 "name": "AveOccup",
#                 "description": "average number of household members, scaled to 0 - 1 interval",
#                 "unit": "",
#                 "preprocessor": lambda k: int(k),
#             },
#             {
#                 "name": "Latitude",
#                 "description": "block group latitude, scaled to 0 - 1 interval",
#                 "unit": "",
#                 "preprocessor": lambda k: int(k),
#             },
#             {
#                 "name": "Longitude",
#                 "description": "block group longitude, scaled to 0 - 1 interval",
#                 "unit": "",
#                 "preprocessor": lambda k: int(k),
#             },
#         ]

#         self.llm_ratings = torch.tensor([[10.0, -3, 7, 5, -4, -2, 1, 1]])


# class AdultDataset(LAATDataset):
#     def __init__(self, *args, **kwargs) -> None:
#         """WORKS SAME AS DIABETES
#         DO NOT TOUCH!!!"""
#         super().__init__()

#         adult = fetch_ucirepo(id=2)

#         # data (as pandas dataframes)
#         X = adult.data.features
#         y = adult.data.targets

#         # remove missing values
#         X = X.dropna()
#         y = y.loc[X.index]

#         X = pd.get_dummies(X, drop_first=True).astype(float)

#         y_filtered = []
#         for i, row in y.iterrows():
#             if "<=50K" in row.iloc[0]:
#                 y_filtered.append(0)
#             else:
#                 y_filtered.append(1)

#         self.X = torch.from_numpy(X.to_numpy()).float()
#         self.y = torch.tensor(y_filtered).flatten().float()

#         self.llm_ratings = torch.tensor(
#             [
#                 [
#                     6.0,  # age
#                     0,  # fnlwgt
#                     8,  # education-num
#                     10,  # capital-gain
#                     7,  # capital-loss
#                     7,  # hours-per-week
#                     5,  # workclass_Federal-gov
#                     3,  # workclass_Local-gov
#                     -10,  # workclass_Never-worked
#                     1,  # workclass_Private
#                     6,  # workclass_Self-emp-inc
#                     2,  # workclass_Self-emp-not-inc
#                     3,  # workclass_State-gov
#                     -10,  # workclass_Without-pay
#                     -5,  # education_11th
#                     -4,  # education_12th
#                     -9,  # education_1st-4th
#                     -8,  # education_5th-6th
#                     -7,  # education_7th-8th
#                     -6,  # education_9th
#                     3,  # education_Assoc-acdm
#                     3,  # education_Assoc-voc
#                     7,  # education_Bachelors
#                     9,  # education_Doctorate
#                     -1,  # education_HS-grad
#                     8,  # education_Masters
#                     -10,  # education_Preschool
#                     9,  # education_Prof-school
#                     2,  # education_Some-college
#                     6,  # marital-status_Married-AF-spouse
#                     8,  # marital-status_Married-civ-spouse
#                     -4,  # marital-status_Married-spouse-absent
#                     -5,  # marital-status_Never-married
#                     -3,  # marital-status_Separated
#                     -2,  # marital-status_Widowed
#                     1,  # occupation_Adm-clerical
#                     3,  # occupation_Armed-Forces
#                     4,  # occupation_Craft-repair
#                     8,  # occupation_Exec-managerial
#                     -5,  # occupation_Farming-fishing
#                     -6,  # occupation_Handlers-cleaners
#                     -2,  # occupation_Machine-op-inspct
#                     -7,  # occupation_Other-service
#                     -10,  # occupation_Priv-house-serv
#                     9,  # occupation_Prof-specialty
#                     6,  # occupation_Protective-serv
#                     3,  # occupation_Sales
#                     5,  # occupation_Tech-support
#                     2,  # occupation_Transport-moving
#                     -3,  # relationship_Not-in-family
#                     -5,  # relationship_Other-relative
#                     -10,  # relationship_Own-child
#                     -4,  # relationship_Unmarried
#                     7,  # relationship_Wife
#                     1,  # race_Asian-Pac-Islander
#                     -2,  # race_Black
#                     -3,  # race_Other
#                     2,  # race_White
#                     8,  # sex_Male
#                     -1,  # native-country_Cambodia
#                     2,  # native-country_Canada
#                     -1,  # native-country_China
#                     -3,  # native-country_Columbia
#                     -2,  # native-country_Cuba
#                     -3,  # native-country_Dominican-Republic
#                     -2,  # native-country_Ecuador
#                     -4,  # native-country_El-Salvador
#                     3,  # native-country_England
#                     3,  # native-country_France
#                     3,  # native-country_Germany
#                     -1,  # native-country_Greece
#                     -4,  # native-country_Guatemala
#                     -3,  # native-country_Haiti
#                     -5,  # native-country_Holand-Netherlands
#                     -4,  # native-country_Honduras
#                     -1,  # native-country_Hong
#                     2,  # native-country_Hungary
#                     3,  # native-country_India
#                     4,  # native-country_Iran
#                     1,  # native-country_Ireland
#                     3,  # native-country_Italy
#                     -2,  # native-country_Jamaica
#                     3,  # native-country_Japan
#                     -4,  # native-country_Laos
#                     -5,  # native-country_Mexico
#                     -4,  # native-country_Nicaragua
#                     -3,  # native-country_Outlying-US(Guam-USVI-etc)
#                     -3,  # native-country_Peru
#                     2,  # native-country_Philippines
#                     2,  # native-country_Poland
#                     -1,  # native-country_Portugal
#                     -3,  # native-country_Puerto-Rico
#                     -1,  # native-country_Scotland
#                     -2,  # native-country_South
#                     3,  # native-country_Taiwan
#                     -3,  # native-country_Thailand
#                     -3,  # native-country_Trinadad&Tobago
#                     1,  # native-country_United-States
#                     -2,  # native-country_Vietnam
#                     2,  # native-country_Yugoslavia
#                 ]
#             ]
#         )


# class PhishingDataset(LAATDataset):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__()

#         phishing_websites = fetch_ucirepo(id=327)

#         # data (as pandas dataframes)
#         X = phishing_websites.data.features
#         y = phishing_websites.data.targets

#         X = X.dropna()
#         y = y.loc[X.index]
#         y[y["result"] == -1] = 0

#         self.X = torch.from_numpy(X.to_numpy()).float()
#         self.y = torch.from_numpy(y.to_numpy()).flatten().float()

#         self.llm_ratings = torch.tensor(
#             [
#                 [
#                     10.0,
#                     6,
#                     9,
#                     10,
#                     7,
#                     6,
#                     7,
#                     -5,
#                     -6,
#                     -3,
#                     8,
#                     -4,
#                     8,
#                     9,
#                     5,
#                     9,
#                     10,
#                     10,
#                     7,
#                     9,
#                     6,
#                     5,
#                     8,
#                     -7,
#                     -8,
#                     -7,
#                     -6,
#                     -6,
#                     -6,
#                     -5,
#                 ]
#             ]
#         )


# class ThyroidCancerDataset(LAATDataset):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__()

#         differentiated_thyroid_cancer_recurrence = fetch_ucirepo(id=915)

#         # data (as pandas dataframes)
#         X = differentiated_thyroid_cancer_recurrence.data.features
#         y = differentiated_thyroid_cancer_recurrence.data.targets

#         X = X.dropna()
#         y = y.loc[X.index]

#         X = pd.get_dummies(X, drop_first=True)
#         y[y["Recurred"] == "Yes"] = 1
#         y[y["Recurred"] == "No"] = 0

#         self.X = torch.from_numpy(X.astype(float).to_numpy()).float()
#         self.y = torch.tensor(y.to_numpy().astype(float)).flatten().float()

#         self.llm_ratings = torch.tensor(
#             [
#                 [
#                     5.0,
#                     2,
#                     3,
#                     2,
#                     4,
#                     -1,
#                     0,
#                     3,
#                     -2,
#                     1,
#                     -3,
#                     2,
#                     2,
#                     8,
#                     4,
#                     -5,
#                     6,
#                     4,
#                     7,
#                     -4,
#                     3,
#                     2,
#                     6,
#                     -6,
#                     2,
#                     4,
#                     6,
#                     7,
#                     9,
#                     10,
#                     6,
#                     7,
#                     10,
#                     3,
#                     6,
#                     8,
#                     10,
#                     -8,
#                     3,
#                     9,
#                 ]
#             ]
#         )


# class BloodDataset(LAATDataset):
#     """WORKS WITH THIS!!! NIJE LUDILO ALI RADI
#         model_kwargs = {
#         # "module": PModel,
#         "llm_ratings": dataset.llm_ratings,
#         "lr": 0.1,
#         "max_epochs": nsteps,
#         # "train_split": None,
#         "optimizer": torch.optim.Adam,
#         # "optimizer__momentum": 0.9,
#         "optimizer__weight_decay": 0.0,
#         "verbose": False,
#         "device": device,
#     }"""

#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__()

#         blood_transfusion_service_center = fetch_ucirepo(id=176)

#         # data (as pandas dataframes)
#         X = blood_transfusion_service_center.data.features
#         y = blood_transfusion_service_center.data.targets

#         X = X.dropna()
#         y = y.loc[X.index]

#         self.X = torch.from_numpy(X.to_numpy()).float()
#         self.y = torch.from_numpy(y.to_numpy()).flatten().float()

#         self.llm_ratings = torch.tensor([[-8.0, 7, 5, 3]])


# class CreditGDataset(LAATDataset):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__()

#         X, y, default_target_attribute, label_list, categorical_indicator = get_feat_llmdataset("credit-g", 2, 69)
#         for i, label in enumerate(label_list):
#             y[y == label] = i
#         X = pd.get_dummies(X)

#         self.X = torch.from_numpy(X.astype(float).to_numpy()).float()
#         self.y = torch.from_numpy(y.astype(float)).flatten().float()

#         self.llm_ratings = torch.tensor(
#             [
#                 [
#                     6.0,  # Duration
#                     -5,  # CreditAmount
#                     -6,  # InstallmentCommitment
#                     4,  # ResidenceSince
#                     3,  # Age
#                     -4,  # ExistingCredits
#                     -3,  # NumDependents
#                     2,  # CheckingStatus_0<=X<200
#                     -8,  # CheckingStatus_<0
#                     8,  # CheckingStatus_>=200
#                     -6,  # CheckingStatus_no checking
#                     7,  # CreditHistory_all paid
#                     -9,  # CreditHistory_critical/other existing credit
#                     -7,  # CreditHistory_delayed previously
#                     5,  # CreditHistory_existing paid
#                     6,  # CreditHistory_no credits/all paid
#                     2,  # Purpose_business
#                     0,  # Purpose_domestic appliance
#                     3,  # Purpose_education
#                     0,  # Purpose_furniture/equipment
#                     2,  # Purpose_new car
#                     -1,  # Purpose_other
#                     -2,  # Purpose_radio/tv
#                     1,  # Purpose_repairs
#                     3,  # Purpose_retraining
#                     1,  # Purpose_used car
#                     4,  # SavingsStatus_100<=X<500
#                     6,  # SavingsStatus_500<=X<1000
#                     -5,  # SavingsStatus_<100
#                     9,  # SavingsStatus_>=1000
#                     -7,  # SavingsStatus_no known savings
#                     3,  # Employment_1<=X<4
#                     5,  # Employment_4<=X<7
#                     -6,  # Employment_<1
#                     7,  # Employment_>=7
#                     -10,  # Employment_unemployed
#                     1,  # PersonalStatus_female div/dep/mar
#                     -1,  # PersonalStatus_male div/sep
#                     2,  # PersonalStatus_male mar/wid
#                     -1,  # PersonalStatus_male single
#                     5,  # OtherParties_co applicant
#                     4,  # OtherParties_guarantor
#                     -3,  # OtherParties_none
#                     2,  # PropertyMagnitude_car
#                     4,  # PropertyMagnitude_life insurance
#                     -6,  # PropertyMagnitude_no known property
#                     8,  # PropertyMagnitude_real estate
#                     -2,  # OtherPaymentPlans_bank
#                     3,  # OtherPaymentPlans_none
#                     -4,  # OtherPaymentPlans_stores
#                     1,  # Housing_for free
#                     7,  # Housing_own
#                     -3,  # Housing_rent
#                     6,  # Job_high qualif/self emp/mgmt
#                     4,  # Job_skilled
#                     -8,  # Job_unemp/unskilled non res
#                     -4,  # Job_unskilled resident
#                     -3,  # OwnTelephone_none
#                     2,  # OwnTelephone_yes
#                     5,  # ForeignWorker_no
#                     -5,  # ForeignWorker_yes
#                 ]
#             ]
#         )


# class MyocardialDataset(LAATDataset):
#     """WORKSSS!!! THE FEATURES ARE A BIT HACKY (142 gpt-4o, 63 gpt-4o-mini). NE DIRAJJJ!!!"""

#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__()

#         X, y, default_target_attribute, label_list, categorical_indicator = get_feat_llmdataset("myocardial", 2, 69)
#         for i, label in enumerate(label_list):
#             y[y == label] = i
#         X = pd.get_dummies(X)

#         self.X = torch.from_numpy(X.astype(float).to_numpy()).float()
#         self.y = torch.from_numpy(y.astype(float)).flatten().float()
#         self.llm_ratings = torch.tensor(
#             [
#                 [
#                     7.0,
#                     3,
#                     2,
#                     -2,
#                     -1,
#                     1,
#                     4,
#                     -1,
#                     1,
#                     3,
#                     7,
#                     5,
#                     -6,
#                     2,
#                     6,
#                     1,
#                     -5,
#                     4,
#                     2,
#                     3,
#                     -2,
#                     3,
#                     6,
#                     8,
#                     -6,
#                     5,
#                     -4,
#                     6,
#                     2,
#                     4,
#                     6,
#                     -5,
#                     -3,
#                     3,
#                     3,
#                     2,
#                     1,
#                     6,
#                     -1,
#                     -6,
#                     2,
#                     1,
#                     4,
#                     9,
#                     7,
#                     -10,
#                     -3,
#                     3,
#                     -1,
#                     2,
#                     -1,
#                     2,
#                     -3,
#                     5,
#                     -5,
#                     7,
#                     -2,
#                     -3,
#                     5,
#                     -2,
#                     -3,
#                     5,
#                     -2,
#                     4,
#                     -1,
#                     -4,
#                     6,
#                     -2,
#                     -3,
#                     -3,
#                     5,
#                     -2,
#                     4,
#                     -1,
#                     2,
#                     -1,
#                     2,
#                     -1,
#                     2,
#                     -2,
#                     3,
#                     -2,
#                     4,
#                     -1,
#                     2,
#                     -5,
#                     8,
#                     -4,
#                     7,
#                     -2,
#                     3,
#                     -2,
#                     3,
#                     -2,
#                     4,
#                     -3,
#                     5,
#                     -3,
#                     3,
#                     5,
#                     4,
#                     -5,
#                     -3,
#                     2,
#                     4,
#                     3,
#                     -4,
#                     -3,
#                     3,
#                     5,
#                     4,
#                     -5,
#                     -3,
#                     3,
#                     5,
#                     4,
#                     -5,
#                     -3,
#                     5,
#                     -3,
#                     3,
#                     -2,
#                     4,
#                     -2,
#                     3,
#                     -2,
#                     -3,
#                     3,
#                     -2,
#                     3,
#                     -2,
#                     3,
#                     -2,
#                     4,
#                     -2,
#                     3,
#                     -2,
#                     4,
#                     -3,
#                     5,
#                     -4,
#                     6,
#                     -2,
#                     6,
#                     -2,
#                     -2,
#                     -1,
#                     -1,
#                     3,
#                     -1,
#                     4,
#                     -2,
#                     -2,
#                     6,
#                     -1,
#                     5,
#                     -1,
#                     5,
#                     -1,
#                     4,
#                     -1,
#                     6,
#                     -1,
#                     4,
#                     -2,
#                     6,
#                     -3,
#                     4,
#                     -3,
#                     4,
#                     -3,
#                     4,
#                     -3,
#                     5,
#                     -3,
#                     5,
#                     -3,
#                     5,
#                     -3,
#                     5,
#                     2,
#                     1,
#                     1,
#                     1,
#                     1,
#                     5,
#                     -3,
#                     -4,
#                     -5,
#                     -2,
#                     5,
#                     -2,
#                     5,
#                     -1,
#                     4,
#                     -1,
#                     3,
#                     -1,
#                     3,
#                     -2,
#                     4,
#                     -2,
#                     4,
#                     -1,
#                     3,
#                 ]
#             ]
#         )


# class PC1Dataset(LAATDataset):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__()

#         X, y, default_target_attribute, label_list, categorical_indicator = get_feat_llmdataset("pc1", 2, 69)
#         for i, label in enumerate(label_list):
#             y[y == label] = i
#         X = pd.get_dummies(X)

#         self.X = torch.from_numpy(X.astype(float).to_numpy()).float()
#         self.y = torch.from_numpy(y.astype(float)).flatten().float()

#         self.llm_ratings = torch.tensor([[5.0, 7, 3, 6, 4, 5, -2, -3, 6, 7, 9, 8, 5, -1, 4, -2, 3, 4, 5, 4, 8]])


# class KC2Dataset(LAATDataset):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__()

#         X, y, default_target_attribute, label_list, categorical_indicator = get_feat_llmdataset("kc2", 2, 69)
#         for i, label in enumerate(label_list):
#             y[y == label] = i
#         X = pd.get_dummies(X)

#         self.X = torch.from_numpy(X.astype(float).to_numpy()).float()
#         self.y = torch.from_numpy(y.astype(float)).flatten().float()

#         self.llm_ratings = torch.tensor([[5.0, 8, 4, 7, 6, 7, 3, 2, 4, 8, 10, 5, 5, -4, 3, -1, 4, 4, 6, 6, 8]])


# class WineDataset(LAATDataset):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__()

#         X, y, default_target_attribute, label_list, categorical_indicator = get_feat_llmdataset("wine", 2, 69)
#         for i, label in enumerate(label_list):
#             y[y == label] = i
#         X = pd.get_dummies(X)

#         self.X = torch.from_numpy(X.astype(float).to_numpy()).float()
#         self.y = torch.from_numpy(y.astype(float)).flatten().float()

#         # o3-mini
#         self.llm_ratings = torch.tensor([[-5.0, -5, +2, +9, -3, +8, +6, -5, -4, -6, -1]])


# class CDCDiabetesDataset(LAATDataset):
#     """WORKS!!! NE DIRAJ!!! Nije bas super predobro, oko 0.05 ROC iznad RF i LR, ali radi"""

#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__()

#         X, y, default_target_attribute, label_list, categorical_indicator = get_feat_llmdataset("cdc-diabetes", 2, 69)
#         for i, label in enumerate(label_list):
#             y[y == label] = i
#         X = pd.get_dummies(X)

#         self.X = torch.from_numpy(X.astype(float).to_numpy()).float()
#         self.y = torch.from_numpy(y.astype(float)).flatten().float()

#         self.llm_ratings = torch.tensor([[7.0, 6, 2, 8, 3, 6, 7, -5, -3, -3, -2, 1, 2, 9, 4, 7, 8, -1, 9, -4, -3]])


# class Rival10Dataset(LAATDataset):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__()

#         X, y, default_target_attribute, label_list, categorical_indicator = get_feat_llmdataset("rival10", 2, 69)
#         for i, label in enumerate(label_list):
#             y[y == label] = i

#         self.X = torch.from_numpy(X.astype(float).to_numpy()).float()
#         self.y = torch.from_numpy(y).flatten().long()

#         # simplify - cat classification
#         wanted_classes = [5, 4]
#         self.X = self.X[(self.y == wanted_classes[0]) | (self.y == wanted_classes[1])]
#         self.y = self.y[(self.y == wanted_classes[0]) | (self.y == wanted_classes[1])]

#         self.y = torch.where(self.y == wanted_classes[1], 1.0, 0.0)

#         self.llm_ratings = torch.tensor([[-4, -8, -10, -2, -7, 6, 9, 7, 10, 3, -9, 8, -6, -3, -1, 2, -5, 5]])  # cat

#         # self.llm_ratings = torch.tensor(
#         #     [
#         #         ### gpt-4o
#         #         [-8.0, -10, 10, 6, -6, -9, -8, -9, -10, -10, -10, -9, 8, 9, 0, 7, 6, -1],  # truck
#         #         [-8, -10, 10, 3, -6, -9, -8, -9, -10, -10, -10, -9, 7, 6, 0, 4, -2, 0],  # car
#         #         [-8, 10, 7, 3, -9, -10, -9, -10, 5, -10, -9, -10, 9, 4, -3, 6, 2, -1],  # plane
#         #         ### gpt-4o-mini
#         #         [-2, -5, -8, 2, 6, -7, -6, -9, -6, -8, -9, -7, 9, 6, 7, 8, 3, 4],  # ship
#         #         [-4, -8, -10, -2, -7, 6, 9, 7, 10, 3, -9, 8, -6, -3, -1, 2, -5, 5],  # cat
#         #         [5, -5, -6, 0, -4, 8, 4, 3, 7, -3, -6, 6, -2, -1, 2, 2, 1, 4],  # dog
#         #         [8, -7, -8, -2, -6, 6, 7, 2, 7, 8, -8, 5, -5, -3, 0, 6, 7, 1],  # equine
#         #         [8, -8, -7, 0, 6, 7, 9, 0, 5, -6, -9, 3, -5, -4, 0, 0, 0, 0],  # deer
#         #         [4, -8, -10, -5, -3, 3, 6, 5, 2, -9, -7, -6, -8, -4, 7, 2, -2, 4],  # frog
#         #         [2, 10, -7, -5, -4, -6, 1, 3, 8, -3, 10, -8, -5, -2, 1, 4, 3, 6],  # bird
#         #     ]
#         # )

#         # gpt-4o-mini at once
#         # self.llm_ratings = torch.tensor(
#         #     [
#         #         [-5.0, -3, 8, 5, -2, -3, -2, -1, -2, -3, -2, -2, 8, 7, -2, 6, 4, -1],
#         #         [-3, -2, 8, 4, -2, -2, -2, -2, -2, -2, -2, -2, 9, 7, -3, 6, 4, -1],
#         #         [-1, 10, -1, 2, -3, -4, -3, -2, -3, -2, -2, -3, 3, 4, -2, 4, 2, -2],
#         #         [-2, -2, 6, 6, -2, -4, -2, -2, -2, -2, -2, -2, 7, 8, 6, 7, 5, -2],
#         #         [5, -3, -3, -2, 2, 5, 6, 6, 6, -1, -2, 5, -3, -2, -1, 1, 2, 5],
#         #         [5, -3, -3, -3, -1, 6, 6, 6, 7, -1, -2, 6, -4, -3, -3, 1, 1, 4],
#         #         [6, -2, -2, -3, 6, 2, 5, 5, 5, 10, -2, 5, -4, -3, -3, 3, 6, 4],
#         #         [4, -2, -3, -4, 3, 2, 7, 3, 6, 4, -2, 5, -5, -4, -4, 4, 7, 6],
#         #         [1, 1, -4, -5, 0, -2, 1, 4, 4, -3, 10, -2, -6, -5, 5, -2, 3, 3],
#         #         [2, 9, -3, -4, -2, -1, 2, 2, 2, -2, 8, -1, -5, -3, 4, 1, 2, 4],
#         #     ]
#         # )

#         # gpt-o3-mini at once
#         # self.llm_ratings = torch.tensor(
#         #     [
#         #         # truck
#         #         [0.0, 0, 8, 5, 2, 0, 0, 0, -3, 0, 0, -3, 8, 5, -1, 5, 4, -2],
#         #         # car
#         #         [0, 0, 8, 5, 2, 0, 0, 0, -3, 0, 0, -3, 8, 5, -1, 2, 0, -2],
#         #         # plane
#         #         [0, 10, 3, 5, 0, 0, 0, 0, 2, 0, 2, -3, 8, 4, -1, 7, 2, -2],
#         #         # ship
#         #         [0, 0, -5, 5, 0, 0, 0, 0, -3, 0, 0, -3, 8, 5, 6, 8, 4, -2],
#         #         # cat
#         #         [-2, 0, -5, -3, -3, 1, 4, 3, 8, -2, -3, 7, -4, -2, -1, 0, 0, 5],
#         #         # dog
#         #         [4, 0, -5, -3, -3, 8, 4, 3, 8, -2, -3, 7, -4, -2, -1, 0, 0, 4],
#         #         # equine
#         #         [2, 0, -5, -3, -3, 0, 3, 2, 6, 9, -3, 6, -4, -2, -1, 3, 5, 4],
#         #         # deer
#         #         [1, 0, -5, -3, -3, 0, 3, 1, 6, -2, -3, 6, -4, -2, -1, 2, 3, 3],
#         #         # frog
#         #         [0, 0, -5, -3, -3, 0, 0, 1, -3, -2, -3, -2, -4, -2, 4, -2, -2, 2],
#         #         # bird
#         #         [0, 10, -5, -3, -3, 0, 2, 2, 4, -2, 10, -2, -4, -2, -1, 1, 1, 3],
#         #     ]
#         # )


# class ViralPneumoniaDataset(LAATDataset):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__()

#         X, y, default_target_attribute, label_list, categorical_indicator = get_feat_llmdataset("covid19", 2, 69)
#         for i, label in enumerate(label_list):
#             y[y == label] = i

#         self.X = torch.from_numpy(X.astype(float).to_numpy()).float()
#         self.y = torch.from_numpy(y.astype(float)).flatten().float()

#         self.llm_ratings = torch.tensor([[8.0, 9, 10, 7, 8, 6, 5, 4, 5, 6]])


# class BreastLjubljanaDataset(LAATDataset):
#     """Kinda radi!!!"""

#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__()

#         X, y, default_target_attribute, label_list, categorical_indicator = get_feat_llmdataset(
#             "breast-ljubljana", 2, 69
#         )
#         for i, label in enumerate(label_list):
#             y[y == label] = i
#         X = pd.get_dummies(X)

#         self.X = torch.from_numpy(X.astype(float).to_numpy()).float()
#         self.y = torch.from_numpy(y.astype(float)).flatten().float()

#         # gpt-o3-mini
#         self.llm_ratings = torch.tensor([[-2.0, 8, 10, 7, 8, -5, -1, -3, 3, 0, 0, 2, 0, 0, 0, 0]])
