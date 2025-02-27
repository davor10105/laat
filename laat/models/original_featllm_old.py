import torch
from torch import nn
import torch.nn.functional as F
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
from langchain_core.runnables import ConfigurableField
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from laat.prompts.original_featllm import ASK_LLM_PROMPT, GENERATE_FUNCTION_PROMPT
from laat.datasets import LAATDataset
import random
from pathlib import Path
import os
import pickle
import json
import time
import copy
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm


class simple_model(nn.Module):
    def __init__(self, X):
        super(simple_model, self).__init__()
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.ones(x_each.shape[1], 1) / x_each.shape[1]) for x_each in X]
        )

    def forward(self, x):
        x_total_score = []
        for idx, x_each in enumerate(x):
            x_score = x_each @ torch.clamp(self.weights[idx], min=0)
            x_total_score.append(x_score)
        x_total_score = torch.cat(x_total_score, dim=-1)
        return x_total_score


class OriginalFeatLLMUtilities:
    @staticmethod
    def _serialize(row) -> str:
        target_str = f""
        for attr_idx, attr_name in enumerate(list(row.index)):
            if attr_idx < len(list(row.index)) - 1:
                target_str += " is ".join([attr_name, str(row[attr_name]).strip(" .'").strip('"').strip()])
                target_str += ". "
            else:
                if len(attr_name.strip()) < 2:
                    continue
                target_str += " is ".join([attr_name, str(row[attr_name]).strip(" .'").strip('"').strip()])
                target_str += "."
        return target_str

    @staticmethod
    def _fill_in_templates(fill_in_dict, template_str):
        for key, value in fill_in_dict.items():
            if key in template_str:
                template_str = template_str.replace(key, value)
        return template_str

    @staticmethod
    def get_prompt_for_asking(
        dataset: LAATDataset, X_train: pd.DataFrame, y_train: pd.DataFrame, num_ensembles: int = 20
    ) -> tuple[Any, Any]:
        prompt_type_str = ASK_LLM_PROMPT

        try:
            with open(dataset.meta_file_name, "r") as f:
                meta_data = json.load(f)
        except:
            meta_data = {}

        task_desc = dataset.feature_descriptions["#DatasetTask#"]
        df_incontext = X_train.copy()
        print(f"Num shots for generating {len(X_train)}")
        # featllm *sigh*
        default_target_attribute = y_train.columns[0]
        num_query = num_ensembles

        df_incontext[default_target_attribute] = y_train.copy()
        df_all = dataset.X.copy()

        format_list = [
            f'10 different conditions for class "{label}":\n- [Condition]\n...' for label in dataset.label_list
        ]
        format_desc = "\n\n".join(format_list)

        template_list = []
        current_query_num = 0
        end_flag = False
        while True:
            if current_query_num >= num_query:
                break

            # Feature bagging
            if len(df_incontext.columns) >= 20:
                total_column_list = []
                for i in range(len(df_incontext.columns) // 10):
                    column_list = df_incontext.columns.tolist()[:-1]
                    random.shuffle(column_list)
                    total_column_list.append(column_list[i * 10 : (i + 1) * 10])
            else:
                total_column_list = [df_incontext.columns.tolist()[:-1]]

            for selected_column in total_column_list:
                if current_query_num >= num_query:
                    break

                # Sample bagging
                threshold = 16
                if len(df_incontext) > threshold:
                    sample_num = int(threshold / df_incontext[default_target_attribute].nunique())
                    df_incontext = df_incontext.groupby(
                        default_target_attribute, group_keys=False, observed=False
                    ).apply(lambda x: x.sample(sample_num))

                feature_name_list = []
                sel_cat_idx = [df_incontext.columns.tolist().index(col_name) for col_name in selected_column]
                is_cat_sel = np.array(dataset.categorical_indicator)[sel_cat_idx]

                for cidx, cname in enumerate(selected_column):
                    if is_cat_sel[cidx] == True:
                        clist = df_all[cname].unique().tolist()
                        if len(clist) > 20:
                            clist_str = f"{clist[0]}, {clist[1]}, ..., {clist[-1]}"
                        else:
                            clist_str = ", ".join(clist)
                        desc = meta_data[cname] if cname in meta_data.keys() else ""
                        feature_name_list.append(
                            f"- {cname}: {desc} (categorical variable with categories [{clist_str}])"
                        )
                    else:
                        desc = meta_data[cname] if cname in meta_data.keys() else ""
                        feature_name_list.append(f"- {cname}: {desc} (numerical variable)")

                feature_desc = "\n".join(feature_name_list)

                in_context_desc = ""
                df_current = df_incontext.copy()
                df_current = df_current.groupby(default_target_attribute, group_keys=False, observed=False).apply(
                    lambda x: x.sample(frac=1)
                )

                for icl_idx, icl_row in df_current.iterrows():
                    answer = icl_row[default_target_attribute]
                    icl_row = icl_row.drop(labels=default_target_attribute)
                    icl_row = icl_row[selected_column]
                    in_context_desc += OriginalFeatLLMUtilities._serialize(icl_row)
                    in_context_desc += f"\nAnswer: {answer}\n"

                fill_in_dict = {
                    "[TASK]": task_desc,
                    "[EXAMPLES]": in_context_desc,
                    "[FEATURES]": feature_desc,
                    "[FORMAT]": format_desc,
                }
                template = OriginalFeatLLMUtilities._fill_in_templates(fill_in_dict, prompt_type_str)
                template_list.append(template)
                current_query_num += 1

        return template_list, feature_desc

    # @staticmethod
    # def query_llm(templates: list[str], llm: BaseChatModel, max_try_num: int = 10) -> list:
    #     result_list = []
    #     for prompt in templates:
    #         curr_try_num = 0
    #         while curr_try_num < max_try_num:
    #             try:
    #                 result = llm.invoke(prompt).content
    #                 result_list.append(result)
    #                 break
    #             except Exception as e:
    #                 print(e)
    #                 curr_try_num += 1
    #                 if curr_try_num >= max_try_num:
    #                     result_list.append(-1)
    #                 time.sleep(10)
    #     return result_list

    @staticmethod
    def query_llm(templates: list[str], llm: BaseChatModel, max_try_num: int = 10) -> list:
        # result_list = []
        # for template in templates:
        #     result_list.append(llm.invoke(template).content)
        result_list = [output.content for output in llm.batch(templates)]
        return result_list

    @staticmethod
    def parse_rules(result_texts, label_list=[]):
        total_rules = []
        splitter = "onditions for class"
        for text in result_texts:
            splitted = text.split(splitter)
            if splitter not in text:
                continue
            if len(label_list) != 0 and len(splitted) != len(label_list) + 1:
                continue

            rule_raws = splitted[1:]
            rule_dict = {}
            for rule_raw in rule_raws:
                class_name = rule_raw.split(":")[0].strip(" .'").strip(' []"')
                rule_parsed = []
                for txt in rule_raw.strip().split("\n")[1:]:
                    if len(txt) < 2:
                        break
                    rule_parsed.append(" ".join(txt.strip().split(" ")[1:]))
                    rule_dict[class_name] = rule_parsed
            total_rules.append(rule_dict)
        return total_rules

    @staticmethod
    def get_prompt_for_generating_function(parsed_rule, feature_desc):
        prompt_type_str = GENERATE_FUNCTION_PROMPT

        template_list = []
        for class_id, each_rule in parsed_rule.items():
            function_name = f"extracting_features_{class_id}"
            rule_str = "\n".join([f"- {k}" for k in each_rule])

            fill_in_dict = {
                "[NAME]": function_name,
                "[CONDITIONS]": rule_str,
                "[FEATURES]": feature_desc,
            }
            template = OriginalFeatLLMUtilities._fill_in_templates(fill_in_dict, prompt_type_str)
            template_list.append(template)

        return template_list

    @staticmethod
    def convert_to_binary_vectors(fct_strs_all, fct_names, label_list, X_train, X_test):
        X_train_all_dict = {}
        X_test_all_dict = {}
        executable_list = []  # Save the parsed functions that are properly working for both train/test sets
        for i in range(len(fct_strs_all)):  # len(fct_strs_all) == # of trials for ensemble
            X_train_dict, X_test_dict = {}, {}
            for label in label_list:
                X_train_dict[label] = {}
                X_test_dict[label] = {}

            # Match function names with each answer class
            fct_idx_dict = {}
            for idx, name in enumerate(fct_names[i]):
                for label in label_list:
                    label_name = "_".join(label.split(" "))
                    if label_name.lower() in name.lower():
                        fct_idx_dict[label] = idx

            # If the number of inferred rules are not the same as the number of answer classes, remove the current trial
            if len(fct_idx_dict) != len(label_list):
                continue

            try:
                for label in label_list:
                    fct_idx = fct_idx_dict[label]
                    exec(fct_strs_all[i][fct_idx].strip('` "'))
                    X_train_each = locals()[fct_names[i][fct_idx]](X_train).astype("int").to_numpy()
                    X_test_each = locals()[fct_names[i][fct_idx]](X_test).astype("int").to_numpy()
                    # assert that the nexamples remain consistent
                    assert X_train.shape[0] == X_train_each.shape[0] and X_test.shape[0] == X_test_each.shape[0]
                    # assert that train and test features are consistent
                    assert X_train_each.shape[1] == X_test_each.shape[1]
                    X_train_dict[label] = torch.tensor(X_train_each).float()
                    X_test_dict[label] = torch.tensor(X_test_each).float()

                X_train_all_dict[i] = X_train_dict
                X_test_all_dict[i] = X_test_dict
                executable_list.append(i)
            except Exception:  # If error occurred during the function call, remove the current trial
                continue

        return executable_list, X_train_all_dict, X_test_all_dict


class OriginalFeatLLMLAATModel(LAATModel):
    def __init__(
        self,
        model_name: str,
        model_class: Type[BaseEstimator],
        pandas_to_numpy_mapper: Callable,
        dataset: LAATDataset,
        reasoning_llm: BaseChatModel,
        parsing_llm: BaseChatModel,
        n_estimators: int = 20,  # number of repetitions of LLM feature engineering and small model training
        save_path: str = "./original_featllm_functions",  # path for saving function strings so that they don't have to be regenerated every time
        num_retries: int = 3,  # number of retries to generate preprocessing function before error
    ):
        self.model_name = model_name
        self.model_class = model_class
        self._pandas_to_numpy_mapper = pandas_to_numpy_mapper
        self.dataset = dataset
        self.reasoning_llm = reasoning_llm.configurable_fields(
            temperature=ConfigurableField(
                id="llm_temperature",
                name="LLM Temperature",
                description="The temperature of the LLM",
            )
        )
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
        X_test, y_test = train_run_info.kwargs["X_test"], train_run_info.kwargs["y_test"]
        # load pre-existing functions if they exist
        self.featllm_preprocess_functions = self._load(train_run_info=train_run_info)
        # if they do not exist, generate them
        # if len(self.featllm_preprocess_functions) != self.n_estimators:
        if len(self.featllm_preprocess_functions) == 0:
            self.featllm_preprocess_functions = self._generate_functions(X_train=X_train, y_train=y_train)
            self._save(train_run_info=train_run_info)
        # preprocess functions
        fct_names, fct_strs_final = self._get_function_names_strings()
        # utilize the functions for preprocessing and fit the models
        executable_list, X_train_all_dict, X_test_all_dict = OriginalFeatLLMUtilities.convert_to_binary_vectors(
            fct_strs_final, fct_names, self.dataset.label_list, X_train, X_test
        )
        # save for prediction
        self.executable_list = executable_list
        self.X_test_all_dict = X_test_all_dict
        # fit the models
        self.models = []
        test_outputs_all = []
        for i in executable_list:
            X_train_now = list(X_train_all_dict[i].values())
            X_test_now = list(X_test_all_dict[i].values())

            # Train
            trained_model = self._train_single(
                X_train_now,
                self.dataset.label_list,
                train_run_info.shot,
                y_train,
                y_test,
            )
            self.models.append(trained_model)

            # evaluate *barf*
            # Evaluate
            test_outputs = trained_model(X_test_now).detach().cpu()
            test_outputs = F.softmax(test_outputs, dim=1).detach()
            test_outputs_all.append(test_outputs)
        test_outputs_all = np.stack(test_outputs_all, axis=0)
        ensembled_probs = test_outputs_all.mean(0)
        ensembled_preds = ensembled_probs.argmax(axis=-1)
        self.ensembled_probs = ensembled_probs
        ensembled_probs = ensembled_probs[:, 1]

        return ensembled_probs, ensembled_preds

    def _train_single(self, X_train_now, label_list, shot, y_train, y_test):
        criterion = nn.CrossEntropyLoss()
        multiclass = True if len(label_list) > 2 else False
        y_train_num = np.array([1 - label_list.index(k) for k in y_train.iloc[:, 0]])
        y_test_num = np.array([1 - label_list.index(k) for k in y_test.iloc[:, 0]])
        if shot == 1:
            model = simple_model(X_train_now)
            opt = torch.optim.Adam(model.parameters(), lr=1e-2)
            for _ in range(200):
                opt.zero_grad()
                outputs = model(X_train_now)
                preds = outputs.argmax(dim=1).numpy()
                acc = (np.array(y_train_num) == preds).sum() / len(preds)
                if acc == 1:
                    break
                loss = criterion(outputs, torch.tensor(y_train_num))
                loss.backward()
                opt.step()
        else:
            if shot <= 2:
                n_splits = 2
            else:
                n_splits = 4

            kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
            model_list = []
            for fold, (train_ids, valid_ids) in enumerate(kfold.split(X_train_now[0], y_train_num)):
                model = simple_model(X_train_now)
                opt = torch.optim.Adam(model.parameters(), lr=1e-2)
                X_train_now_fold = [x_train_now[train_ids] for x_train_now in X_train_now]
                X_valid_now_fold = [x_train_now[valid_ids] for x_train_now in X_train_now]
                y_train_fold = y_train_num[train_ids]
                y_valid_fold = y_train_num[valid_ids]

                max_acc = -1
                for _ in range(200):
                    opt.zero_grad()
                    outputs = model(X_train_now_fold)
                    loss = criterion(outputs, torch.tensor(y_train_fold))
                    loss.backward()
                    opt.step()

                    valid_outputs = model(X_valid_now_fold)
                    preds = valid_outputs.argmax(dim=1).numpy()
                    acc = (np.array(y_valid_fold) == preds).sum() / len(preds)
                    if max_acc < acc:
                        max_acc = acc
                        final_model = copy.deepcopy(model)
                        if max_acc >= 1:
                            break
                model_list.append(final_model)

            sdict = model_list[0].state_dict()
            for key in sdict:
                sdict[key] = torch.stack([model.state_dict()[key] for model in model_list], dim=0).mean(dim=0)

            model = simple_model(X_train_now)
            model.load_state_dict(sdict)
        return model

    def _get_function_names_strings(self) -> tuple[list[str], list[str]]:
        # Get function names and strings
        fct_names = []
        fct_strs_final = []
        for fct_str_pair in self.featllm_preprocess_functions:
            if len(fct_str_pair) <= 1:
                continue
            fct_pair_name = []
            if "def" not in fct_str_pair[0] or "(" not in fct_str_pair[0]:
                continue
            if "def" not in fct_str_pair[1] or "(" not in fct_str_pair[1]:
                continue

            for fct_str in fct_str_pair:
                fct_pair_name.append(fct_str.split("def")[1].split("(")[0].strip())
            fct_names.append(fct_pair_name)
            fct_strs_final.append(fct_str_pair)
        return fct_names, fct_strs_final

    def _generate_functions(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> list[str]:
        templates, feature_descriptions = OriginalFeatLLMUtilities.get_prompt_for_asking(
            dataset=self.dataset, X_train=X_train, y_train=y_train, num_ensembles=self.n_estimators
        )
        rules = OriginalFeatLLMUtilities.query_llm(templates=templates, llm=self.reasoning_llm)
        parsed_rules = OriginalFeatLLMUtilities.parse_rules(rules, self.dataset.label_list)

        featllm_preprocess_functions = []
        for parsed_rule in tqdm(parsed_rules):
            fct_templates = OriginalFeatLLMUtilities.get_prompt_for_generating_function(
                parsed_rule, feature_descriptions
            )
            fct_results = OriginalFeatLLMUtilities.query_llm(
                templates=fct_templates, llm=self.reasoning_llm.with_config(configurable={"llm_temperature": 0.0})
            )
            fct_strs = []
            for fct_result in fct_results:
                if "<start>" in fct_result and "<end>" in fct_result:
                    fct_strs.append(fct_result.split("<start>")[1].split("<end>")[0].strip())
            featllm_preprocess_functions.append(fct_strs)

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
        return featllm_preprocess_functions

    def delete(self, train_run_info: TrainRunInfo) -> tuple[list[str], bool]:
        os.remove(os.path.join(self.save_path, f"{self}_{train_run_info}.pickle"))

    def __str__(self) -> str:
        try:
            model_name = self.reasoning_llm.model_name
        except:
            model_name = self.reasoning_llm.model.split("/")[-1]
        return f"{self.dataset.dataset_name}_{model_name}_{self.n_estimators}"

    def predict(self, X: pd.DataFrame) -> np.array:
        raise NotImplementedError("predict not implemented for featllm")

    def predict_proba(self, X: pd.DataFrame) -> np.array:
        return self.ensembled_probs
        print("I pred the proba")
        executable_list, X_test_all_dict = self.executable_list, self.X_test_all_dict
        test_outputs_all = []
        for i, model in zip(executable_list, self.models):
            X_test_now = list(X_test_all_dict[i].values())
            # Evaluate
            test_outputs = model(X_test_now).detach().cpu()
            test_outputs = F.softmax(test_outputs, dim=1).detach()
            test_outputs_all.append(test_outputs)
        test_outputs_all = np.stack(test_outputs_all, axis=0)
        ensembled_probs = test_outputs_all.mean(0)

        return ensembled_probs

    def clear(self) -> None:
        """Delete model and free up memory"""
        del self.executable_list
        del self.X_test_all_dict
        del self.models
        del self.featllm_preprocess_functions
        torch.cuda.empty_cache()
