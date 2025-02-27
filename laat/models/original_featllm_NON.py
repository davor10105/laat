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
    def get_prompt_for_asking(
        data_name,
        df_all,
        df_x,
        df_y,
        label_list,
        default_target_attribute,
        file_name,
        meta_file_name,
        is_cat,
        num_query=20,
    ) -> tuple[str, str]:
        with open(file_name, "r") as f:
            prompt_type_str = f.read()

        try:
            with open(meta_file_name, "r") as f:
                meta_data = json.load(f)
        except:
            meta_data = {}

        task_desc = f"{TASK_DICT[data_name]}\n"
        df_incontext = df_x.copy()
        df_incontext[default_target_attribute] = df_y

        format_list = [f'10 different conditions for class "{label}":\n- [Condition]\n...' for label in label_list]
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
                    df_incontext = df_incontext.groupby(default_target_attribute, group_keys=False).apply(
                        lambda x: x.sample(sample_num)
                    )

                feature_name_list = []
                sel_cat_idx = [df_incontext.columns.tolist().index(col_name) for col_name in selected_column]
                is_cat_sel = np.array(is_cat)[sel_cat_idx]

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
                df_current = df_current.groupby(default_target_attribute, group_keys=False).apply(
                    lambda x: x.sample(frac=1)
                )

                for icl_idx, icl_row in df_current.iterrows():
                    answer = icl_row[default_target_attribute]
                    icl_row = icl_row.drop(labels=default_target_attribute)
                    icl_row = icl_row[selected_column]
                    in_context_desc += serialize(icl_row)
                    in_context_desc += f"\nAnswer: {answer}\n"

                fill_in_dict = {
                    "[TASK]": task_desc,
                    "[EXAMPLES]": in_context_desc,
                    "[FEATURES]": feature_desc,
                    "[FORMAT]": format_desc,
                }
                template = fill_in_templates(fill_in_dict, prompt_type_str)
                template_list.append(template)
                current_query_num += 1

        return template_list, feature_desc


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
        X_test, y_test = train_run_info.kwargs["X_test"], train_run_info.kwargs["y_test"]
        df, target_attr, label_list, is_cat = (
            self.dataset.df,
            self.dataset.target_attr,
            self.dataset.label_list,
            self.dataset.categorical_indicator,
        )
        X_all = df.drop(target_attr, axis=1)
        ask_file_name = "./templates/ask_llm.txt"
        meta_data_name = f"./data/{_DATA}-metadata.json"
        templates, feature_desc = utils.get_prompt_for_asking(
            _DATA,
            X_all,
            X_train,
            y_train,
            label_list,
            target_attr,
            ask_file_name,
            meta_data_name,
            is_cat,
            num_query=_NUM_QUERY,
        )

        _DIVIDER = "\n\n---DIVIDER---\n\n"
        _VERSION = "\n\n---VERSION---\n\n"

        rule_file_name = f"./rules/rule-{_DATA}-{_SHOT}-{_SEED}-{_MODEL.model_name}-{repetition}.out"
        if os.path.isfile(rule_file_name) == False:
            results = utils.query_gpt(_MODEL, templates, _API_KEY, max_tokens=1500, temperature=0.5)
            with open(rule_file_name, "w") as f:
                total_rules = _DIVIDER.join(results)
                f.write(total_rules)
        else:
            with open(rule_file_name, "r") as f:
                total_rules_str = f.read().strip()
                results = total_rules_str.split(_DIVIDER)

        parsed_rules = utils.parse_rules(results, label_list)

        saved_file_name = f"./rules/function-{_DATA}-{_SHOT}-{_SEED}-{_MODEL.model_name}-{repetition}.out"
        if os.path.isfile(saved_file_name) == False:
            function_file_name = "./templates/ask_for_function.txt"
            fct_strs_all = []
            for parsed_rule in tqdm(parsed_rules):
                fct_templates = utils.get_prompt_for_generating_function(parsed_rule, feature_desc, function_file_name)
                fct_results = utils.query_gpt(_MODEL, fct_templates, _API_KEY, max_tokens=1500, temperature=0)
                fct_strs = [fct_txt.split("<start>")[1].split("<end>")[0].strip() for fct_txt in fct_results]
                fct_strs_all.append(fct_strs)

            with open(saved_file_name, "w") as f:
                total_str = _VERSION.join([_DIVIDER.join(x) for x in fct_strs_all])
                f.write(total_str)
        else:
            with open(saved_file_name, "r") as f:
                total_str = f.read().strip()
                fct_strs_all = [x.split(_DIVIDER) for x in total_str.split(_VERSION)]

        # Get function names and strings
        fct_names = []
        fct_strs_final = []
        for fct_str_pair in fct_strs_all:
            fct_pair_name = []
            if "def" not in fct_str_pair[0]:
                continue

            for fct_str in fct_str_pair:
                fct_pair_name.append(fct_str.split("def")[1].split("(")[0].strip())
            fct_names.append(fct_pair_name)
            fct_strs_final.append(fct_str_pair)

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
