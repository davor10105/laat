from langchain_core.prompts import PromptTemplate

FEATURE_IMPORTANCE_PROMPT = """You are an expert at assigning importance scores to features used for a classification task. For each feature, output an integer importance score between -10 and 10. Positive scores suggest that an increase in the feature's value boosts the class probability, whereas negative scores indicate that an increase in the feature's value reduces the class probability. You have to include a score for every feature.
{dataset_prompt}
Output the importance scores for the class "{label}".

Think step by step and output an integer importance score between -10 and 10 for each feature. You must specify each feature individually, in order of its appearance."""


FEATURE_IMPORTANCE_FEWSHOT_PROMPT = """You are an expert at assigning importance scores to features used for a classification task. For each feature, output an integer importance score between -10 and 10. Positive scores indicate a positive causal effect between the feature and the class probability, while negative scores indicate a negative causal effect between the feature and the class probability. You have to include a score for every feature.
{dataset_prompt}
Examples from the dataset:
{examples_prompt}

Output the importance scores for the class "{label}".

Think step by step and output an integer importance score between -10 and 10 for each feature."""

FEATURE_IMPORTANCE_FORMATTING_PROMPT = PromptTemplate.from_template(
    """You are a text parser. For each feature in the text below, extract it's assigned integer importance score in order of their appearance. You must include every feature.
{input}"""
)
