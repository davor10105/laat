from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate

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

METADATA_GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a feature description generator. User will supply you "
                "with a dataset description and a list of features. Your task is "
                "to, for each feature, provide a brief description of the feature. "
                "Any feature containing an underscore indicates a specific category of a feature."
            ),
        ),
        (
            "user",
            "Provide a description for the following dataset:\n"
            "Dataset description: {dataset_description}\n"
            "Features: {features}",
        ),
    ]
)
