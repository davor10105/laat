from langchain_core.prompts import PromptTemplate

CONDITION_EXTRACTION_PROMPT = """You are an expert. Given the task description and the list of features and data examples, you are extracting conditions for each answer class to solve the task.
{dataset_prompt}

Examples:
{examples_prompt}

Let's first understand the problem and solve the problem step by step.

Step 1. Analyze the causal relationship or tendency between each feature and task description based on general knowledge and common sense within a short sentence. 

Step 2. Based on the above examples and Step 1's results, infer 10 different conditions per answer, following the format below. The condition should make sense, well match examples, and must match the format for [condition] according to value type.

Format for Response:
{conditions_prompt}

Format for [Condition]:
For the categorical variable only,
- [Feature_name] is in [list of Categorical_values]
For the numerical variable only,
- [Feature_name] (> or >= or < or <=) [Numerical_value]
- [Feature_name] is within range of [Numerical_range_start, Numerical_range_end]"""


CONDITION_FORMATTING_PROMPT = PromptTemplate.from_template(
    """Extract all conditions related to a class from the following text in order of their appearance:
{input}"""
)

FUNCTION_GENERATION_PROMPT = PromptTemplate.from_template(
    """Provide me a Python code for a function, given description below.
The function must have the following format:
def add_new_features(df: pd.Dataframe) -> pd.Dataframe:
    # function pseudocode
    new_df = pd.DataFrame()
    for condition in conditions:
        new_df[condition_name] = <condition>
    return new_df

The function must utilize pandas to create a new column for each condition described in the Conditions below. The new columns utilize existing features found in the dataframe described under Existing features. Make sure to include all conditions into the new dataframe. Make sure that the function code well matches with its feature type (i.e., numerical, categorical).

Existing features:
{features_prompt}

Conditions: 
{conditions_prompt}


Return only python code in Markdown format, e.g.:
```python
....
```
Do not add any comments, descriptions, and package imports."""
)
