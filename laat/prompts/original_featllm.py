ASK_LLM_PROMPT = """You are an expert. Given the task description and the list of features and data examples, you are extracting conditions for each answer class to solve the task.

Task: [TASK]

Features:
[FEATURES]

Examples:
[EXAMPLES]

Let's first understand the problem and solve the problem step by step.

Step 1. Analyze the causal relationship or tendency between each feature and task description based on general knowledge and common sense within a short sentence. 

Step 2. Based on the above examples and Step 1's results, infer 10 different conditions per answer, following the format below. The condition should make sense, well match examples, and must match the format for [condition] according to value type.

Format for Response:
[FORMAT]


Format for [Condition]:
For the categorical variable only,
- [Feature_name] is in [list of Categorical_values]
For the numerical variable only,
- [Feature_name] (> or >= or < or <=) [Numerical_value]
- [Feature_name] is within range of [Numerical_range_start, Numerical_range_end]


Answer: 
Step 1. The relationship between each feature and the task description: """

GENERATE_FUNCTION_PROMPT = """Output pandas code for a function, given description below.
Function format:
<start>
def <function_name>(df_input: pd.DataFrame) -> pd.DataFrame:
    <function_code>
    return df_output
<end>

Function name: [NAME]

Input: Dataframe df_input

Input Features:
[FEATURES]

Output: Dataframe df_output. Create a new dataframe df_output. Each column in df_output refers whether the selected column in df_input follows the condition (True or False). Be sure that the function code well matches with its feature type (i.e., numerical, categorical).

Conditions: 
[CONDITIONS]


You must wrap the function part with <start> and <end> XML tags, and do not add any comments, descriptions, and package importing lines in the code. Make sure that the number of output features is always consistent, regardless of the specific data in the df_input."""
