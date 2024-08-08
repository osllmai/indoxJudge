# AnswerRelevancy

Class for evaluating the relevancy of language model outputs by analyzing statements, generating verdicts, and calculating relevancy scores.

## Initialization

The `AnswerRelevancy` class is initialized with the following parameters:

- `query`: The query being evaluated.
- `llm_response`: The response generated by the language model.
- `model`: The language model to use for evaluation. Defaults to `None`.
- `threshold`: The threshold for determining relevancy. Defaults to `0.5`.
- `include_reason`: Whether to include reasoning for the relevancy verdicts. Defaults to `True`.
- `strict_mode`: Whether to use strict mode, which forces a score of 0 if relevancy is below the threshold. Defaults to `False`.

```python
class AnswerRelevancy:
    """
    Class for evaluating the relevancy of language model outputs by analyzing statements,
    generating verdicts, and calculating relevancy scores.
    """
    def __init__(self, query: str, llm_response: str, model=None, threshold: float = 0.5, include_reason: bool = True,
                 strict_mode: bool = False):
        """
        Initializes the AnswerRelevancy class with the query, LLM response, and evaluation settings.

        :param query: The query being evaluated.
        :param llm_response: The response generated by the language model.
        :param model: The language model to use for evaluation. Defaults to None.
        :param threshold: The threshold for determining relevancy. Defaults to 0.5.
        :param include_reason: Whether to include reasoning for the relevancy verdicts. Defaults to True.
        :param strict_mode: Whether to use strict mode, which forces a score of 0 if relevancy is below the threshold. Defaults to False.
        """
```
# Hyperparameters Explanation

- **query**: A string containing the query for which relevancy is being evaluated.

- **llm_response**: A string containing the response from the language model that needs to be evaluated.

- **model**: This is the language model object used for the evaluation process. If not provided, the default model is used.

- **threshold**: A float value representing the minimum relevancy score required for a response to be considered relevant. The default value is 0.5.

- **include_reason**: A boolean indicating whether the evaluation should include detailed reasons for the relevancy verdict. Default is True.

- **strict_mode**: A boolean that, when set to True, ensures that any score below the threshold results in a relevancy score of 0. This is useful for enforcing strict relevancy criteria. Default is False.

# Usage Example

Here is an example of how to use the `AnswerRelevancy` class:

```python
import os
from dotenv import load_dotenv
from indoxJudge.models import OpenAi
from indoxJudge.metrics import AnswerRelevancy
from indoxJudge import Evaluator

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the language model
llm = OpenAi(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")

# Define the query and the response to be evaluated
query = "What is the capital of France?"
llm_response = "The capital of France is Paris."

# Initialize the AnswerRelevancy metric
answer_relevancy_metric = AnswerRelevancy(
    query=query, 
    llm_response=llm_response, 
    model=llm, 
    threshold=0.5, 
    include_reason=True, 
    strict_mode=False
)

evaluator = Evaluator(model=llm, metrics=[answer_relevancy_metric])
result = evaluator.judge()
```

# Hyperparameters Explanation

- **query**: A string representing the query for which the contextual relevancy is being evaluated.

- **retrieval_context**: A list of strings, each representing a context retrieved in response to the query.
