# RagEvaluator
The `RagEvaluator` class is designed to evaluate various aspects of language model outputs using specified metrics.
# Initialization
The `RagEvaluator` class is initialized with a language model, a language model response, a retrieval context, and a query.

``` python
def __init__(self, llm_as_judge: object, llm_response: str, retrieval_context: Union[str, list[str]], query: str):
    """
    Initializes the RagEvaluator with a language model and a list of metrics.

    Args:
        llm_as_judge: The language model.
        llm_response: The response from the language model.
        retrieval_context: The context retrieved for the query.
        query: The input query.
    """
    self.model = llm_as_judge
    self.metrics = [
        Faithfulness(llm_response=llm_response, retrieval_context=retrieval_context),
        AnswerRelevancy(query=query, llm_response=llm_response),
        ContextualRelevancy(query=query, retrieval_context=retrieval_context),
        GEval(parameters="Rag Pipeline", llm_response=llm_response, query=query, retrieval_context=retrieval_context),
        Hallucination(llm_response=llm_response, retrieval_context=retrieval_context),
        KnowledgeRetention(messages=[{"query": query, "llm_response": llm_response}]),
        BertScore(llm_response=llm_response, retrieval_context=retrieval_context),
        METEOR(llm_response=llm_response, retrieval_context=retrieval_context),
    ]
    logger.info("RagEvaluator initialized with model and metrics.")
    self.set_model_for_metrics()
    self.evaluation_score = 0
    self.metrics_score = {}
```
# Parameters Explanation

- **llm_as_judge**: object: The language model.
- **llm_response**: str: The response from the language model.
- **retrieval_context**: Union[str, list[str]]: The context retrieved for the query.
- **query**: str: The input query.

# Usage Example
Here's an example of how to use the RagEvaluator class:

``` python
from indoxJudge.rag_evaluator import RagEvaluator

# Define a sample response and context and query
response = "The quick brown fox jumps over the lazy dog. This is a common English pangram that contains all the letters of the alphabet."
context = ["The quick brown fox is a phrase that demonstrates the use of every letter in the English alphabet.", "Pangrams are sentences that contain all the letters of the alphabet."]
query = "What is a pangram?"

# Initialize the RagEvaluator
rag_evaluator = RagEvaluator(llm_as_judge=llm, llm_response=response, retrieval_context=context, query=query)

# Evaluate the language model
results = rag_evaluator.judge()
print(results)

scores = rag_evaluator.metrics_score
print(scores)

# Plot
rag_evaluator.plot()
```