
---

# SafetyEvaluator

## Overview

The `SafetyEvaluator` class is designed to assess various safety-related aspects of a given input using a set of predefined metrics. This class includes metrics such as Fairness, Harmfulness, Privacy, Misinformation, MachineEthics, and StereotypeBias. It provides a structured approach to evaluating these dimensions and visualizing the results, making it suitable for comprehensive safety evaluations of text inputs.

## Initialization

The `SafetyEvaluator` class is initialized with two main components:

- **model**: The model used for evaluating the safety metrics.
- **input**: The input text or sentence that will be evaluated against the safety metrics.

### Example

```python
class SafetyEvaluator:
    def __init__(self, model, input):
        """
        Initializes the SafetyEvaluator with a model and input sentence.

        Args:
            model: The model used for evaluating the metrics.
            input: The input text that will be evaluated.
        """
        self.model = model
        self.metrics = [
            Fairness(input_sentence=input),
            Harmfulness(input_sentence=input),
            Privacy(input_sentence=input),
            Misinformation(input_sentence=input),
            MachineEthics(input_sentence=input),
            StereotypeBias(input_sentence=input)
        ]
        logger.info("Evaluator initialized with model and metrics.")
        self.set_model_for_metrics()
        self.evaluation_score = 0
        self.metrics_score = {}
        self.metrics_reasons = {}
```

## Setting the Model for Metrics

The `set_model_for_metrics` method ensures that the model is properly set for each metric that requires it. This step is crucial for metrics that need direct access to the model to perform their evaluations.

---
### Usage Example

```python
import os
from dotenv import load_dotenv
from indoxJudge.pipelines.safetyEvaluator.safety_evaluator import *
load_dotenv()
INDOX_API_KEY = os.getenv("INDOX_API_KEY")
from indoxJudge.models import IndoxApi
model = IndoxApi(api_key=INDOX_API_KEY)

input_sentence = "Most people believe that teamwork is essential for successful project completion, although some individuals may work better independently."

metrics_to_evaluate = ["Fairness", "Harmfulness"]

safety_model = SafetyEvaluator(model=model, input=input_sentence)

metrics_score, metrics_reasons = safety_model.judge()

# Print the evaluation results
print("Metrics Scores:")
print(json.dumps(metrics_score, indent=4))

print("\nMetrics Reasons:")
print(json.dumps(metrics_reasons, indent=4))
print("Transformed Metrics:", transformed_metrics)


```


