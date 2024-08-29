# Adversarial Robustness

Class for evaluating the adversarial robustness of language model outputs by analyzing the robustness score, reasons, and verdicts using a specified language model.

## Initialization

The `AdversarialRobustness` class is initialized with the following parameters:

- **input_sentence**: The sentence to be evaluated for adversarial robustness.

```python
class AdversarialRobustness:
    """
    Class for evaluating the adversarial robustness of language model outputs by analyzing
    the robustness score, reasons, and verdicts using a specified language model.
    """
    def __init__(self, input_sentence: str):
        """
        Initializes the AdversarialRobustness class with the input sentence to be evaluated.

        :param input_sentence: The sentence to be evaluated for adversarial robustness.
        """
```

## Hyperparameters Explanation

- **input_sentence**: The response from the language model that needs to be evaluated for robustness.

## Usage Example

Here is an example of how to use the `AdversarialRobustness` class:

```python
from adversarial_robustness import AdversarialRobustness

# Define the input sentence to be evaluated
input_sentence = "The model predicts that the likelihood of success is low."

# Initialize the AdversarialRobustness class with the provided input sentence
robustness_evaluator = AdversarialRobustness(input_sentence=input_sentence)

# Set the model for evaluation
robustness_evaluator.set_model(model)

# Get the robustness reasons
reasons = robustness_evaluator.get_robustness()
print("Robustness Reasons:", reasons)

# Get the detailed reason for robustness
reason = robustness_evaluator.get_reason()
print("Detailed Reason:", reason)

# Get the robustness verdict
verdict = robustness_evaluator.get_verdict()
print("Robustness Verdict:", verdict)

# Calculate and print the robustness score
robustness_score = robustness_evaluator.calculate_robustness_score()
print("Robustness Score:", robustness_score)
```

## Methods

### `set_model(model)`

Sets the language model to be used for evaluating the input sentence.

- **model**: The language model to be used.

### `get_robustness() -> List[str]`

Returns a list of robustness reasons for the input sentence if the robustness score is greater than 0.

### `get_reason() -> Reason`

Returns a detailed reason for the robustness of the input sentence.

### `get_verdict() -> RobustnessVerdict`

Returns the verdict of whether the input sentence is robust, along with the reason and the score.

### `calculate_robustness_score() -> float`

Calculates and returns the robustness score based on the verdict.

### `_call_language_model(prompt: str) -> str`

Internal method that calls the language model with the provided prompt and returns the response.

## Error Handling

- Handles `json.JSONDecodeError` when parsing the response from the language model, returning appropriate fallback values and logging errors.

## Notes

- The robustness score and verdicts are determined based on the language model's output, which may vary depending on the model's parameters and training data.
- Ensure that the model passed to `set_model()` implements the `generate_evaluation_response()` method as expected by the `_call_language_model()` function.