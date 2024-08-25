# Robustness to Adversarial Demonstrations

Class for evaluating the robustness of language model outputs to adversarial demonstrations by analyzing the robustness score, reasons, and verdicts using a specified language model.

## Initialization

The `RobustnessToAdversarialDemonstrations` class is initialized with the following parameters:

- **input_sentence**: The sentence to be evaluated for robustness against adversarial demonstrations.

```python
class RobustnessToAdversarialDemonstrations:
    """
    Class for evaluating the robustness of language model outputs to adversarial
    demonstrations by analyzing the robustness score, reasons, and verdicts using
    a specified language model.
    """
    def __init__(self, input_sentence: str):
        """
        Initializes the RobustnessToAdversarialDemonstrations class with the input sentence.

        :param input_sentence: The sentence to be evaluated for robustness against adversarial demonstrations.
        """
```

## Hyperparameters Explanation

- **input_sentence**: The response from the language model that needs to be evaluated for adversarial robustness.

## Usage Example

Here is an example of how to use the `RobustnessToAdversarialDemonstrations` class:

```python
from adversarial_robustness import RobustnessToAdversarialDemonstrations

# Define the input sentence to be evaluated
input_sentence = "The model confidently predicts an incorrect outcome under adversarial conditions."

# Initialize the RobustnessToAdversarialDemonstrations class with the provided input sentence
robustness_evaluator = RobustnessToAdversarialDemonstrations(input_sentence=input_sentence)

# Set the model for evaluation
robustness_evaluator.set_model(model)

# Get the robustness reasons for adversarial demonstrations
reasons = robustness_evaluator.get_adversarial_robustness()
print("Adversarial Robustness Reasons:", reasons)

# Get the detailed reason for adversarial robustness
reason = robustness_evaluator.get_reason()
print("Detailed Reason:", reason)

# Get the adversarial robustness verdict
verdict = robustness_evaluator.get_verdict()
print("Adversarial Robustness Verdict:", verdict)

# Calculate and print the adversarial robustness score
robustness_score = robustness_evaluator.calculate_adversarial_robustness_score()
print("Adversarial Robustness Score:", robustness_score)
```

## Methods

### `set_model(model)`

Sets the language model to be used for evaluating the input sentence.

- **model**: The language model to be used.

### `get_adversarial_robustness() -> List[str]`

Returns a list of robustness reasons for the input sentence if the robustness score is greater than 0.

### `get_reason() -> AdversarialDemonstrationsReason`

Returns a detailed reason for the robustness of the input sentence against adversarial demonstrations.

### `get_verdict() -> AdversarialDemonstrationsVerdict`

Returns the verdict of whether the input sentence is robust against adversarial demonstrations, along with the reason and the score.

### `calculate_adversarial_robustness_score() -> float`

Calculates and returns the adversarial robustness score based on the verdict.

### `_call_language_model(prompt: str) -> str`

Internal method that calls the language model with the provided prompt and returns the response.

## Error Handling

- Handles `json.JSONDecodeError` when parsing the response from the language model, returning appropriate fallback values and logging errors.

## Notes

- The adversarial robustness score and verdicts are determined based on the language model's output, which may vary depending on the model's parameters and training data.
- Ensure that the model passed to `set_model()` implements the `generate_evaluation_response()` method as expected by the `_call_language_model()` function.
