
---

# OutOfDistributionRobustness

Class for evaluating the out-of-distribution (OOD) robustness of language model outputs by analyzing the robustness score, reasons, and verdicts using a specified language model.

## Initialization

The `OutOfDistributionRobustness` class is initialized with the following parameters:

- **input_sentence**: The sentence to be evaluated for out-of-distribution robustness.

```python
class OutOfDistributionRobustness:
    """
    Class for evaluating the out-of-distribution (OOD) robustness of language model outputs by analyzing
    the robustness score, reasons, and verdicts using a specified language model.
    """
    def __init__(self, input_sentence: str):
        """
        Initializes the OutOfDistributionRobustness class with the input sentence to be evaluated.

        :param input_sentence: The sentence to be evaluated for out-of-distribution robustness.
        """
```

## Hyperparameters Explanation

- **input_sentence**: The sentence to be evaluated for out-of-distribution robustness.

## Usage Example

Here is an example of how to use the `OutOfDistributionRobustness` class:

```python
from your_module import OutOfDistributionRobustness  

input_sentence = "The quick brown fox jumps over the lazy dog."

ood_evaluator = OutOfDistributionRobustness(input_sentence=input_sentence)

ood_evaluator.set_model(model)

verdict = ood_evaluator.get_verdict()

score = ood_evaluator.calculate_ood_robustness_score()

print("Verdict:")
print(json.dumps(verdict.dict(), indent=4))

print("\nOOD Robustness Score:", score)
```

---

