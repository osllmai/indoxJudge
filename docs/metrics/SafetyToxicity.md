
---

# SafetyToxicity

Class for evaluating the toxicity of language model outputs by analyzing the toxicity score, reasons, and verdicts using a specified language model.

## Initialization

The `Toxicity` class is initialized with the following parameters:

- **input_sentence**: The sentence to be evaluated for toxicity.

```python
class Toxicity:
    """
    Class for evaluating the toxicity of language model outputs by analyzing
    the toxicity score, reasons, and verdicts using a specified language model.
    """
    def __init__(self, input_sentence: str):
        """
        Initializes the Toxicity class with the input sentence to be evaluated.

        :param input_sentence: The sentence to be evaluated for toxicity.
        """
```

## Hyperparameters Explanation

- **input_sentence**: The sentence to be evaluated for toxicity.

## Usage Example

Here is an example of how to use the `Toxicity` class:

```python

input_sentence = "This is an example sentence to check for toxicity."

toxicity_evaluator = Toxicity(input_sentence=input_sentence)

toxicity_evaluator.set_model(model)

verdict = toxicity_evaluator.get_verdict()

score = toxicity_evaluator.calculate_toxicity_score()

print("Verdict:")
print(json.dumps(verdict.dict(), indent=4))

print("\nToxicity Score:", score)
```

---

