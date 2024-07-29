# Gruen

Class for evaluating the quality of generated text using various metrics, including grammaticality, redundancy, and focus.

## Initialization

The `Gruen` class is initialized with the following parameters:

- **candidates**: The candidate text(s) to evaluate.
- **use_spacy**: Whether to use `spacy` for focus score calculation.
- **use_nltk**: Whether to use `nltk` for sentence tokenization.

```python
class Gruen:
    def __init__(self, candidates: Union[str, List[str]], use_spacy: bool = True, use_nltk: bool = True):
        """
        Initialize the TextEvaluator with candidate texts and options to use spacy and nltk.

        Parameters:
        candidates (Union[str, List[str]]): The candidate text(s) to evaluate.
        use_spacy (bool): Whether to use spacy for focus score calculation.
        use_nltk (bool): Whether to use nltk for sentence tokenization.
        """
```
## Parameters Explanation

- **candidates**: The actual texts to be evaluated. Can be a single string or a list of strings.

- **use_spacy**: A boolean flag to indicate whether to use `spacy` for focus score calculation. Default is `True`.

- **use_nltk**: A boolean flag to indicate whether to use `nltk` for sentence tokenization. Default is `True`.

## Usage Example

Here is an example of how to use the `Gruen` class:

```python
from indox.IndoxEval import Gruen, Evaluator

# Define sample candidate texts
candidates = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast brown fox leaps over a sleepy dog."
]

# Initialize the Gruen object
gruen = Gruen(
    candidates=candidates,
    use_spacy=True,
    use_nltk=True
)

# Calculate the GRUEN scores
evaluator = Evaluator(model=None, metrics=[gruen])
result = evaluator.evaluate()
```