
# TextPreprocessor

A class for preprocessing text data, including methods for tokenization, stopword removal, stemming, and lemmatization.

## Initialization

The `TextPreprocessor` class is initialized with a list of stopwords and instances of `PorterStemmer` and `WordNetLemmatizer`.

```python
class TextPreprocessor:
    def __init__(self, stopwords: List[str] = stopwords):
        """
        Initializes the TextPreprocessor with:
        - A set of English stopwords.
        - A Porter Stemmer instance.
        - A WordNet Lemmatizer instance.

        Parameters:
        stopwords (List[str]): A list of stopwords to use for text preprocessing.
        """

        nltk_download()
        self.stop_words = stopwords
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

```
## Parameters Explanation

- **stopwords**: A list of stopwords to use for text preprocessing.

## Usage Example

Here is an example of how to use the `TextPreprocessor` class:

```python
from indoxJudge.utils import TextPreprocessor
# Define a sample text"
text = "The quick brown fox jumps over the lazy dog 123."

# Initialize the TextPreprocessor object
preprocessor = TextPreprocessor()

# Preprocess the text
processed_text = preprocessor.preprocess_text(
    text,
    to_lower=True,
    keep_alpha_numeric=True,
    remove_number=True,
    remove_stopword=True,
    stem_word=False,
    lemmatize_word=True,
    top_n_stopwords=5
)

print("Processed Text:", processed_text)
```