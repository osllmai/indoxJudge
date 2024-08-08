from collections import Counter
from typing import List, Union, Tuple
from indoxJudge.utils import TextPreprocessor


class METEOR:
    def __init__(self, llm_response: str, retrieval_context: Union[str, List[str]]):
        """
        Initialize the METEOR evaluator.

        Parameters:
        llm_response (str): The response generated by the Language Model.
        retrieval_context (Union[str, List[str]]): The reference context(s) against which the response is evaluated.
        """
        self.llm_response = llm_response
        self.retrieval_context = retrieval_context

    def measure(self) -> float:
        """
        Compute the METEOR score between the actual response and the reference context(s).

        Returns:
        float: The computed METEOR score.
        """
        self.score = self._calculate_score(
            llm_answer=self.llm_response, context=self.retrieval_context
        )
        return self.score

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess the given text by applying various text preprocessing methods.

        Parameters:
        text (str): The text to preprocess.

        Returns:
        str: The preprocessed text.
        """
        preprocessor = TextPreprocessor()
        preprocessing_methods = [
            preprocessor.to_lower,
            preprocessor.keep_alpha_numeric,
            preprocessor.remove_number,
            preprocessor.remove_stopword,
            preprocessor.lemmatize_word,
        ]
        preprocessed_text = preprocessor.preprocess_text(text, preprocessing_methods)
        return preprocessed_text

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the given text into a list of words.

        Parameters:
        text (str): The text to tokenize.

        Returns:
        List[str]: The list of tokens.
        """
        return text.split()

    def precision_recall(self, candidate: str, reference: str) -> Tuple[float, float]:
        """
        Compute the precision and recall between the candidate and reference texts.

        Parameters:
        candidate (str): The candidate text generated by the language model.
        reference (str): The reference text to compare against.

        Returns:
        Tuple[float, float]: The precision and recall scores.
        """
        candidate_tokens = self.tokenize(self.preprocess_text(candidate))
        reference_tokens = self.tokenize(self.preprocess_text(reference))

        candidate_counts = Counter(candidate_tokens)
        reference_counts = Counter(reference_tokens)

        matches = sum((candidate_counts & reference_counts).values())
        precision = matches / len(candidate_tokens) if candidate_tokens else 0
        recall = matches / len(reference_tokens) if reference_tokens else 0

        return precision, recall

    def fragmentation_penalty(self, candidate: str, reference: str) -> float:
        """
        Compute the fragmentation penalty for the candidate and reference texts.

        Parameters:
        candidate (str): The candidate text generated by the language model.
        reference (str): The reference text to compare against.

        Returns:
        float: The fragmentation penalty score.
        """
        candidate_tokens = self.tokenize(self.preprocess_text(candidate))
        reference_tokens = self.tokenize(self.preprocess_text(reference))

        if not candidate_tokens or not reference_tokens:
            return 0

        matches = [1 if token in reference_tokens else 0 for token in candidate_tokens]

        chunks = 1
        for i in range(1, len(matches)):
            if matches[i] == 1 and matches[i - 1] == 0:
                chunks += 1

        if matches[0] == 0:
            chunks -= 1

        return 0.5 * (chunks / sum(matches)) if sum(matches) else 0

    def meteor_score(self, candidate: str, reference: str) -> float:
        """
        Compute the METEOR score for the candidate and reference texts.

        Parameters:
        candidate (str): The candidate text generated by the language model.
        reference (str): The reference text to compare against.

        Returns:
        float: The METEOR score.
        """
        precision, recall = self.precision_recall(candidate, reference)

        if precision + recall == 0:
            f_mean = 0
        else:
            f_mean = (10 * precision) / (recall + (9 * precision))

        penalty = self.fragmentation_penalty(candidate, reference)
        score = (1 - penalty) * f_mean
        return score

    def _calculate_score(
        self, context: Union[str, List[str]], llm_answer: str
    ) -> float:
        """
        Calculate the METEOR score for the given context(s) and language model response.

        Parameters:
        context (Union[str, List[str]]): The reference context(s).
        llm_answer (str): The language model's response.

        Returns:
        float: The average METEOR score.
        """
        if isinstance(context, str):
            context = [context]

        if isinstance(llm_answer, list):
            llm_answer = " ".join(llm_answer)

        scores = []
        for ctx in context:
            scores.append(self.meteor_score(llm_answer, ctx))
        average_score = sum(scores) / len(scores) if scores else 0
        return round(average_score, 2)
