import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import Union, List, Dict


class BertScore:
    def __init__(
        self,
        llm_response: str,
        retrieval_context: Union[str, List[str]],
        model_name: str = "roberta-base",
        max_length: int = 1024,
    ):
        """
        Initialize the BertScore class to evaluate the similarity between a generated response
        and one or more expected responses using a specified pre-trained transformer model.

        Parameters:
        llm_response (str): The response generated by a language model.
        retrieval_context (Union[str, List[str]]): The expected response(s) to compare against the actual response.
        model_name (str): The identifier for the pre-trained model to be used for generating embeddings.
                          Defaults to "roberta-base".
        max_length (int): The maximum length of input sequences to be processed by the model. Defaults to 1024.
        """
        self.llm_response = llm_response
        self.retrieval_context = retrieval_context
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length

    def measure(self) -> float:
        """
        Compute the similarity score between the actual response and the expected response(s).

        Returns:
        float: The computed similarity score.
        """
        self.score = self._calculate_scores(
            llm_answer=self.llm_response, context=self.retrieval_context
        )
        return self.score

    def get_embeddings(self, text: str) -> torch.Tensor:
        """
        Generate embeddings for the given text using the pre-trained model.

        Parameters:
        text (str): The text to generate embeddings for.

        Returns:
        torch.Tensor: The generated embeddings.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.squeeze(0)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute the cosine similarity between two numpy arrays.

        Parameters:
        a (np.ndarray): The first array.
        b (np.ndarray): The second array.

        Returns:
        float: The cosine similarity between the two arrays.
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _calculate_scores(
        self, llm_answer: Union[str, List[str]], context: Union[str, List[str]]
    ) -> Dict[str, float]:
        """
        Calculate the precision, recall, and F1-score between the actual response and the expected response(s).

        Parameters:
        llm_answer (Union[str, List[str]]): The actual response generated by a language model.
        context (Union[str, List[str]]): The expected response(s) to compare against the actual response.

        Returns:
        Dict[str, float]: A dictionary containing the precision, recall, and F1-score.
        """
        if not isinstance(llm_answer, str):
            llm_answer = " ".join(llm_answer)
        llm_answer_embeddings = self.get_embeddings(llm_answer)

        if isinstance(context, str):
            context = [context]

        precisions = []
        recalls = []
        f1_scores = []
        for ctx in context:
            context_embeddings = self.get_embeddings(ctx)

            similarities = np.zeros(
                (len(llm_answer_embeddings), len(context_embeddings))
            )
            for i, a_emb in enumerate(llm_answer_embeddings):
                for j, c_emb in enumerate(context_embeddings):
                    similarities[i, j] = self.cosine_similarity(
                        a_emb.numpy(), c_emb.numpy()
                    )

            precision = np.mean(np.max(similarities, axis=1))
            recall = np.mean(np.max(similarities, axis=0))
            f1_score = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1_score)

        average_precision = np.mean(precisions) if precisions else 0
        average_recall = np.mean(recalls) if recalls else 0
        average_f1_score = np.mean(f1_scores) if f1_scores else 0

        scores = {
            "Precision": average_precision,
            "Recall": average_recall,
            "F1-score": average_f1_score,
        }

        return scores
