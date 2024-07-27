from typing import List, Optional
from pydantic import BaseModel, Field
import json

from .template import AnswerRelevancyTemplate


class Statements(BaseModel):
    """
    Model representing a list of statements extracted from the LLM response.
    """
    statements: List[str]


class AnswerRelevancyVerdict(BaseModel):
    """
    Model representing a verdict on the relevancy of an answer,
    including the verdict itself and the reasoning behind it.
    """
    verdict: str
    reason: str = Field(default=None)


class Verdicts(BaseModel):
    """
    Model representing a list of AnswerRelevancyVerdict instances.
    """
    verdicts: List[AnswerRelevancyVerdict]


class Reason(BaseModel):
    """
    Model representing the reason provided for any irrelevant statements found in the response.
    """
    reason: str


class AnswerRelevancy:
    """
    Class for evaluating the relevancy of language model outputs by analyzing statements,
    generating verdicts, and calculating relevancy scores.
    """
    def __init__(self, query: str, llm_response: str, model=None, threshold: float = 0.5, include_reason: bool = True,
                 strict_mode: bool = False):
        """
        Initializes the AnswerRelevancy class with the query, LLM response, and evaluation settings.

        :param query: The query being evaluated.
        :param llm_response: The response generated by the language model.
        :param model: The language model to use for evaluation. Defaults to None.
        :param threshold: The threshold for determining relevancy. Defaults to 0.5.
        :param include_reason: Whether to include reasoning for the relevancy verdicts. Defaults to True.
        :param strict_mode: Whether to use strict mode, which forces a score of 0 if relevancy is below the threshold. Defaults to False.
        """
        self.model = model
        self.query = query
        self.llm_response = llm_response
        self.threshold = 1 if strict_mode else threshold
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.evaluation_cost = None
        self.statements = []
        self.verdicts = []
        self.reason = None
        self.score = 0
        self.success = False

    def set_model(self, model):
        """
        Sets the language model to be used for evaluation.

        :param model: The language model to use.
        """
        self.model = model

    def measure(self) -> float:
        """
        Measures the relevancy of the LLM response by generating statements, verdicts, and reasons,
        then calculating the relevancy score.

        :return: The calculated relevancy score.
        """
        self.statements = self._generate_statements()
        self.verdicts = self._generate_verdicts()
        self.score = self._calculate_score()
        self.reason = self._generate_reason(self.query)
        self.success = self.score >= self.threshold

        return self.score

    def _generate_statements(self) -> List[str]:
        """
        Generates a list of statements from the LLM response using a prompt template.

        :return: A list of statements.
        """
        prompt = AnswerRelevancyTemplate.generate_statements(llm_response=self.llm_response)
        response = self._call_language_model(prompt)
        data = json.loads(response)
        return data["statements"]

    def _generate_verdicts(self) -> List[AnswerRelevancyVerdict]:
        """
        Generates a list of verdicts on the relevancy of the statements.

        :return: A list of AnswerRelevancyVerdict instances.
        """
        prompt = AnswerRelevancyTemplate.generate_verdicts(query=self.query, llm_response=self.statements)
        response = self._call_language_model(prompt)
        data = json.loads(response)
        return [AnswerRelevancyVerdict(**item) for item in data["verdicts"]]

    def _generate_reason(self, query: str) -> Optional[str]:
        """
        Generates the reasoning behind the relevancy score if include_reason is set to True.

        :param query: The query being evaluated.
        :return: A string containing the reasoning or None if not included.
        """
        if not self.include_reason:
            return None

        irrelevant_statements = [verdict.reason for verdict in self.verdicts if verdict.verdict.strip().lower() == "no"]

        prompt = AnswerRelevancyTemplate.generate_reason(
            irrelevant_statements=irrelevant_statements,
            query=query,
            score=format(self.score, ".2f"),
        )

        response = self._call_language_model(prompt)
        data = json.loads(response)
        return data["reason"]

    def _calculate_score(self) -> float:
        """
        Calculates the relevancy score based on the number of relevant verdicts.

        :return: The calculated relevancy score.
        """
        number_of_verdicts = len(self.verdicts)
        if number_of_verdicts == 0:
            return 1

        relevant_count = sum(1 for verdict in self.verdicts if verdict.verdict.strip().lower() != "no")

        score = relevant_count / number_of_verdicts
        return 0 if self.strict_mode and score < self.threshold else score

    def _call_language_model(self, prompt: str) -> str:
        """
        Calls the language model with the given prompt and returns the response.

        :param prompt: The prompt to provide to the language model.
        :return: The response from the language model.
        """
        response = self.model.generate_evaluation_response(prompt=prompt)
        return response