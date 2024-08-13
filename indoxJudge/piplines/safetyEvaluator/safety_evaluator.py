import sys
from typing import Tuple, Dict, List

from loguru import logger

from indoxJudge.metrics import (Fairness, Harmfulness, Privacy, Misinformation, MachineEthics, StereotypeBias)

# Set up logging
logger.remove()  # Remove the default logger
logger.add(sys.stdout,
           format="<green>{level}</green>: <level>{message}</level>",
           level="INFO")
logger.add(sys.stdout,
           format="<red>{level}</red>: <level>{message}</level>",
           level="ERROR")


class SafetyEvaluator:
    """
    A class to evaluate the safety of a model based on several metrics.

    Attributes
    ----------
    model : object
        The model to be evaluated.
    metrics : list
        A list of metric instances used for evaluation.
    evaluation_score : float
        The cumulative score of all metrics.
    metrics_score : dict
        A dictionary storing the score for each metric.
    metrics_reasons : dict
        A dictionary storing the reason for the score of each metric.

    Methods
    -------
    set_model_for_metrics():
        Sets the model for all metrics.
    judge() -> Tuple[Dict[str, float], Dict[str, str]]:
        Evaluates all metrics and returns their scores and reasons.
    plot(mode="external"):
        Plots the evaluation results.
    """

    def __init__(self, model, input):
        """
        Initializes the SafetyEvaluator with the given model and input sentence.

        Parameters
        ----------
        model : object
            The model to be evaluated.
        input : str
            The input sentence to evaluate the model on.
        """
        try:
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

        except FileNotFoundError as e:
            logger.error(f"Input file not found: {e}")
            raise

        except Exception as e:
            logger.error(f"An error occurred during initialization: {e}")
            raise

    def set_model_for_metrics(self):
        """
        Sets the model for all metrics.

        This method iterates over all metrics and sets the model for each metric that has the `set_model` method.
        """
        try:
            for metric in self.metrics:
                if hasattr(metric, 'set_model'):
                    metric.set_model(self.model)
            logger.info("Model set for all metrics.")

        except Exception as e:
            logger.error(f"An error occurred while setting the model for metrics: {e}")
            raise

    def judge(self) -> Tuple[Dict[str, float], Dict[str, str]]:
        """
        Evaluates all metrics and returns their scores and reasons.

        This method calculates the score and reason for each metric and aggregates them.

        Returns
        -------
        Tuple[Dict[str, float], Dict[str, str]]
            A tuple containing a dictionary of metric scores and a dictionary of reasons for each score.
        """
        try:
            for metric in self.metrics:
                metric_name = metric.__class__.__name__

                logger.info(f"Evaluating metric: {metric_name}")

                if isinstance(metric, Fairness):
                    score = metric.calculate_fairness_score()
                    reason = metric.get_reason()
                    self.evaluation_score += score
                    self.metrics_score["Fairness"] = score
                    self.metrics_reasons["Fairness"] = reason.reason

                elif isinstance(metric, Harmfulness):
                    score = metric.calculate_harmfulness_score()
                    reason = metric.get_reason()
                    self.evaluation_score += score
                    self.metrics_score["Harmfulness"] = score
                    self.metrics_reasons["Harmfulness"] = reason.reason

                elif isinstance(metric, Privacy):
                    score = metric.calculate_privacy_score()
                    reason = metric.get_reason()
                    self.evaluation_score += score
                    self.metrics_score["Privacy"] = score
                    self.metrics_reasons["Privacy"] = reason.reason

                elif isinstance(metric, Misinformation):
                    score = metric.calculate_misinformation_score()
                    reason = metric.get_reason()
                    self.evaluation_score += score
                    self.metrics_score["Misinformation"] = score
                    self.metrics_reasons["Misinformation"] = reason.reason

                elif isinstance(metric, MachineEthics):
                    score = metric.calculate_ethics_score()
                    reason = metric.get_reason()
                    self.evaluation_score += score
                    self.metrics_score["MachineEthics"] = score
                    self.metrics_reasons["MachineEthics"] = reason.reason

                elif isinstance(metric, StereotypeBias):
                    score = metric.calculate_stereotype_bias_score()
                    reason = metric.get_reason()
                    self.evaluation_score += score
                    self.metrics_score["StereotypeBias"] = score
                    self.metrics_reasons["StereotypeBias"] = reason.reason

            return self.metrics_score, self.metrics_reasons

        except Exception as e:
            logger.error(f"An error occurred during evaluation: {e}")
            raise

    def plot(self, mode="external"):
        """
        Plots the evaluation results.

        Parameters
        ----------
        mode : str, optional
            The mode for plotting, by default "external".

        Returns
        -------
        object
            A visualization object representing the evaluation results.
        """
        try:
            from indoxJudge.graph import Visualization
            from indoxJudge.utils import create_model_dict

            graph_input = create_model_dict(name="RAG Evaluator", metrics=self.metrics_score,
                                            score=self.evaluation_score / 6)
            visualizer = Visualization(data=graph_input, mode="safety")
            return visualizer.plot(mode=mode)

        except Exception as e:
            logger.error(f"An error occurred during plotting: {e}")
            raise
    # def transform_metrics(self) -> List[Dict[str, float]]:
    #     average_score = sum(self.metrics_score.values()) / len(self.metrics_score)
    #     average_score = int(average_score * 100) / 100.0
    #
    #     model = {
    #         'name': "Indox_API",
    #         'score': average_score,
    #         'metrics': self.metrics_score
    #     }
    #
    #     models = [model]
    #
    #     return models
