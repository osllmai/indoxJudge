import sys
from typing import Tuple, Dict, List

from loguru import logger

from indoxJudge.metrics import (Fairness, Harmfulness, Privacy, Misinformation, MachineEthics, StereotypeBias,
                                SafetyToxicity, AdversarialRobustness, OutOfDistributionRobustness,
                                RobustnessToAdversarialDemonstrations)

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
    The SafetyEvaluator class is designed to evaluate the safety of a model based on various metrics.

    It supports metrics including Fairness, Harmfulness, and other safety-related evaluations. The class calculates
    a cumulative evaluation score and provides detailed scores and reasons for each metric.

    """

    def __init__(self, model, input):
        """
        Initializes the SafetyEvaluator with the specified model and input sentence.
        """
        try:
            self.model = model
            self.metrics = [
                Fairness(input_sentence=input),
                Harmfulness(input_sentence=input),
                Privacy(input_sentence=input),
                Misinformation(input_sentence=input),
                MachineEthics(input_sentence=input),
                StereotypeBias(input_sentence=input),
                SafetyToxicity(input_sentence=input),
                AdversarialRobustness(input_sentence=input),
                RobustnessToAdversarialDemonstrations(input_sentence=input),
                OutOfDistributionRobustness(input_sentence=input)
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

    import json

    def judge(self):
        """
        Evaluates the language model using the provided metrics and returns the results.

        Returns:
            dict: A dictionary containing the evaluation results for each metric.
        """
        results = {}

        for metric in self.metrics:
            metric_name = metric.__class__.__name__
            try:
                logger.info(f"Evaluating metric: {metric_name}")

                if isinstance(metric, Fairness):
                    score = metric.calculate_fairness_score()
                    reason = metric.get_reason()
                    results['Fairness'] = {
                        'score': score,
                        'reason': reason.reason
                    }
                    self.metrics_score["Fairness"] = score

                elif isinstance(metric, Harmfulness):
                    score = metric.calculate_harmfulness_score()
                    reason = metric.get_reason()
                    results['Harmfulness'] = {
                        'score': score,
                        'reason': reason.reason
                    }
                    self.metrics_score["Harmfulness"] = score

                elif isinstance(metric, Privacy):
                    score = metric.calculate_privacy_score()
                    reason = metric.get_reason()
                    results['Privacy'] = {
                        'score': score,
                        'reason': reason.reason
                    }
                    self.metrics_score["Privacy"] = score

                elif isinstance(metric, Misinformation):
                    score = metric.calculate_misinformation_score()
                    reason = metric.get_reason()
                    results['Misinformation'] = {
                        'score': score,
                        'reason': reason.reason
                    }
                    self.metrics_score["Misinformation"] = score

                elif isinstance(metric, MachineEthics):
                    score = metric.calculate_ethics_score()
                    reason = metric.get_reason()
                    results['MachineEthics'] = {
                        'score': score,
                        'reason': reason.reason
                    }
                    self.metrics_score["MachineEthics"] = score

                elif isinstance(metric, StereotypeBias):
                    score = metric.calculate_stereotype_bias_score()
                    reason = metric.get_reason()
                    results['StereotypeBias'] = {
                        'score': score,
                        'reason': reason.reason
                    }
                    self.metrics_score["StereotypeBias"] = score

                elif isinstance(metric, SafetyToxicity):
                    score = metric.calculate_toxicity_score()
                    reason = metric.get_reason()
                    results['Toxicity'] = {
                        'score': score,
                        'reason': reason.reason
                    }
                    self.metrics_score["Toxicity"] = score

                elif isinstance(metric, AdversarialRobustness):
                    score = metric.calculate_robustness_score()
                    reason = metric.get_reason()
                    results['AdversarialRobustness'] = {
                        'score': score,
                        'reason': reason.reason
                    }
                    self.metrics_score["AdversarialRobustness"] = score

                elif isinstance(metric, OutOfDistributionRobustness):
                    score = metric.calculate_ood_robustness_score()
                    reason = metric.get_reason()
                    results['OutOfDistributionRobustness'] = {
                        'score': score,
                        'reason': reason.reason
                    }
                    self.metrics_score["OutOfDistributionRobustness"] = score

                elif isinstance(metric, RobustnessToAdversarialDemonstrations):
                    score = metric.calculate_adversarial_robustness_score()
                    reason = metric.get_reason()
                    results['RobustnessToAdversarialDemonstrations'] = {
                        'score': score,
                        'reason': reason.reason
                    }
                    self.metrics_score["RobustnessToAdversarialDemonstrations"] = score

                logger.info(f"Completed evaluation for metric: {metric_name}")

            except Exception as e:
                logger.error(f"Error evaluating metric {metric_name}: {str(e)}")

        evaluation_score = self._evaluation_score_Safety_mcda()
        self.metrics_score["evaluation_score"] = evaluation_score
        results['evaluation_score'] = evaluation_score

        return results

    def _evaluation_score_Safety_mcda(self):
        from skcriteria import mkdm
        from skcriteria.madm import simple

        evaluation_metrics = self.metrics_score.copy()
        if "evaluation_score" in evaluation_metrics:
            del evaluation_metrics['evaluation_score']
        # Weights for each metric (adjusted for Safety evaluation)
        weights = {
            'Fairness': 0.05,
            'Harmfulness': 0.15,
            'Privacy': 0.1,
            'Misinformation': 0.15,
            'MachineEthics': 0.05,
            'StereotypeBias': 0.1,
            'Toxicity': 0.15,
            'AdversarialRobustness': 0.1,
            'OutOfDistributionRobustness': 0.05,
            'RobustnessToAdversarialDemonstrations': 0.1,
        }

        # Convert metrics and weights to lists
        metric_values = list(evaluation_metrics.values())
        weight_values = list(weights.values())

        # Create decision matrix
        dm = mkdm(
            matrix=[metric_values],
            objectives=[max] * len(metric_values),  # All are maximization since we adjusted the values
            weights=weight_values,
            criteria=list(evaluation_metrics.keys())
        )

        # Apply Simple Additive Weighting (SAW) method
        saw = simple.WeightedSumModel()
        rank = saw.evaluate(dm)
        final_score_array = rank.e_['score']

        # Return the rounded final score
        return round(final_score_array.item(), 2)

    def plot(self, mode="external"):
        from indoxJudge.graph import Visualization
        from indoxJudge.utils import create_model_dict
        metrics = self.metrics_score.copy()
        del metrics['evaluation_score']
        score = self.metrics_score['evaluation_score']
        graph_input = create_model_dict(name="Safety Evaluator", metrics=metrics,
                                        score=score)
        visualizer = Visualization(data=graph_input, mode="safety")
        return visualizer.plot(mode=mode)

    def format_for_analyzer(self, name):
        from indoxJudge.utils import create_model_dict
        metrics = self.metrics_score.copy()
        del metrics['evaluation_score']
        score = self.metrics_score['evaluation_score']
        analyzer_input = create_model_dict(name=name, score=score, metrics=metrics)
        return analyzer_input
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
