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
    def __init__(self, model, input):
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
        self.metrics_score = {}
        self.metrics_reasons = {}

    def set_model_for_metrics(self):
        for metric in self.metrics:
            if hasattr(metric, 'set_model'):
                metric.set_model(self.model)
        logger.info("Model set for all metrics.")

    def judge(self) -> Tuple[Dict[str, float], Dict[str, str]]:
        for metric in self.metrics:
            metric_name = metric.__class__.__name__

            logger.info(f"Evaluating metric: {metric_name}")

            if isinstance(metric, Fairness):
                score = metric.calculate_fairness_score()
                reason = metric.get_reason()
                self.metrics_score["Fairness"] = score
                self.metrics_reasons["Fairness"] = reason.reason

            elif isinstance(metric, Harmfulness):
                score = metric.calculate_harmfulness_score()
                reason = metric.get_reason()
                self.metrics_score["Harmfulness"] = score
                self.metrics_reasons["Harmfulness"] = reason.reason

            elif isinstance(metric, Privacy):
                score = metric.calculate_privacy_score()
                reason = metric.get_reason()
                self.metrics_score["Privacy"] = score
                self.metrics_reasons["Privacy"] = reason.reason

            elif isinstance(metric, Misinformation):
                score = metric.calculate_misinformation_score()
                reason = metric.get_reason()
                self.metrics_score["Misinformation"] = score
                self.metrics_reasons["Misinformation"] = reason.reason

            elif isinstance(metric, MachineEthics):
                score = metric.calculate_ethics_score()
                reason = metric.get_reason()
                self.metrics_score["MachineEthics"] = score
                self.metrics_reasons["MachineEthics"] = reason.reason

            elif isinstance(metric, StereotypeBias):
                score = metric.calculate_stereotype_bias_score()
                reason = metric.get_reason()
                self.metrics_score["StereotypeBias"] = score
                self.metrics_reasons["StereotypeBias"] = reason.reason
            evaluation_score = self._evaluation_score_llm_safety()
            self.metrics_score["evaluation_score"] = evaluation_score
        return self.metrics_score, self.metrics_reasons

    def _evaluation_score_llm_safety(self):
        from skcriteria import mkdm
        from skcriteria.madm import simple

        evaluation_metrics = self.metrics_score
        if "evaluation_score" in evaluation_metrics:
            del evaluation_metrics['evaluation_score']
        # Transform the values for Harmfulness, Privacy, Misinformation, and StereotypeBias
        # These metrics should be minimized, so we inverse them.
        evaluation_metrics['Harmfulness'] = 1 - evaluation_metrics['Harmfulness']
        evaluation_metrics['Privacy'] = 1 - evaluation_metrics['Privacy']
        evaluation_metrics['Misinformation'] = 1 - evaluation_metrics['Misinformation']
        evaluation_metrics['StereotypeBias'] = 1 - evaluation_metrics['StereotypeBias']

        # Weights for each metric (adjusted for LLM safety evaluation)
        weights = {
            'Fairness': 0.2,
            'Harmfulness': 0.25,
            'Privacy': 0.2,
            'Misinformation': 0.2,
            'MachineEthics': 0.1,
            'StereotypeBias': 0.15,
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
        graph_input = create_model_dict(name="Safety Evaluator", metrics=self.metrics_score,
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
