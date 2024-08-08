import sys
from loguru import logger

from indoxJudge.metrics import (Fairness, Harmfulness, Privacy, Misinformation, MachineEthics, StereotypeBias)
from indoxJudge.piplines.safetyEvaluator.graph.metrics_visualizer import MetricsVisualizer

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
        self.evaluation_score = 0
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

    def plot(self):
        visualizer = MetricsVisualizer(metrics=self.metrics_score, score=self.evaluation_score)
        return visualizer.plot()

    def transform_metrics(self) -> List[Dict[str, float]]:
        average_score = sum(self.metrics_score.values()) / len(self.metrics_score)
        average_score = int(average_score * 100) / 100.0

        model = {
            'name': "Indox_API",
            'score': average_score,
            'metrics': self.metrics_score
        }

        models = [model]

        return models
