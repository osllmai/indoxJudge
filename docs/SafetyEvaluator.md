# Safety Evaluation System

## Overview

This code defines a system for evaluating various safety-related metrics of a given input sentence using a language model. The system assesses metrics such as fairness, harmfulness, privacy, misinformation, machine ethics, and stereotype bias, then calculates an overall evaluation score. Additionally, it generates visualizations for the metric scores. The system is composed of several classes and methods, and it integrates with a model to produce responses based on predefined templates.

## Classes

### SafetyEvaluator

A class that evaluates the safety of an input sentence across multiple metrics.

#### Attributes

- **model**: The language model used for generating responses.
- **metrics**: A list of metric instances (e.g., Fairness, Harmfulness, Privacy, etc.) initialized with the input sentence.
- **evaluation_score**: The cumulative evaluation score across all metrics.
- **metrics_score**: A dictionary containing individual scores for each metric.
- **metrics_reasons**: A dictionary containing reasons for each metric's score.

#### Methods

- **__init__(self, model, input: str)**: Initializes the SafetyEvaluator class with a model and an input sentence.
  ```python
  def __init__(self, model, input: str):
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
  ```

- **set_model_for_metrics(self)**: Sets the model for all metrics that support this functionality.
  ```python
  def set_model_for_metrics(self):
      for metric in self.metrics:
          if hasattr(metric, 'set_model'):
              metric.set_model(self.model)
      logger.info("Model set for all metrics.")
  ```

- **judge(self) -> Tuple[Dict[str, float], Dict[str, str]]**: Evaluates the input sentence using each metric, calculates scores, and returns dictionaries of scores and reasons.
  ```python
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
  ```

- **plot(self)**: Generates and returns visualizations of the metrics scores using the `MetricsVisualizer` class.
  ```python
  def plot(self):
      visualizer = MetricsVisualizer(metrics=self.metrics_score, score=self.evaluation_score)
      return visualizer.plot()
  ```

- **transform_metrics(self) -> List[Dict[str, float]]**: Transforms the metrics scores into a list of dictionaries with the average score.
  ```python
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
  ```

## Integration

This system integrates various metric classes to evaluate different aspects of safety. Each metric class (e.g., `Fairness`, `Harmfulness`) should implement specific methods to calculate its respective score and provide reasons for the score. The `SafetyEvaluator` class manages the evaluation process, integrates results from all metrics, and provides visualization through the `MetricsVisualizer` class.
