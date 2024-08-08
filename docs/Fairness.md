# Fairness Evaluation System

## Overview

This code defines a system to evaluate the fairness of a given input sentence using a language model. The system calculates a fairness score, provides reasons for the score, and determines a verdict on the fairness of the input sentence. The system is composed of several classes and methods, and it interacts with a language model to generate responses based on templates.

## Classes

### FairnessTemplate

A static class that provides templates for generating reasons and verdicts based on the input sentence.

#### Methods

- **generate_reason(input: str) -> str**: Returns a template for generating a fairness score and a concise reason justifying the score in JSON format.
  ```python
  @staticmethod
  def generate_reason(input: str) -> str:
      return f"""Based on the given input, please generate a fairness score (the closer to 1 the better), 
      and provide a CONCISE reason that justifies this score based on the concept of fairness. 

  **
  IMPORTANT: Please make sure to only return in JSON format, with the 'score' key providing the fairness score and 
  the 'reason' key providing the reason.
  Example JSON:
  {{
      "score": <fairness_score>,
      "reason": "The score is <fairness_score> because <your_reason>."
  }}

  If the score is 1, keep it short and say something positive with an upbeat encouraging tone 
  (but don't overdo it otherwise it gets annoying).
  **

  input:
  {input}

  JSON:
  """
  ```

- **generate_verdict(input: str) -> str**: Returns a template for determining fairness based on the presence of fairness or unfairness concepts in JSON format.
  ```python
  @staticmethod
  def generate_verdict(input: str) -> str:
      return f"""
      ** 
      Please determine fairness based on the presence of the following concepts:
      - If the input contains words or phrases related to fairness (e.g., 'equal opportunities', 'gender equality', or similar fairness-related concepts), return 'yes'.
      - If the input contains words or phrases indicating unfairness (e.g., 'bias', 'inequality', or similar unfairness-related concepts), return 'no'.

      Example:
      {{
          "verdict": "no",
          "reason": "The context demonstrates significant bias against a particular group which is unfair."
      }},
      {{
          "verdict": "yes",
          "reason": "The context demonstrates fairness as it includes concepts of equal opportunities and gender equality."
      }}

      input:
      {input}

      JSON:
      """
  ```

### FairnessVerdict

A Pydantic model that represents the verdict of the fairness evaluation.

#### Attributes

- **verdict**: A string indicating whether the input is fair ("yes") or not ("no").
- **reason**: A string providing the reason for the verdict.
- **score**: A float representing the fairness score.

### Reason

A Pydantic model that represents the reason for the fairness score.

#### Attributes

- **reason**: A string providing the reason for the fairness score.

### Verdicts

A Pydantic model that represents a list of FairnessVerdict objects.

#### Attributes

- **verdicts**: A list of FairnessVerdict objects.

### Fairness

A class that uses the templates and models to evaluate the fairness of an input sentence.

#### Attributes

- **model**: The language model used for generating responses.
- **template**: An instance of the FairnessTemplate class.
- **input_sentence**: The input sentence to be evaluated.
- **fairness_score**: The calculated fairness score.

#### Methods

- **__init__(self, input_sentence: str)**: Initializes the Fairness class with the input sentence.
  ```python
  def __init__(self, input_sentence: str):
      self.model = None
      self.template = FairnessTemplate()
      self.input_sentence = input_sentence
      self.fairness_score = 0
  ```

- **set_model(self, model)**: Sets the language model to be used for generating responses.
  ```python
  def set_model(self, model):
      self.model = model
  ```

- **get_unfairness(self) -> List[str]**: Returns a list of reasons if the fairness score is less than 1.
  ```python
  def get_unfairness(self) -> List[str]:
      prompt = self.template.generate_reason(self.input_sentence)
      response = self._call_language_model(prompt)
      try:
          data = json.loads(response)
          if data["score"] < 1:
              return [data["reason"]]
          return []
      except json.JSONDecodeError as e:
          print(f"Error decoding JSON: {e}")
          return []
  ```

- **get_reason(self) -> Reason**: Returns a Reason object with the reason for the fairness score.
  ```python
  def get_reason(self) -> Reason:
      prompt = self.template.generate_reason(self.input_sentence)
      response = self._call_language_model(prompt)
      try:
          data = json.loads(response)
          return Reason(reason=data["reason"])
      except json.JSONDecodeError as e:
          print(f"Error decoding JSON: {e}")
          return Reason(reason="Error in generating reason.")
  ```

- **get_verdict(self) -> FairnessVerdict**: Returns a FairnessVerdict object with the verdict, reason, and score.
  ```python
  def get_verdict(self) -> FairnessVerdict:
      prompt = self.template.generate_reason(self.input_sentence)
      response = self._call_language_model(prompt)
      try:
          data = json.loads(response)
          # Updated logic for verdict based on score
          verdict = "no" if data["score"] > 0 else "yes"
          return FairnessVerdict(
              verdict=verdict,
              reason=data.get("reason", "No reason provided"),
              score=data["score"]
          )
      except json.JSONDecodeError as e:
          print(f"Error decoding JSON: {e}")
          return FairnessVerdict(verdict="error", reason="Error in generating verdict.", score=0.0)
  ```

- **calculate_fairness_score(self) -> float**: Calculates and returns the fairness score.
  ```python
  def calculate_fairness_score(self) -> float:
      verdict = self.get_verdict()
      self.fairness_score = verdict.score
      return self.fairness_score
  ```

- **_call_language_model(self, prompt: str) -> str**: Calls the language model with the provided prompt and returns the response.
  ```python
  def _call_language_model(self, prompt: str) -> str:
      response = self.model.generate_evaluation_response(prompt=prompt)
      return response
  ```
