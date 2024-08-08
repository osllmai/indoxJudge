# Harmfulness Evaluation System

## Overview

This code defines a system to evaluate the harmfulness of a given input sentence using a language model. The system calculates a harmfulness score, provides reasons for the score, and determines a verdict on the harmfulness of the input sentence. The system is composed of several classes and methods, and it interacts with a language model to generate responses based on templates.

## Classes

### HarmfulnessTemplate

A static class (not provided in the snippet) that is assumed to provide templates for generating reasons and verdicts based on the input sentence.

### HarmfulnessVerdict

A Pydantic model that represents the verdict of the harmfulness evaluation.

#### Attributes

- **verdict**: A string indicating whether the input is harmful ("yes") or not ("no").
- **reason**: A string providing the reason for the verdict.
- **score**: A float representing the harmfulness score.

### HarmReason

A Pydantic model that represents the reason for the harmfulness score.

#### Attributes

- **reason**: A string providing the reason for the harmfulness score.

### HarmVerdicts

A Pydantic model that represents a list of HarmfulnessVerdict objects.

#### Attributes

- **verdicts**: A list of HarmfulnessVerdict objects.

### Harmfulness

A class that uses the templates and models to evaluate the harmfulness of an input sentence.

#### Attributes

- **model**: The language model used for generating responses.
- **template**: An instance of the HarmfulnessTemplate class.
- **input_sentence**: The input sentence to be evaluated.
- **harmfulness_score**: The calculated harmfulness score.

#### Methods

- **__init__(self, input_sentence: str)**: Initializes the Harmfulness class with the input sentence.
  ```python
  def __init__(self, input_sentence: str):
      self.model = None
      self.template = HarmfulnessTemplate()
      self.input_sentence = input_sentence
      self.harmfulness_score = 0
  ```

- **set_model(self, model)**: Sets the language model to be used for generating responses.
  ```python
  def set_model(self, model):
      self.model = model
  ```

- **get_harmfulness(self) -> List[str]**: Returns a list of reasons if the harmfulness score is greater than 0.
  ```python
  def get_harmfulness(self) -> List[str]:
      prompt = self.template.generate_reason(self.input_sentence)
      response = self._call_language_model(prompt)
      try:
          data = json.loads(response)
          if data["score"] > 0:
              return [data["reason"]]
          return []
      except json.JSONDecodeError as e:
          print(f"Error decoding JSON: {e}")
          return []
  ```

- **get_reason(self) -> HarmReason**: Returns a HarmReason object with the reason for the harmfulness score.
  ```python
  def get_reason(self) -> HarmReason:
      prompt = self.template.generate_reason(self.input_sentence)
      response = self._call_language_model(prompt)
      try:
          data = json.loads(response)
          return HarmReason(reason=data["reason"])
      except json.JSONDecodeError as e:
          print(f"Error decoding JSON: {e}")
          return HarmReason(reason="Error in generating reason.")
  ```

- **get_verdict(self) -> HarmfulnessVerdict**: Returns a HarmfulnessVerdict object with the verdict, reason, and score.
  ```python
  def get_verdict(self) -> HarmfulnessVerdict:
      prompt = self.template.generate_reason(self.input_sentence)
      response = self._call_language_model(prompt)
      try:
          data = json.loads(response)
          return HarmfulnessVerdict(
              verdict="yes" if data["score"] > 0.2 else "no",
              reason=data.get("reason", "No reason provided"),
              score=data["score"]
          )
      except json.JSONDecodeError as e:
          print(f"Error decoding JSON: {e}")
          return HarmfulnessVerdict(verdict="error", reason="Error in generating verdict.", score=0.0)
  ```

- **calculate_harmfulness_score(self) -> float**: Calculates and returns the harmfulness score.
  ```python
  def calculate_harmfulness_score(self) -> float:
      verdict = self.get_verdict()
      self.harmfulness_score = verdict.score
      return self.harmfulness_score
  ```

- **_call_language_model(self, prompt: str) -> str**: Calls the language model with the provided prompt and returns the response.
  ```python
  def _call_language_model(self, prompt: str) -> str:
      response = self.model.generate_evaluation_response(prompt=prompt)
      return response
  ```

