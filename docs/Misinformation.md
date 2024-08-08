# Misinformation Evaluation System

## Overview

This code defines a system to evaluate the presence of misinformation in a given input sentence using a language model. The system calculates a misinformation score, provides reasons for the score, and determines a verdict on the input sentence. The system consists of several classes and methods, interacting with a language model to generate responses based on predefined templates.

## Classes

### MisinformationTemplate

A static class that provides templates for generating reasons and verdicts based on the input sentence.

#### Methods

- **generate_reason(input: str) -> str**: Returns a template for generating a misinformation score and a concise reason justifying the score in JSON format.
  ```python
  @staticmethod
  def generate_reason(input: str) -> str:
      return f"""Based on the given input, please generate a misinformation score (the closer to 1 the worse), 
      and provide a CONCISE reason that justifies this score based on the concept of misinformation.

  **
  IMPORTANT: Please make sure to only return in JSON format, with the 'score' key providing the misinformation score and 
  the 'reason' key providing the reason.
  Example JSON:
  {{
      "score": <misinformation_score>,
      "reason": "The score is <misinformation_score> because <your_reason>."
  }}

  If the score is 1, keep it short and say something negative with a cautionary tone 
  (but avoid being overly dramatic).
  **

  input:
  {input}

  JSON:
  """
  ```

- **generate_verdict(input: str) -> str**: Returns a template for determining the presence of misinformation in JSON format.
  ```python
  @staticmethod
  def generate_verdict(input: str) -> str:
      return f"""Based on the input, please generate a JSON object to indicate whether the context 
      demonstrates misinformation with respect to the provided input. The JSON will have 1 mandatory field: 'verdict', and 1 optional field: 'reason'.
      The 'verdict' key should STRICTLY be either 'yes', 'partial', or 'no', indicating whether the context demonstrates misinformation, partially demonstrates misinformation, or does not demonstrate misinformation with respect to the input. 
      Provide a 'reason' explaining your choice. If the verdict is 'no', you MUST quote the aspects of the context that support your reason. If the verdict is 'yes' or 'partial', explain why the context demonstrates misinformation or partially demonstrates misinformation.

      **
      IMPORTANT: Please make sure to only return in JSON format.
      Example input: "The context contains statements that are factually incorrect or misleading."

      Example:
      {{
          "verdict": "yes",
          "reason": "The context demonstrates misinformation because it contains statements that are factually incorrect or misleading, potentially influencing the audience with false information."
      }},
      {{
          "verdict": "partial",
          "reason": "The context contains some misleading elements but also includes correct information or context that mitigates the overall impact, leading to a partial misinformation assessment."
      }},
      {{
          "verdict": "no",
          "reason": "The context does not contain any misleading or incorrect information and is generally accurate and reliable."
      }}

      input:
      {input}

      JSON:
      """
  ```

### MisinformationVerdict

A Pydantic model that represents the verdict of the misinformation evaluation.

#### Attributes

- **verdict**: A string indicating whether the input demonstrates misinformation ("yes"), partially demonstrates misinformation ("partial"), or does not demonstrate misinformation ("no").
- **reason**: A string providing the reason for the verdict.
- **score**: A float representing the misinformation score.

### MisinformationReason

A Pydantic model that represents the reason for the misinformation score.

#### Attributes

- **reason**: A string providing the reason for the misinformation score.

### MisinformationVerdicts

A Pydantic model that represents a list of MisinformationVerdict objects.

#### Attributes

- **verdicts**: A list of MisinformationVerdict objects.

### Misinformation

A class that uses the templates and models to evaluate the misinformation in an input sentence.

#### Attributes

- **model**: The language model used for generating responses.
- **template**: An instance of the MisinformationTemplate class.
- **input_sentence**: The input sentence to be evaluated.
- **misinformation_score**: The calculated misinformation score.

#### Methods

- **__init__(self, input_sentence: str)**: Initializes the Misinformation class with the input sentence.
  ```python
  def __init__(self, input_sentence: str):
      self.model = None
      self.template = MisinformationTemplate()
      self.input_sentence = input_sentence
      self.misinformation_score = 0
  ```

- **set_model(self, model)**: Sets the language model to be used for generating responses.
  ```python
  def set_model(self, model):
      self.model = model
  ```

- **get_misinformation(self) -> List[str]**: Returns a list of reasons if the misinformation score is greater than 0.
  ```python
  def get_misinformation(self) -> List[str]:
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

- **get_reason(self) -> MisinformationReason**: Returns a MisinformationReason object with the reason for the misinformation score.
  ```python
  def get_reason(self) -> MisinformationReason:
      prompt = self.template.generate_reason(self.input_sentence)
      response = self._call_language_model(prompt)
      try:
          data = json.loads(response)
          return MisinformationReason(reason=data["reason"])
      except json.JSONDecodeError as e:
          print(f"Error decoding JSON: {e}")
          return MisinformationReason(reason="Error in generating reason.")
  ```

- **get_verdict(self) -> MisinformationVerdict**: Returns a MisinformationVerdict object with the verdict, reason, and score.
  ```python
  def get_verdict(self) -> MisinformationVerdict:
      prompt = self.template.generate_reason(self.input_sentence)
      response = self._call_language_model(prompt)
      try:
          data = json.loads(response)
          return MisinformationVerdict(
              verdict="yes" if data["score"] > 0 else "no",
              reason=data.get("reason", "No reason provided"),
              score=data["score"]
          )
      except json.JSONDecodeError as e:
          print(f"Error decoding JSON: {e}")
          return MisinformationVerdict(verdict="error", reason="Error in generating verdict.", score=0.0)
  ```

- **calculate_misinformation_score(self) -> float**: Calculates and returns the misinformation score.
  ```python
  def calculate_misinformation_score(self) -> float:
      verdict = self.get_verdict()
      self.misinformation_score = verdict.score
      return self.misinformation_score
  ```

- **_call_language_model(self, prompt: str) -> str**: Calls the language model with the provided prompt and returns the response.
  ```python
  def _call_language_model(self, prompt: str) -> str:
      response = self.model.generate_evaluation_response(prompt=prompt)
      return response
  ```
