<p align="center">


<div style="position: relative; width: 100%; text-align: center;">
    <h1>inDoxJudge</h1>
    <a href="https://github.com/osllmai/inDoxJudge">
<img src="https://readme-typing-svg.demolab.com?font=Georgia&size=16&duration=3000&pause=500&multiline=true&width=700&height=100&lines=InDoxJudge;LLM+Evaluation+%7C+RAG+Evaluation+%7C+Safety+Evaluation+%7C+LLM+Comparison;Copyright+©️+OSLLAM.ai" alt="Typing SVG" style="margin-top: 20px;"/>
    </a>
</div>

</br>

[![License](https://img.shields.io/github/license/osllmai/inDox)](https://github.com/osllmai/inDox/blob/main/LICENSE)
[![PyPI](https://badge.fury.io/py/indoxJudge.svg)](https://pypi.org/project/IndoxJudge/0.0.2/)
[![Python](https://img.shields.io/pypi/pyversions/indoxJudge.svg)](https://pypi.org/project/indoxJudge/0.0.2/)
[![Downloads](https://static.pepy.tech/badge/indoxJudge)](https://pepy.tech/project/indoxJudge)

[![Discord](https://img.shields.io/discord/1223867382460579961?label=Discord&logo=Discord&style=social)](https://discord.com/invite/ossllmai)
[![GitHub stars](https://img.shields.io/github/stars/osllmai/inDoxJudge?style=social)](https://github.com/osllmai/inDoxJudge)




<p align="center">
  <a href="https://osllm.ai">Official Website</a> &bull; <a href="https://github.com/osllmai/inDox/wiki">Documentation</a> &bull; <a href="https://discord.gg/qrCc56ZR">Discord</a>
</p>


<p align="center">
  <b>NEW:</b> <a href="https://docs.google.com/forms/d/1CQXJvxLUqLBSXnjqQmRpOyZqD6nrKubLz2WTcIJ37fU/prefill">Subscribe to our mailing list</a> for updates and news!
</p>


Welcome to IndoxJudge! This repository provides a comprehensive suite of evaluation metrics for assessing the performance and quality of large language models (LLMs). Whether you're a researcher, developer, or enthusiast, this toolkit offers essential tools to measure various aspects of LLMs, including knowledge retention, bias, toxicity, and more.



![IndoxJudge](docs\image\IndoxJudge Evaluate LLMs with metrics & ModelSafety.png)

## Overview

IndoxJudge is designed to provide a standardized and extensible framework for evaluating LLMs. With a focus on accuracy, fairness, and relevancy, this toolkit supports a wide range of evaluation metrics and is continuously updated to include the latest advancements in the field.

## Features

- **Comprehensive Metrics**: Evaluate LLMs across multiple dimensions, including accuracy, bias, toxicity, and contextual relevancy.
- **RAG Evaluation**: Includes specialized metrics for evaluating retrieval-augmented generation (RAG) models.
- **Safety Evaluation**: Assess the safety of model outputs, focusing on toxicity, bias, and ethical considerations.
- **Extensible Framework**: Easily integrate new metrics or customize existing ones to suit specific needs.
- **User-Friendly Interface**: Intuitive and easy-to-use interface for seamless evaluation.
- **Continuous Updates**: Regular updates to incorporate new metrics and improvements.

## Supported Models

IndoxJudge currently supports the following LLM models:

- **OpenAi**
- **GoogleAi**
- **IndoxApi**
- **HuggingFaceModel**
- **Mistral**
- **Pheonix**
- **Ollama**

## Metrics

IndoxJudge includes the following metrics, with more being added:

- **GEval**: General evaluation metric for LLMs.
- **KnowledgeRetention**: Assesses the ability of LLMs to retain factual information.
- **BertScore**: Measures the similarity between generated and reference sentences.
- **Toxicity**: Evaluates the presence of toxic content in model outputs.
- **Bias**: Analyzes the potential biases in LLM outputs.
- **Hallucination**: Identifies instances where the model generates false or misleading information.
- **Faithfulness**: Checks the alignment of generated content with source material.
- **ContextualRelevancy**: Assesses the relevance of responses in context.
- **Rouge**: Measures the overlap of n-grams between generated and reference texts.
- **BLEU**: Evaluates the quality of text generation based on precision.
- **AnswerRelevancy**: Assesses the relevance of answers to questions.
- **METEOR**: Evaluates machine translation quality.
- **Gruen**: Measures the quality of generated text by assessing grammaticality, redundancy, and focus.

## Installation

To install IndoxJudge, follow these steps:


```bash
git clone https://github.com/yourusername/indoxjudge.git
cd indoxjudge
```
## Setting Up the Python Environment

If you are running this project in your local IDE, please create a Python environment to ensure all dependencies are correctly managed. You can follow the steps below to set up a virtual environment named `indox_judge`:

### Windows

1. **Create the virtual environment:**
```bash
python -m venv indox_judge
```
2. **Activate the virtual environment:**
```bash
indox_judge\Scripts\activate
```

### macOS/Linux

1. **Create the virtual environment:**
   ```bash
   python3 -m venv indox_judge
```

2. **Activate the virtual environment:**
    ```bash
   source indox_judge/bin/activate
```
### Install Dependencies

Once the virtual environment is activated, install the required dependencies by running:

```bash
pip install -r requirements.txt
```




## Usage

To use IndoxJudge, load your API key, select the model, and choose the evaluation metrics. Here's an example demonstrating how to evaluate a model's response for faithfulness:

```python
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Import IndoxJudge and supported models
from indoxJudge.piplines import CustomEvaluator
from indoxJudge.models import OpenAi
from indoxJudge.metrics import Faithfulness

# Initialize the model with your API key
model = OpenAi(api_key=OPENAI_API_KEY,model="gpt-4o")

# Define your query and retrieval context
query = "What are the benefits of a Mediterranean diet?"
retrieval_context = [
    "The Mediterranean diet emphasizes eating primarily plant-based foods, such as fruits and vegetables, whole grains, legumes, and nuts. It also includes moderate amounts of fish and poultry, and low consumption of red meat. Olive oil is the main source of fat, providing monounsaturated fats which are beneficial for heart health.",
    "Research has shown that the Mediterranean diet can reduce the risk of heart disease, stroke, and type 2 diabetes. It is also associated with improved cognitive function and a lower risk of Alzheimer's disease. The diet's high content of fiber, antioxidants, and healthy fats contributes to its numerous health benefits.",
    "A Mediterranean diet has been linked to a longer lifespan and a reduced risk of chronic diseases. It promotes healthy aging and weight management due to its emphasis on whole, unprocessed foods and balanced nutrition."
]

# Obtain the model's response
response = "The Mediterranean diet is known for its health benefits, including reducing the risk of heart disease, stroke, and diabetes. It encourages the consumption of fruits, vegetables, whole grains, nuts, and olive oil, while limiting red meat. Additionally, this diet has been associated with better cognitive function and a reduced risk of Alzheimer's disease, promoting longevity and overall well-being."

# Initialize the Faithfulness metric
faithfulness_metrics = Faithfulness(llm_response=response, retrieval_context=retrieval_context)

# Create an evaluator with the selected metrics
evaluator = CustomEvaluator(metrics=[faithfulness_metrics], model=model)

# Evaluate the response
faithfulness_result = evaluator.judge()

# Output the evaluation result
print(faithfulness_result)
```
## Example Output

```json
{
  "faithfulness": {
    "claims": [
      "The Mediterranean diet is known for its health benefits.",
      "The Mediterranean diet reduces the risk of heart disease.",
      "The Mediterranean diet reduces the risk of stroke.",
      "The Mediterranean diet reduces the risk of diabetes.",
      "The Mediterranean diet encourages the consumption of fruits.",
      "The Mediterranean diet encourages the consumption of vegetables.",
      "The Mediterranean diet encourages the consumption of whole grains.",
      "The Mediterranean diet encourages the consumption of nuts.",
      "The Mediterranean diet encourages the consumption of olive oil.",
      "The Mediterranean diet limits red meat consumption.",
      "The Mediterranean diet is associated with better cognitive function.",
      "The Mediterranean diet is associated with a reduced risk of Alzheimer's disease.",
      "The Mediterranean diet promotes longevity.",
      "The Mediterranean diet promotes overall well-being."
    ],
    "truths": [
      "The Mediterranean diet is known for its health benefits.",
      "The Mediterranean diet reduces the risk of heart disease, stroke, and diabetes.",
      "The Mediterranean diet encourages the consumption of fruits, vegetables, whole grains, nuts, and olive oil.",
      "The Mediterranean diet limits red meat consumption.",
      "The Mediterranean diet has been associated with better cognitive function.",
      "The Mediterranean diet has been associated with a reduced risk of Alzheimer's disease.",
      "The Mediterranean diet promotes longevity and overall well-being."
    ],
    "reason": "The score is 1.0 because the 'actual output' aligns perfectly with the information presented in the 'retrieval context', showcasing the health benefits, disease risk reduction, cognitive function improvement, and overall well-being promotion of the Mediterranean diet."
  }
}
```
## Roadmap

We have an exciting roadmap planned for IndoxJudge:

  | Plan                                                                 |
| -------------------------------------------------------------------- |
 | Integration of additional metrics such as Diversity and Coherence.   |
| Introduction of a graphical user interface (GUI) for easier evaluation. |
  | Expansion of the toolkit to support evaluation in multiple languages. |
  | Release of a benchmarking suite for standardizing LLM evaluations.   |

## Contributing

We welcome contributions from the community! If you'd like to contribute, please fork the repository and create a pull request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a pull request


