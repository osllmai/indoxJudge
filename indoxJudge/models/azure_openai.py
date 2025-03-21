from tenacity import retry, stop_after_attempt, wait_random_exponential
from loguru import logger
import sys

# Set up logging
logger.remove()  # Remove the default logger
logger.add(
    sys.stdout, format="<green>{level}</green>: <level>{message}</level>", level="INFO"
)

logger.add(
    sys.stdout, format="<red>{level}</red>: <level>{message}</level>", level="ERROR"
)


class AzureOpenAi:
    """
    A class to interface with Azure OpenAI's models for evaluation purposes.

    This class uses the Azure OpenAI API to send requests and receive responses, which are utilized
    for evaluating the performance of language models.
    """

    def __init__(
        self,
        api_key: str,
        azure_endpoint: str,
        deployment_name: str,
        api_version: str = "2024-02-15-preview",
        max_tokens: int = 4096,
    ):
        """
        Initializes the AzureOpenAi class with the specified credentials and configuration.

        Args:
            api_key (str): The API key for accessing the Azure OpenAI API.
            azure_endpoint (str): The Azure OpenAI endpoint URL.
            deployment_name (str): The deployment name of your model in Azure.
            api_version (str, optional): The API version to use. Defaults to "2024-02-15-preview".
            max_tokens (int, optional): Maximum number of tokens for responses. Defaults to 4096.
        """
        from openai import AzureOpenAI

        try:
            logger.info(f"Initializing AzureOpenAi with deployment: {deployment_name}")
            self.deployment_name = deployment_name
            self.max_tokens = max_tokens
            self.client = AzureOpenAI(
                api_key=api_key, api_version=api_version, azure_endpoint=azure_endpoint
            )

        except Exception as e:
            logger.error(f"Error initializing AzureOpenAi: {e}")
            raise

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _generate_response(
        self,
        messages: list,
        temperature: float = 0.00001,
    ) -> str:
        """
        Generates a response from the Azure OpenAI model.

        Args:
            messages (list): The list of messages to send to the model, formatted as a conversation.
            temperature (float, optional): The sampling temperature, influencing response randomness.
                Defaults to 0.00001.

        Returns:
            str: The generated response.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=temperature,
                max_tokens=self.max_tokens,
            )
            result = response.choices[0].message.content.strip()
            return result
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def generate_evaluation_response(
        self,
        prompt: str,
    ) -> str:
        """
        Generates a response to a custom evaluation prompt using the Azure OpenAI model.

        Args:
            prompt (str): The custom prompt to generate a response for.

        Returns:
            str: The generated response.
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are an assistant for LLM evaluation",
                },
                {"role": "user", "content": prompt},
            ]
            response = self._generate_response(
                messages,
                temperature=0.00001,
            )
            if response.startswith("```json") and response.endswith("```"):
                response = response[7:-3].strip()
            return response
        except Exception as e:
            logger.error(f"Error generating response to custom prompt: {e}")
            return str(e)

    def generate_interpretation(
        self,
        models_data,
        mode: str,
    ):
        """
        Generates interpretations based on different evaluation modes.

        Args:
            models_data: The data to be interpreted
            mode (str): The evaluation mode ('comparison', 'rag', 'safety', or 'llm')

        Returns:
            str: The generated interpretation
        """
        prompt = ""
        if mode == "comparison":
            from .interpretation_template.comparison_template import (
                ModelComparisonTemplate,
            )

            prompt = ModelComparisonTemplate.generate_comparison(
                models=models_data, mode="llm model quality"
            )
        elif mode == "rag":
            from .interpretation_template.rag_interpretation_template import (
                RAGEvaluationTemplate,
            )

            prompt = RAGEvaluationTemplate.generate_interpret(data=models_data)
        elif mode == "safety":
            from .interpretation_template.safety_interpretation_template import (
                SafetyEvaluationTemplate,
            )

            prompt = SafetyEvaluationTemplate.generate_interpret(data=models_data)
        elif mode == "llm":
            from .interpretation_template.llm_interpretation_template import (
                LLMEvaluatorTemplate,
            )

            prompt = LLMEvaluatorTemplate.generate_interpret(data=models_data)

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant to analyze charts",
            },
            {"role": "user", "content": prompt},
        ]
        response = self._generate_response(
            messages=messages,
        )

        if response.startswith("```json") and response.endswith("```"):
            response = response[7:-3].strip()
        return response
