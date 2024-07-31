import json
import requests
import urllib3
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

def create_prompt(user_input):
    """
    Create a prompt for the robustness category and user input.

    Parameters
    ----------
    user_input : str
        The user input to be analyzed.

    Returns
    -------
    str
        The generated prompt.
    """
    prompt = f"""Based on the concept and definition of robustness,

    Please analyze the following paragraph and determine whether it falls into the robustness category:
    Paragraph: "{user_input}"

    For robustness, provide:
    1. A score from 0.000 to 1.000 indicating the relevance of the paragraph to the robustness category.
    2. A brief explanation justifying the score, with examples if necessary.
    Return the output in the following JSON format:
    {{
        "robustness": {{"score": , "explanation": ""}}
    }}
    """
    return prompt

def get_openai_response(prompt):
    """
    Get the OpenAI response for the given prompt.

    Parameters
    ----------
    prompt : str
        The prompt to be sent to the OpenAI API.

    Returns
    -------
    dict or None
        The response from the OpenAI API, or None if an error occurred.
    """
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    api_url = 'http://5.78.55.161/api/chat_completion/generate/'
    params = {
        "model": "gpt-3.5-turbo-0125",
        "messages": [
            {"role": "system", "content": "You are an AI assistant specialized in evaluating text based on specific categories. Provide detailed and accurate analysis according to these criteria."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 100,
        "temperature": 0.0,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stream": False
    }

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    response = requests.post(api_url, headers=headers, json=params, verify=False)

    if response.status_code == 200:
        return response.json()
    elif response.status_code == 403:
        print("Error 403: Forbidden. Check your API key and permissions.")
    else:
        print(f"Error: {response.status_code}, {response.text}")
    return None

class Analysis:
    """
    Analyze a paragraph based on the robustness category.

    Attributes
    ----------
    None
    """
    def __init__(self):
        """
        Constructs all the necessary attributes for the Analysis object.
        """
        pass

    def analyze(self, paragraph):
        """
        Analyze the given paragraph.

        Parameters
        ----------
        paragraph : str
            The paragraph to be analyzed.

        Returns
        -------
        str
            The analysis results in JSON format.
        """
        results = {}
        prompt = create_prompt(paragraph)
        response = get_openai_response(prompt)
        if response is None:
            results["robustness"] = {"score": 0, "explanation": "No response received"}
        else:
            content = response.get('text_message', '{}')
            try:
                results["robustness"] = json.loads(content).get("robustness", {"score": 0, "explanation": "No explanation"})
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON for category robustness: {e}")
                print(f"Raw response content for category robustness: {content}")
                results["robustness"] = {"score": 0, "explanation": "Invalid JSON format"}

        final_results = {
            'paragraph': paragraph,
            'model': "gpt-3.5-turbo-0125",
            'results': results
        }
        return json.dumps(final_results, indent=4)

class EvaluateResponse:
    """
    Evaluate the response for the robustness category.

    Attributes
    ----------
    response : str
        The response to be evaluated.
    """
    def __init__(self, response):
        """
        Constructs all the necessary attributes for the EvaluateResponse object.

        Parameters
        ----------
        response : str
            The response to be evaluated.
        """
        self.response = response
        try:
            self.response_json = json.loads(response)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            self.response_json = {'results': {"robustness": {"score": 0, "explanation": "Invalid JSON format"}}}

    def score(self):
        """
        Get the score from the response.

        Returns
        -------
        float
            The score for the robustness category.
        """
        return self.response_json['results'].get("robustness", {}).get("score", 0)

    def feedback(self):
        """
        Get the feedback from the response.

        Returns
        -------
        str
            The feedback explanation for the score.
        """
        return self.response_json['results'].get("robustness", {}).get("explanation", "No feedback available")
