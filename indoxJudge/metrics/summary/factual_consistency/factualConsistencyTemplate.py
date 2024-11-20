from typing import List, Dict, Optional
import json


class FactualConsistencyTemplate:
    @staticmethod
    def extract_claims(text: str) -> str:
        return f"""Extract all factual claims from the text and categorize them into:
1. Numerical Claims (statistics, measurements, dates)
2. Entity Claims (about people, organizations, places)
3. Causal Claims (cause-effect relationships)
4. Descriptive Claims (properties, characteristics)
5. Comparative Claims (comparisons, rankings)

IMPORTANT: Return only in JSON format.
Example:
{{
    "claims": [
        {{
            "category": "numerical_claims",
            "claim": "The study involved 500 participants",
            "context": "Sample size description in methodology"
        }},
        {{
            "category": "causal_claims",
            "claim": "The intervention reduced anxiety by targeting stress responses",
            "context": "Discussion of mechanism of action"
        }}
    ]
}}

Text to analyze:
{text}

JSON:"""

    @staticmethod
    def verify_claims(summary_claims: List[Dict], source_text: str) -> str:
        claims_json = json.dumps(summary_claims, indent=2)
        return f"""Verify each claim from the summary against the source text. For each claim:
1. Assess factual consistency (0.0-1.0)
2. Identify supporting evidence from source
3. Note any errors or distortions
4. Provide explanation

Error types: contradiction, exaggeration, misattribution, unsupported, oversimplification

IMPORTANT: Return only in JSON format.
Example:
{{
    "verified_claims": [
        {{
            "claim": "The study showed 50% improvement",
            "source_evidence": "The study demonstrated a 48% improvement rate",
            "consistency_score": 0.9,
            "error_type": "minor_exaggeration",
            "explanation": "Slight overstatement but essentially accurate"
        }}
    ]
}}

Claims to verify:
{claims_json}

Source Text:
{source_text}

JSON:"""

    @staticmethod
    def generate_category_verdict(verified_claims: List[Dict]) -> str:
        return f"""Analyze the verified claims and provide a verdict for each category of claims.
Include overall score, consistent and inconsistent claims, and explanation.

IMPORTANT: Return only in JSON format.
Example:
{{
    "scores": [
        {{
            "category": "numerical_claims",
            "score": 0.85,
            "consistent_claims": ["claim1", "claim2"],
            "inconsistent_claims": [
                {{"claim": "claim3", "error": "20% overstated"}}
            ],
            "reason": "Most numerical claims accurate with minor deviations"
        }}
    ]
}}

Verified Claims:
{json.dumps(verified_claims, indent=2)}

JSON:"""
