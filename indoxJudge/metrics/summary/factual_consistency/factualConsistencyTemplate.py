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

For identical or nearly identical text comparison, exact matches should be treated as fully consistent.

IMPORTANT: Return only in JSON format with meaningful categorization.

Example:
{{
    "claims": [
        {{
            "category": "numerical_claims",
            "claim": "The study involved 500 participants",
            "context": "Sample size description in methodology",
            "text_span": "involved 500 participants"
        }},
        {{
            "category": "causal_claims",
            "claim": "The intervention reduced anxiety by targeting stress responses",
            "context": "Discussion of mechanism of action",
            "text_span": "reduced anxiety by targeting stress responses"
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
1. Assess factual consistency (0.0-1.0):
   - Score 1.0 for exact matches or semantically identical content
   - Score 0.9-0.99 for minor wording differences with same meaning
   - Score 0.7-0.89 for mostly consistent with slight variations
   - Score 0.5-0.69 for partially consistent with notable differences
   - Score 0.0-0.49 for inconsistent or unsupported claims

2. Find supporting evidence using exact text matching when possible
3. Identify any discrepancies or errors
4. Provide detailed explanation

Error types should only be assigned if there's a genuine discrepancy:
- contradiction: Direct conflict with source
- exaggeration: Clear overstatement of facts
- misattribution: Incorrect source or attribution
- unsupported: No supporting evidence found
- oversimplification: Significant loss of accuracy

For identical or nearly identical texts, claims should receive high consistency scores (0.9-1.0) when the content matches.

IMPORTANT: Return only in JSON format.
Example:
{{
    "verified_claims": [
        {{
            "claim": "The study involved 500 participants",
            "source_evidence": "The study involved 500 participants",
            "consistency_score": 1.0,
            "error_type": null,
            "explanation": "Exact match found in source text"
        }},
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
Calculate scores based on:
1. Exact matches should receive maximum category scores
2. Average consistency scores for each category
3. Proportion of fully consistent claims
4. Impact of any errors on overall reliability

For identical or nearly identical texts, category scores should reflect the high consistency.

IMPORTANT: Return only in JSON format.
Example:
{{
    "scores": [
        {{
            "category": "numerical_claims",
            "score": 1.0,
            "consistent_claims": ["The study had 500 participants"],
            "inconsistent_claims": [],
            "reason": "All numerical claims exactly match source text"
        }},
        {{
            "category": "causal_claims",
            "score": 0.85,
            "consistent_claims": ["Treatment reduced symptoms"],
            "inconsistent_claims": [
                {{"claim": "Treatment cured all conditions", "error": "Unsupported generalization"}}
            ],
            "reason": "Most causal claims accurate with one unsupported claim"
        }}
    ]
}}

Verified Claims:
{json.dumps(verified_claims, indent=2)}

JSON:"""
