from typing import List, Dict
import json


class CoverageTemplate:
    @staticmethod
    def extract_information_elements(text: str) -> str:
        return f"""Analyze the text and identify key information elements in these categories:
1. Core Facts (main events, findings, or claims)
2. Supporting Details (evidence, examples, or explanations)
3. Context (background information, setting, or framework)
4. Relationships (connections, causes, effects)
5. Conclusions (outcomes, implications, or recommendations)

For each element:
- Assign importance score between 0.0-1.0
- Consider relative importance within the text
- Ensure importance scores are well-distributed
- Maximum score of 1.0 reserved for truly critical elements

IMPORTANT: Return only in JSON format with elements grouped by category.
Example:
{{
    "elements": [
        {{
            "category": "core_facts",
            "content": "The study found a 25% increase in productivity",
            "importance": 0.9
        }},
        {{
            "category": "supporting_details",
            "content": "The increase was measured across 12 months",
            "importance": 0.6
        }}
    ]
}}

Text to analyze:
{text}

JSON:"""

    @staticmethod
    def evaluate_coverage(summary: str, elements: List[Dict]) -> str:
        elements_json = json.dumps(elements, indent=2)
        return f"""Evaluate how well the summary covers each information element.

Scoring Guidelines:
- Score range: 0.0 to 1.0 (never exceed 1.0)
- 1.0: Perfect coverage with all key details
- 0.8-0.9: Strong coverage with minor omissions
- 0.6-0.7: Adequate coverage with some gaps
- 0.4-0.5: Partial coverage with significant gaps
- 0.0-0.3: Poor coverage or major omissions

For each category, provide:
1. Coverage score following above guidelines
2. List of elements covered and missed
3. Clear reason for the score

Consider:
- Weight importance scores in evaluation
- Check both explicit and implicit coverage
- Verify meaning preservation
- Normalize scores to ensure they never exceed 1.0

IMPORTANT: Return only in JSON format.
Example:
{{
    "scores": [
        {{
            "category": "core_facts",
            "score": 0.85,
            "elements_covered": ["fact 1", "fact 2"],
            "elements_missed": ["fact 3"],
            "reason": "Summary captures main findings but misses one key statistic"
        }}
    ]
}}

Information Elements:
{elements_json}

Summary to analyze:
{summary}

JSON:"""

    @staticmethod
    def generate_final_verdict(
        scores: Dict, weighted_score: float, coverage_stats: Dict
    ) -> str:
        # Ensure weighted_score is bounded
        bounded_score = min(max(weighted_score, 0.0), 1.0)

        return f"""Based on the coverage analysis, provide a concise assessment of the summary's information coverage.

Guidelines for verdict:
- Focus on coverage quality and completeness
- Highlight strengths and gaps
- Consider importance weights
- Maintain consistency with scores
- Ensure all metrics are properly bounded (0-1)

Coverage Scores:
{json.dumps(scores, indent=2)}

Coverage Statistics:
{json.dumps(coverage_stats, indent=2)}

Final Weighted Score: {bounded_score:.2f}

Return only in JSON format with a 'verdict' key.
Example:
{{
    "verdict": "The summary achieves strong coverage of core facts (85%) but shows gaps in supporting details. Critical information is well preserved while less essential elements show partial coverage."
}}

JSON:"""