"""
Query classifier to distinguish between statistical and trend queries
"""

import re
import json
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class QueryClassifier:
    """Classifies queries as statistical or trend-based"""
    
    def __init__(self, llm_service=None):
        self.llm_service = llm_service
        
        # Statistical query keywords
        self.statistical_keywords = [
            'average', 'mean', 'sum', 'total', 'count', 'maximum', 'minimum', 'max', 'min',
            'median', 'percentile', 'aggregate', 'group by', 'statistics', 'statistical',
            'how many', 'how much', 'total count', 'busiest', 'highest', 'lowest',
            'top', 'bottom', 'rank', 'ranking', 'most', 'least', 'number of',
            'aggregation', 'summary', 'summarize'
        ]
        
        # Trend query keywords
        self.trend_keywords = [
            'trend', 'trends', 'change', 'changes', 'increase', 'decrease', 'growth',
            'decline', 'pattern', 'patterns', 'compare', 'comparison', 'versus', 'vs',
            'over time', 'throughout', 'across', 'during', 'evolution', 'progression',
            'fluctuation', 'variation', 'seasonal', 'cyclical', 'correlation',
            'relationship', 'behavior', 'movement', 'shift', 'transition'
        ]
    
    def classify_query(self, question: str) -> Dict[str, Any]:
        """
        Classify a query as statistical or trend-based
        
        Returns:
            Dict with classification results including:
            - query_type: 'statistical' or 'trend'
            - confidence: float between 0 and 1
            - reasoning: explanation of classification
            - extracted_entities: relevant entities from the query
        """
        try:
            # Try LLM-based classification first
            if self.llm_service and self.llm_service.is_available():
                return self._llm_classify_query(question)
            else:
                return self._rule_based_classify_query(question)
                
        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            return self._rule_based_classify_query(question)
    
    def _llm_classify_query(self, question: str) -> Dict[str, Any]:
        """Use LLM for query classification"""
        try:
            prompt = f"""
Analyze this question about crowd flow data and classify it as either "statistical" or "trend".

Question: "{question}"

Classification rules:
- STATISTICAL: Questions asking for specific numbers, aggregations, rankings, or quantitative analysis
  Examples: "What was the average crowd count?", "Which location had the most people?", "How many people were there last week?"
  
- TREND: Questions asking about patterns, changes over time, comparisons, or qualitative analysis
  Examples: "How has the crowd changed over time?", "Compare the trends between locations", "What patterns do you see?"

Respond with a JSON object containing:
1. "query_type": "statistical" or "trend"
2. "confidence": float between 0.0 and 1.0
3. "reasoning": brief explanation of why this classification was chosen
4. "extracted_entities": object with "locations": [], "time_ranges": [], "metrics": []

Respond with valid JSON only, no additional text.
"""

            response = self.llm_service.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Clean up the response
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            
            result = json.loads(result_text)
            
            # Validate the result
            if result.get("query_type") not in ["statistical", "trend"]:
                raise ValueError("Invalid query_type")
            
            confidence = float(result.get("confidence", 0.5))
            if not 0.0 <= confidence <= 1.0:
                confidence = 0.5
            
            return {
                "query_type": result["query_type"],
                "confidence": confidence,
                "reasoning": result.get("reasoning", "LLM classification"),
                "extracted_entities": result.get("extracted_entities", {
                    "locations": [],
                    "time_ranges": [],
                    "metrics": []
                })
            }
            
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return self._rule_based_classify_query(question)
    
    def _rule_based_classify_query(self, question: str) -> Dict[str, Any]:
        """Rule-based query classification as fallback"""
        question_lower = question.lower()
        
        # Count keyword matches
        statistical_score = sum(1 for keyword in self.statistical_keywords 
                              if keyword in question_lower)
        trend_score = sum(1 for keyword in self.trend_keywords 
                         if keyword in question_lower)
        
        # Extract entities
        entities = self._extract_entities(question)
        
        # Determine classification
        if statistical_score > trend_score:
            query_type = "statistical"
            confidence = min(0.8, 0.5 + (statistical_score * 0.1))
            reasoning = f"Contains {statistical_score} statistical keywords"
        elif trend_score > statistical_score:
            query_type = "trend"
            confidence = min(0.8, 0.5 + (trend_score * 0.1))
            reasoning = f"Contains {trend_score} trend keywords"
        else:
            # Default to statistical for ambiguous queries
            query_type = "statistical"
            confidence = 0.5
            reasoning = "Ambiguous query, defaulting to statistical"
        
        return {
            "query_type": query_type,
            "confidence": confidence,
            "reasoning": reasoning,
            "extracted_entities": entities
        }
    
    def _extract_entities(self, question: str) -> Dict[str, list]:
        """Extract entities from the question"""
        entities = {
            "locations": [],
            "time_ranges": [],
            "metrics": []
        }
        
        # Extract location names (simple pattern matching)
        location_patterns = [
            r'\b([A-Z][a-z]+ Street)\b',
            r'\b([A-Z][a-z]+ Avenue)\b',
            r'\b([A-Z][a-z]+ Road)\b',
            r'\b([A-Z][a-z]+ Boulevard)\b',
            r'\b([A-Z][a-z]+ Lane)\b'
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, question)
            entities["locations"].extend(matches)
        
        # Extract time ranges
        time_patterns = [
            r'\b(last \d+ days?)\b',
            r'\b(past \d+ days?)\b',
            r'\b(\d+ days? ago)\b',
            r'\b(last week)\b',
            r'\b(last month)\b',
            r'\b(last year)\b',
            r'\b(today)\b',
            r'\b(yesterday)\b'
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, question.lower())
            entities["time_ranges"].extend(matches)
        
        # Extract metrics
        metric_keywords = [
            'totalcount', 'total count', 'crowd count', 'footfall',
            'lastweek', 'last week', 'lastyear', 'last year',
            'average', 'mean', 'maximum', 'minimum'
        ]
        
        for metric in metric_keywords:
            if metric in question.lower():
                entities["metrics"].append(metric)
        
        return entities
