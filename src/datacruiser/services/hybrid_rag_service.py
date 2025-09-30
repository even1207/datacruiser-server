"""
Hybrid RAG service that supports both statistical and trend queries
"""

import logging
from typing import Dict, Any, List, Optional
import json

from .rag_service import RAGService
from .query_classifier import QueryClassifier
from .text_to_sql_service import TextToSQLService
from .duckdb_service import DuckDBService
from .llm_service import LLMService
from ..config import Config

logger = logging.getLogger(__name__)


class HybridRAGService:
    """Hybrid RAG service supporting both statistical and trend queries"""
    
    def __init__(self, model_service, llm_service: LLMService, csv_file_path: str):
        self.model_service = model_service
        self.llm_service = llm_service
        
        # Initialize components
        self.rag_service = RAGService(model_service, llm_service)
        self.query_classifier = QueryClassifier(llm_service)
        self.duckdb_service = DuckDBService(csv_file_path)
        self.text_to_sql_service = TextToSQLService(llm_service, self.duckdb_service)
        
        logger.info("✅ Hybrid RAG service initialized")
    
    def initialize_data(self) -> bool:
        """Initialize the RAG service data"""
        return self.rag_service.initialize_data()
    
    def process_question(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Process a question using the hybrid approach
        
        Args:
            question: Natural language question
            top_k: Number of similar records to retrieve for trend queries
            
        Returns:
            Dict with response, query type, and metadata
        """
        try:
            logger.info(f"🔍 Processing question: {question}")
            
            # Step 1: Classify the query
            classification = self.query_classifier.classify_query(question)
            query_type = classification["query_type"]
            confidence = classification["confidence"]
            
            logger.info(f"📊 Query classified as: {query_type} (confidence: {confidence:.2f})")
            
            # Step 2: Route to appropriate handler
            if query_type == "statistical":
                return self._handle_statistical_query(question, classification)
            else:  # trend
                return self._handle_trend_query(question, classification, top_k)
                
        except Exception as e:
            logger.error(f"❌ Error processing question: {e}")
            return {
                "question": question,
                "answer": f"I encountered an error processing your question: {str(e)}",
                "query_type": "error",
                "success": False,
                "error": str(e)
            }
    
    def _handle_statistical_query(self, question: str, classification: Dict[str, Any]) -> Dict[str, Any]:
        """Handle statistical queries using Text-to-SQL"""
        try:
            logger.info("📈 Processing statistical query with Text-to-SQL")
            
            # Execute the question using Text-to-SQL
            sql_result = self.text_to_sql_service.execute_question(question)
            
            if not sql_result["success"]:
                return {
                    "question": question,
                    "answer": f"I couldn't process your statistical query: {sql_result.get('error', 'Unknown error')}",
                    "query_type": "statistical",
                    "success": False,
                    "error": sql_result.get("error")
                }
            
            # Generate natural language response from SQL results
            answer = self._generate_statistical_response(question, sql_result)
            
            return {
                "question": question,
                "answer": answer,
                "query_type": "statistical",
                "success": True,
                "sql_query": sql_result["sql_query"],
                "sql_results": sql_result["results"],
                "confidence": sql_result["confidence"],
                "explanation": sql_result["explanation"]
            }
            
        except Exception as e:
            logger.error(f"Error handling statistical query: {e}")
            return {
                "question": question,
                "answer": f"I encountered an error processing your statistical query: {str(e)}",
                "query_type": "statistical",
                "success": False,
                "error": str(e)
            }
    
    def _handle_trend_query(self, question: str, classification: Dict[str, Any], top_k: int) -> Dict[str, Any]:
        """Handle trend queries using RAG"""
        try:
            logger.info("📊 Processing trend query with RAG")
            
            # Use existing RAG service for trend analysis
            similar_records = self.rag_service.search_similar_records_with_context(question, top_k)
            
            if not similar_records:
                return {
                    "question": question,
                    "answer": "I couldn't find any relevant data to analyze trends for your question.",
                    "query_type": "trend",
                    "success": False,
                    "similar_records": []
                }
            
            # Generate LLM response
            answer = self.llm_service.generate_response(question, similar_records)
            
            return {
                "question": question,
                "answer": answer,
                "query_type": "trend",
                "success": True,
                "similar_records": [record.to_dict() for record in similar_records],
                "total_records": len(self.rag_service.records),
                "from_cache": self.rag_service.cache_manager.is_cache_valid()
            }
            
        except Exception as e:
            logger.error(f"Error handling trend query: {e}")
            return {
                "question": question,
                "answer": f"I encountered an error processing your trend query: {str(e)}",
                "query_type": "trend",
                "success": False,
                "error": str(e)
            }
    
    def _generate_statistical_response(self, question: str, sql_result: Dict[str, Any]) -> str:
        """Generate natural language response from SQL results"""
        try:
            if not self.llm_service.is_available():
                return self._generate_fallback_statistical_response(sql_result)
            
            # Format SQL results for LLM
            results_text = self._format_sql_results(sql_result["results"])
            
            prompt = f"""
You are a data analysis assistant. Generate a clear, natural language response based on the SQL query results.

Question: "{question}"

SQL Query: {sql_result["sql_query"]}

Query Results:
{results_text}

Instructions:
1. Answer the user's question directly based on the data
2. Include specific numbers and values from the results
3. Be clear and concise
4. If the results are empty, explain why
5. If there are multiple rows, summarize the key findings
6. Use professional but accessible language

Answer in English.
"""

            response = self.llm_service.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=400
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating statistical response: {e}")
            return self._generate_fallback_statistical_response(sql_result)
    
    def _format_sql_results(self, results: List[Dict[str, Any]]) -> str:
        """Format SQL results for LLM consumption"""
        if not results:
            return "No results found."
        
        if len(results) == 1:
            # Single result
            result = results[0]
            formatted = []
            for key, value in result.items():
                formatted.append(f"{key}: {value}")
            return "\n".join(formatted)
        else:
            # Multiple results - show first few
            formatted = []
            for i, result in enumerate(results[:5]):  # Limit to first 5
                row_data = []
                for key, value in result.items():
                    row_data.append(f"{key}={value}")
                formatted.append(f"Row {i+1}: {', '.join(row_data)}")
            
            if len(results) > 5:
                formatted.append(f"... and {len(results) - 5} more rows")
            
            return "\n".join(formatted)
    
    def _generate_fallback_statistical_response(self, sql_result: Dict[str, Any]) -> str:
        """Generate fallback response when LLM is not available"""
        results = sql_result["results"]
        
        if not results:
            return "No data found matching your criteria."
        
        if len(results) == 1:
            result = results[0]
            if "average_count" in result:
                return f"The average count is {result['average_count']:.2f}."
            elif "max_count" in result:
                return f"The maximum count is {result['max_count']} at {result.get('Location_Name', 'unknown location')}."
            elif "min_count" in result:
                return f"The minimum count is {result['min_count']} at {result.get('Location_Name', 'unknown location')}."
            elif "total_count" in result:
                return f"The total count is {result['total_count']}."
            elif "record_count" in result:
                return f"Found {result['record_count']} records."
            else:
                return f"Query executed successfully. Result: {result}"
        else:
            return f"Query executed successfully. Found {len(results)} results. Please check the detailed results for more information."
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and status"""
        return {
            "rag_initialized": self.rag_service.is_initialized(),
            "duckdb_connected": self.duckdb_service.connection is not None,
            "llm_available": self.llm_service.is_available(),
            "total_records": len(self.rag_service.records) if self.rag_service.records else 0,
            "location_count": len(self.duckdb_service.get_location_names()),
            "date_range": self.duckdb_service.get_date_range()
        }
    
    def close(self):
        """Clean up resources"""
        if self.duckdb_service:
            self.duckdb_service.close()
        logger.info("🔒 Hybrid RAG service closed")
