#!/usr/bin/env python3
"""
Example usage of the hybrid RAG system
"""

import os
import sys
import json

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from datacruiser.services.hybrid_rag_service import HybridRAGService
from datacruiser.services.model_service import ModelService
from datacruiser.services.llm_service import LLMService

def main():
    """Demonstrate hybrid RAG usage"""
    
    print("🚀 Hybrid RAG System Example")
    print("=" * 40)
    
    # Initialize services
    print("Initializing services...")
    model_service = ModelService()
    llm_service = LLMService()
    
    # Initialize hybrid service
    csv_file_path = os.path.join(os.path.dirname(__file__), 'data', 'data.csv')
    hybrid_service = HybridRAGService(model_service, llm_service, csv_file_path)
    
    # Initialize data
    print("Loading data...")
    if not hybrid_service.initialize_data():
        print("❌ Failed to initialize data")
        return
    
    print("✅ System initialized!")
    
    # Test questions
    test_questions = [
        # Statistical queries
        "What was the average crowd count in the past 5 days?",
        "Which location had the most people last week?",
        "What is the total count for Park Street today?",
        
        # Trend queries
        "Compare the recent trend of Park Street vs Market Street",
        "How has the crowd flow changed over time?",
        "What patterns do you see in the data?"
    ]
    
    # Process questions
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")
        print("-" * 50)
        
        try:
            result = hybrid_service.process_question(question)
            
            print(f"Query Type: {result.get('query_type', 'unknown')}")
            print(f"Success: {result.get('success', False)}")
            
            if result.get('success'):
                print(f"Answer: {result.get('answer', 'No answer')}")
                
                # Show additional info based on query type
                if result.get('query_type') == 'statistical':
                    if 'sql_query' in result:
                        print(f"SQL Query: {result['sql_query']}")
                    if 'sql_results' in result:
                        print(f"SQL Results: {len(result['sql_results'])} rows")
                elif result.get('query_type') == 'trend':
                    if 'similar_records' in result:
                        print(f"Similar Records: {len(result['similar_records'])}")
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error processing question: {e}")
    
    # Clean up
    hybrid_service.close()
    print("\n✅ Example completed!")

if __name__ == "__main__":
    main()
