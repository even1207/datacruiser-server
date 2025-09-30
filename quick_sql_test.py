#!/usr/bin/env python3
"""
Quick test of SQL generation for specific queries
"""

import os
import sys
import json

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from datacruiser.services.duckdb_service import DuckDBService
from datacruiser.services.text_to_sql_service import TextToSQLService
from datacruiser.services.llm_service import LLMService

def quick_test():
    """Quick test of specific SQL queries"""
    
    print("🚀 Quick SQL Test")
    print("=" * 30)
    
    # Initialize services
    llm_service = LLMService()
    csv_file_path = os.path.join(os.path.dirname(__file__), 'data', 'data.csv')
    duckdb_service = DuckDBService(csv_file_path)
    text_to_sql_service = TextToSQLService(llm_service, duckdb_service)
    
    # Test specific queries
    queries = [
        "What was the average crowd count last week?",
        "Which location had the most people last week?",
        "What is the total count for Park Street today?"
    ]
    
    for question in queries:
        print(f"\n🔍 Question: {question}")
        print("-" * 50)
        
        # Generate and execute
        result = text_to_sql_service.execute_question(question)
        
        if result['success']:
            print(f"✅ SQL Query:")
            print(f"   {result['sql_query']}")
            print(f"\n📊 Results ({len(result['results'])} rows):")
            
            for i, row in enumerate(result['results'][:3]):  # Show first 3 rows
                print(f"   Row {i+1}: {row}")
            
            if len(result['results']) > 3:
                print(f"   ... and {len(result['results']) - 3} more rows")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
    
    duckdb_service.close()

if __name__ == "__main__":
    quick_test()
