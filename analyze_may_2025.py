#!/usr/bin/env python3
"""
Analyze May 2025 data specifically
"""

import os
import sys
import json

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from datacruiser.services.duckdb_service import DuckDBService
from datacruiser.services.text_to_sql_service import TextToSQLService
from datacruiser.services.llm_service import LLMService

def analyze_may_2025():
    """Analyze May 2025 data specifically"""
    
    print("🔍 May 2025 Data Analysis")
    print("=" * 40)
    
    # Initialize services
    llm_service = LLMService()
    csv_file_path = os.path.join(os.path.dirname(__file__), 'data', 'data.csv')
    duckdb_service = DuckDBService(csv_file_path)
    text_to_sql_service = TextToSQLService(llm_service, duckdb_service)
    
    # Test the question
    question = "Which location has the most crowded in 2025 May"
    
    print(f"Question: {question}")
    print("-" * 50)
    
    # Generate SQL
    sql_result = text_to_sql_service.generate_sql(question)
    print(f"Generated SQL: {sql_result['sql_query']}")
    
    # Execute and show results
    try:
        results = duckdb_service.execute_query(sql_result['sql_query'])
        print(f"\nResults from generated SQL:")
        for i, row in enumerate(results):
            print(f"  {i+1}. {row}")
    except Exception as e:
        print(f"SQL Error: {e}")
    
    # Let's manually check May 2025 data
    print(f"\n" + "="*50)
    print("MANUAL VERIFICATION OF MAY 2025 DATA")
    print("="*50)
    
    # Check if we have any 2025 data at all
    check_2025 = """
    SELECT COUNT(*) as count_2025
    FROM crowd_flow 
    WHERE EXTRACT(YEAR FROM Date) = 2025
    """
    
    result_2025 = duckdb_service.execute_query(check_2025)
    print(f"Records in 2025: {result_2025[0]['count_2025']}")
    
    # Check if we have any May data
    check_may = """
    SELECT COUNT(*) as count_may
    FROM crowd_flow 
    WHERE EXTRACT(YEAR FROM Date) = 2025 AND EXTRACT(MONTH FROM Date) = 5
    """
    
    result_may = duckdb_service.execute_query(check_may)
    print(f"Records in May 2025: {result_may[0]['count_may']}")
    
    # Check actual date range
    date_range = """
    SELECT 
        MIN(Date) as min_date,
        MAX(Date) as max_date
    FROM crowd_flow
    """
    
    date_result = duckdb_service.execute_query(date_range)
    print(f"Actual date range: {date_result[0]['min_date']} to {date_result[0]['max_date']}")
    
    # If no May 2025 data, let's see what we do have
    if result_may[0]['count_may'] == 0:
        print(f"\nNo May 2025 data found. Let's see what months we have:")
        
        months_query = """
        SELECT 
            EXTRACT(YEAR FROM Date) as year,
            EXTRACT(MONTH FROM Date) as month,
            COUNT(*) as count
        FROM crowd_flow
        GROUP BY EXTRACT(YEAR FROM Date), EXTRACT(MONTH FROM Date)
        ORDER BY year DESC, month DESC
        LIMIT 10
        """
        
        months_result = duckdb_service.execute_query(months_query)
        for row in months_result:
            print(f"  {row['year']}-{row['month']:02d}: {row['count']} records")
    
    # Let's also check what the LLM might be interpreting as "May 2025"
    print(f"\n" + "="*50)
    print("CHECKING WHAT LLM MIGHT BE INTERPRETING")
    print("="*50)
    
    # Check if there are any dates that might be interpreted as May 2025
    check_may_like = """
    SELECT 
        Date,
        Location_Name,
        TotalCount
    FROM crowd_flow 
    WHERE Date LIKE '%05%' OR Date LIKE '%May%'
    ORDER BY TotalCount DESC
    LIMIT 10
    """
    
    may_like_result = duckdb_service.execute_query(check_may_like)
    print(f"Records that might be interpreted as May:")
    for row in may_like_result:
        print(f"  {row['Date']} | {row['Location_Name']} | {row['TotalCount']}")
    
    # Check the actual maximum values
    print(f"\n" + "="*50)
    print("ACTUAL MAXIMUM VALUES")
    print("="*50)
    
    max_query = """
    SELECT 
        Location_Name,
        Date,
        TotalCount
    FROM crowd_flow
    ORDER BY TotalCount DESC
    LIMIT 10
    """
    
    max_result = duckdb_service.execute_query(max_query)
    print(f"Top 10 highest counts:")
    for i, row in enumerate(max_result):
        print(f"  {i+1:2d}. {row['Location_Name']:15s} | {row['Date']} | {row['TotalCount']:4d}")
    
    duckdb_service.close()

if __name__ == "__main__":
    analyze_may_2025()
