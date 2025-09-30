#!/usr/bin/env python3
"""
Fix DuckDB column mapping issue
"""

import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from datacruiser.services.duckdb_service import DuckDBService

def fix_duckdb_mapping():
    """Fix the DuckDB column mapping issue"""
    
    print("🔧 Fixing DuckDB Column Mapping")
    print("=" * 50)
    
    # Initialize DuckDB service
    csv_file_path = os.path.join(os.path.dirname(__file__), 'data', 'data.csv')
    duckdb_service = DuckDBService(csv_file_path)
    
    # Check the current table structure
    print("1. Current table structure:")
    print("-" * 30)
    
    describe_query = "DESCRIBE crowd_flow"
    describe_result = duckdb_service.execute_query(describe_query)
    for row in describe_result:
        print(f"  {row['column_name']}: {row['column_type']}")
    
    # Check the first few rows to see the mapping
    print(f"\n2. First 5 rows from current table:")
    print("-" * 40)
    
    sample_query = "SELECT * FROM crowd_flow LIMIT 5"
    sample_result = duckdb_service.execute_query(sample_query)
    for i, row in enumerate(sample_result):
        print(f"  Row {i+1}: {row}")
    
    # Check the specific problematic record
    print(f"\n3. Problematic record (Park Street 2025-05-29 02:00:00):")
    print("-" * 60)
    
    problem_query = """
    SELECT 
        Location_Name,
        Date,
        TotalCount,
        Hour,
        Day
    FROM crowd_flow 
    WHERE Location_Name = 'Park Street' 
    AND Date = '2025-05-29 02:00:00'
    """
    
    problem_result = duckdb_service.execute_query(problem_query)
    print(f"Current result: {problem_result}")
    
    # Try to recreate the table with explicit column mapping
    print(f"\n4. Recreating table with explicit column mapping:")
    print("-" * 50)
    
    # First, let's check what the CSV header looks like
    with open(csv_file_path, 'r') as f:
        header_line = f.readline().strip()
        print(f"CSV Header: {header_line}")
    
    # Create a new table with explicit column order
    recreate_sql = f"""
    CREATE OR REPLACE TABLE crowd_flow_fixed AS 
    SELECT 
        Location_code,
        Location_Name,
        CAST(Date AS TIMESTAMP) as Date,
        CAST(TotalCount AS INTEGER) as TotalCount,
        CAST(Hour AS INTEGER) as Hour,
        Day,
        CAST(DayNo AS INTEGER) as DayNo,
        Week,
        CAST(LastWeek AS INTEGER) as LastWeek,
        CAST(Previous4DayTimeAvg AS INTEGER) as Previous4DayTimeAvg,
        CAST(ObjectId AS INTEGER) as ObjectId,
        CAST(LastYear AS INTEGER) as LastYear,
        CAST(Previous52DayTimeAvg AS INTEGER) as Previous52DayTimeAvg
    FROM read_csv_auto('{csv_file_path}', header=true)
    """
    
    try:
        duckdb_service.connection.execute(recreate_sql)
        print("✅ Table recreated successfully")
        
        # Check the fixed record
        fixed_query = """
        SELECT 
            Location_Name,
            Date,
            TotalCount,
            Hour,
            Day
        FROM crowd_flow_fixed 
        WHERE Location_Name = 'Park Street' 
        AND Date = '2025-05-29 02:00:00'
        """
        
        fixed_result = duckdb_service.execute_query(fixed_query)
        print(f"Fixed result: {fixed_result}")
        
        # Check if this matches the CSV
        print(f"\n5. Comparing with CSV data:")
        print("-" * 30)
        print("CSV Line 3003: A004,Park Street,2025/05/29 02:00:00+00,310,2,Thursday,4,2025.22,229,272,3002,273,283")
        print("Expected: Date=2025-05-29 02:00:00, TotalCount=310, Hour=2")
        print(f"Actual:   Date={fixed_result[0]['Date']}, TotalCount={fixed_result[0]['TotalCount']}, Hour={fixed_result[0]['Hour']}")
        
        if fixed_result[0]['TotalCount'] == 310 and fixed_result[0]['Hour'] == 2:
            print("✅ FIXED! The data now matches the CSV")
        else:
            print("❌ Still not fixed. Need to investigate further.")
            
    except Exception as e:
        print(f"❌ Error recreating table: {e}")
    
    # Let's also check if there are any other records that might be affected
    print(f"\n6. Checking other records for consistency:")
    print("-" * 45)
    
    consistency_query = """
    SELECT 
        Location_Name,
        Date,
        TotalCount,
        Hour
    FROM crowd_flow_fixed 
    WHERE Location_Name = 'Park Street' 
    AND Date >= '2025-05-29' AND Date < '2025-05-30'
    ORDER BY Hour
    LIMIT 5
    """
    
    consistency_result = duckdb_service.execute_query(consistency_query)
    print("Sample records from fixed table:")
    for i, row in enumerate(consistency_result):
        print(f"  {i+1}. {row['Location_Name']} | {row['Date']} | Count: {row['TotalCount']:4d} | Hour: {row['Hour']:2d}")
    
    duckdb_service.close()
    print(f"\n✅ Analysis completed!")

if __name__ == "__main__":
    fix_duckdb_mapping()
