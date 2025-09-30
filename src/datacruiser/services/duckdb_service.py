"""
DuckDB service for statistical queries on CSV data
"""

import duckdb
import os
import logging
from typing import Dict, Any, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)

class DuckDBService:
    """Service for DuckDB operations on CSV data"""

    def __init__(self, csv_file_path: str):
        self.csv_file_path = csv_file_path
        self.connection = None
        self.table_name = "crowd_flow"
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize DuckDB connection and create table"""
        try:
            self.connection = duckdb.connect()
            
            # Create table from CSV with proper type casting
            create_table_sql = f"""
            CREATE OR REPLACE TABLE {self.table_name} AS 
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
            FROM read_csv_auto('{self.csv_file_path}', header=true)
            """
            
            self.connection.execute(create_table_sql)
            logger.info(f"✅ DuckDB table '{self.table_name}' created successfully")
            
            # Get table info
            result = self.connection.execute(f"SELECT COUNT(*) as total_rows FROM {self.table_name}").fetchone()
            logger.info(f"📊 Loaded {result[0]} rows into DuckDB")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize DuckDB: {e}")
            raise
    
    def execute_query(self, sql_query: str) -> List[Dict[str, Any]]:
        """Execute SQL query and return results as list of dictionaries"""
        try:
            if not self.connection:
                raise Exception("DuckDB connection not initialized")
            
            # Execute query
            result = self.connection.execute(sql_query).fetchall()
            columns = [desc[0] for desc in self.connection.description]
            
            # Convert to list of dictionaries
            results = []
            for row in result:
                row_dict = {}
                for i, value in enumerate(row):
                    row_dict[columns[i]] = value
                results.append(row_dict)
            
            logger.info(f"✅ Executed query, returned {len(results)} rows")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error executing query: {e}")
            raise
    
    def get_table_schema(self) -> Dict[str, str]:
        """Get the table schema"""
        try:
            schema_query = f"DESCRIBE {self.table_name}"
            result = self.connection.execute(schema_query).fetchall()
            
            schema = {}
            for row in result:
                schema[row[0]] = row[1]
            
            return schema
            
        except Exception as e:
            logger.error(f"Error getting schema: {e}")
            return {}
    
    def get_sample_data(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get sample data from the table"""
        try:
            query = f"SELECT * FROM {self.table_name} LIMIT {limit}"
            return self.execute_query(query)
        except Exception as e:
            logger.error(f"Error getting sample data: {e}")
            return []
    
    def get_location_names(self) -> List[str]:
        """Get all unique location names"""
        try:
            query = f"SELECT DISTINCT Location_Name FROM {self.table_name} ORDER BY Location_Name"
            result = self.connection.execute(query).fetchall()
            return [row[0] for row in result]
        except Exception as e:
            logger.error(f"Error getting location names: {e}")
            return []
    
    def get_date_range(self) -> Dict[str, str]:
        """Get the date range of the data"""
        try:
            query = f"""
            SELECT 
                MIN(Date) as min_date,
                MAX(Date) as max_date
            FROM {self.table_name}
            """
            result = self.connection.execute(query).fetchone()
            return {
                "min_date": str(result[0]) if result[0] else None,
                "max_date": str(result[1]) if result[1] else None
            }
        except Exception as e:
            logger.error(f"Error getting date range: {e}")
            return {"min_date": None, "max_date": None}
    
    def close(self):
        """Close the DuckDB connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("🔒 DuckDB connection closed")
    
    def __del__(self):
        """Cleanup on deletion"""
        self.close()
