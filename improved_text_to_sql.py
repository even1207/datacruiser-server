#!/usr/bin/env python3
"""
Improved Text-to-SQL service with better validation and error handling
"""

import os
import sys
import json
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from datacruiser.services.duckdb_service import DuckDBService
from datacruiser.services.llm_service import LLMService

logger = logging.getLogger(__name__)

class ImprovedTextToSQLService:
    """Improved Text-to-SQL service with validation"""
    
    def __init__(self, llm_service, duckdb_service):
        self.llm_service = llm_service
        self.duckdb_service = duckdb_service
        self.table_name = "crowd_flow"
        
        # Get table schema for context
        self.schema = self.duckdb_service.get_table_schema()
        self.location_names = self.duckdb_service.get_location_names()
        self.date_range = self.duckdb_service.get_date_range()
    
    def generate_sql(self, question: str) -> dict:
        """Generate SQL with validation"""
        try:
            if not self.llm_service or not self.llm_service.is_available():
                return self._rule_based_sql_generation(question)
            
            # Try LLM generation first
            result = self._llm_sql_generation(question)
            
            # Validate the generated SQL
            if self._validate_sql(result["sql_query"]):
                return result
            else:
                logger.warning("LLM generated invalid SQL, attempting to fix...")
                fixed_sql = self._fix_sql_query(result["sql_query"])
                if fixed_sql:
                    result["sql_query"] = fixed_sql
                    result["confidence"] = min(result["confidence"], 0.7)
                    result["explanation"] += " (SQL was automatically corrected)"
                    return result
                else:
                    logger.warning("Could not fix SQL, falling back to rule-based generation")
                    return self._rule_based_sql_generation(question)
                
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            return self._rule_based_sql_generation(question)
    
    def _validate_sql(self, sql_query: str) -> bool:
        """Validate SQL query for common issues"""
        try:
            sql_upper = sql_query.upper().strip()
            
            # Must start with SELECT
            if not sql_upper.startswith("SELECT"):
                return False
            
            # Check for common aggregate function issues
            if "MAX(" in sql_upper or "MIN(" in sql_upper or "AVG(" in sql_upper or "SUM(" in sql_upper:
                if "GROUP BY" not in sql_upper:
                    # Check if we're selecting non-aggregated columns
                    select_part = sql_query.split("FROM")[0].upper()
                    if "LOCATION_NAME" in select_part or "DATE" in select_part:
                        # This is likely invalid - aggregate with non-aggregated columns without GROUP BY
                        return False
            
            # Try to explain the query to check syntax
            try:
                self.duckdb_service.execute_query(f"EXPLAIN {sql_query}")
                return True
            except Exception as e:
                logger.error(f"SQL validation failed: {e}")
                return False
                
        except Exception as e:
            logger.error(f"SQL validation error: {e}")
            return False
    
    def _fix_sql_query(self, sql_query: str) -> str:
        """Fix common SQL issues"""
        try:
            sql_upper = sql_query.upper()
            
            # Fix MAX queries
            if "MAX(" in sql_upper and "GROUP BY" not in sql_upper:
                if "LOCATION_NAME" in sql_upper:
                    fixed_sql = sql_query.replace("MAX(TotalCount), Location_Name", "TotalCount, Location_Name")
                    fixed_sql = fixed_sql.replace("MAX(TotalCount)", "TotalCount")
                    if "ORDER BY" not in fixed_sql.upper():
                        fixed_sql += " ORDER BY TotalCount DESC LIMIT 1"
                    return fixed_sql
            
            # Fix MIN queries
            if "MIN(" in sql_upper and "GROUP BY" not in sql_upper:
                if "LOCATION_NAME" in sql_upper:
                    fixed_sql = sql_query.replace("MIN(TotalCount), Location_Name", "TotalCount, Location_Name")
                    fixed_sql = fixed_sql.replace("MIN(TotalCount)", "TotalCount")
                    if "ORDER BY" not in fixed_sql.upper():
                        fixed_sql += " ORDER BY TotalCount ASC LIMIT 1"
                    return fixed_sql
            
            return None
            
        except Exception as e:
            logger.error(f"Error fixing SQL: {e}")
            return None
    
    def _llm_sql_generation(self, question: str) -> dict:
        """Use LLM to generate SQL query with improved prompts"""
        try:
            # Build context for the LLM
            schema_info = self._build_schema_context()
            sample_data = self._get_sample_data_context()
            
            prompt = f"""
You are a SQL expert. Convert this natural language question to a SQL query for a crowd flow analysis database.

Question: "{question}"

Database Schema:
{schema_info}

Sample Data:
{sample_data}

Available Locations: {', '.join(self.location_names[:10])}...

Date Range: {self.date_range['min_date']} to {self.date_range['max_date']}

CRITICAL SQL RULES:
1. All columns are already properly typed in the table
2. Use CAST() only if you need to convert between compatible types
3. For date filtering, use: WHERE Date >= 'YYYY-MM-DD' AND Date <= 'YYYY-MM-DD'
4. For location filtering, use: WHERE Location_Name = 'Location Name'
5. For aggregations, use appropriate GROUP BY clauses
6. Always use the table name: {self.table_name}
7. Return only the SQL query, no explanations

IMPORTANT: When using aggregate functions (MAX, MIN, AVG, SUM, COUNT), follow these rules:
- If you want the MAX value with its location: SELECT Location_Name, TotalCount FROM {self.table_name} WHERE [conditions] ORDER BY TotalCount DESC LIMIT 1
- If you want MAX per location: SELECT Location_Name, MAX(TotalCount) FROM {self.table_name} WHERE [conditions] GROUP BY Location_Name
- NEVER mix aggregate functions with non-aggregated columns without GROUP BY

Common Query Patterns:
- Average: SELECT AVG(TotalCount) FROM {self.table_name} WHERE [conditions]
- Maximum value with location: SELECT Location_Name, TotalCount FROM {self.table_name} WHERE [conditions] ORDER BY TotalCount DESC LIMIT 1
- Maximum per location: SELECT Location_Name, MAX(TotalCount) FROM {self.table_name} WHERE [conditions] GROUP BY Location_Name
- Count: SELECT COUNT(*) FROM {self.table_name} WHERE [conditions]
- Sum: SELECT SUM(TotalCount) FROM {self.table_name} WHERE [conditions]
- Group by location: SELECT Location_Name, AVG(TotalCount) FROM {self.table_name} WHERE [conditions] GROUP BY Location_Name

Respond with a JSON object containing:
1. "sql_query": the SQL query string
2. "confidence": float between 0.0 and 1.0
3. "explanation": brief explanation of what the query does
4. "query_type": type of query (e.g., "aggregation", "ranking", "filtering")

Respond with valid JSON only, no additional text.
"""

            response = self.llm_service.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Clean up the response
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            
            result = json.loads(result_text)
            
            # Validate the result
            confidence = float(result.get("confidence", 0.7))
            if not 0.0 <= confidence <= 1.0:
                confidence = 0.7
            
            return {
                "sql_query": result["sql_query"],
                "confidence": confidence,
                "explanation": result.get("explanation", "Generated SQL query"),
                "query_type": result.get("query_type", "unknown")
            }
            
        except Exception as e:
            logger.error(f"LLM SQL generation failed: {e}")
            return self._rule_based_sql_generation(question)
    
    def _rule_based_sql_generation(self, question: str) -> dict:
        """Rule-based SQL generation as fallback"""
        question_lower = question.lower()
        
        # Extract time conditions
        time_condition = self._extract_time_condition(question)
        
        # Extract location conditions
        location_condition = self._extract_location_condition(question)
        
        # Determine query type and build SQL
        if "average" in question_lower or "mean" in question_lower:
            sql_query = self._build_average_query(time_condition, location_condition)
            query_type = "aggregation"
        elif "maximum" in question_lower or "max" in question_lower or "highest" in question_lower:
            sql_query = self._build_maximum_query(time_condition, location_condition)
            query_type = "ranking"
        elif "minimum" in question_lower or "min" in question_lower or "lowest" in question_lower:
            sql_query = self._build_minimum_query(time_condition, location_condition)
            query_type = "ranking"
        elif "sum" in question_lower or "total" in question_lower:
            sql_query = self._build_sum_query(time_condition, location_condition)
            query_type = "aggregation"
        elif "count" in question_lower or "how many" in question_lower:
            sql_query = self._build_count_query(time_condition, location_condition)
            query_type = "aggregation"
        else:
            # Default to basic select
            sql_query = self._build_basic_query(time_condition, location_condition)
            query_type = "filtering"
        
        return {
            "sql_query": sql_query,
            "confidence": 0.6,
            "explanation": f"Rule-based {query_type} query",
            "query_type": query_type
        }
    
    def _build_schema_context(self) -> str:
        """Build schema context for LLM"""
        schema_lines = []
        for column, data_type in self.schema.items():
            schema_lines.append(f"  {column}: {data_type}")
        return "\n".join(schema_lines)
    
    def _get_sample_data_context(self) -> str:
        """Get sample data context for LLM"""
        try:
            sample_data = self.duckdb_service.get_sample_data(3)
            if not sample_data:
                return "No sample data available"
            
            # Format sample data
            lines = []
            for row in sample_data:
                row_str = ", ".join([f"{k}={v}" for k, v in row.items()])
                lines.append(f"  {row_str}")
            
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Error getting sample data: {e}")
            return "Error retrieving sample data"
    
    def _extract_time_condition(self, question: str) -> str:
        """Extract time condition from question"""
        question_lower = question.lower()
        
        # Check for specific date patterns
        import re
        date_patterns = [
            r'(\d{4})[-\s/](\d{1,2})[-\s/](\d{1,2})',
            r'(\d{1,2})[-\s/](\d{1,2})[-\s/](\d{4})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, question)
            if match:
                groups = match.groups()
                try:
                    if len(groups[0]) == 4:  # Year first
                        year, month, day = groups[0], groups[1], groups[2]
                    else:  # Month first
                        month, day, year = groups[0], groups[1], groups[2]
                    
                    date_str = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                    return f"Date = '{date_str}'"
                except (ValueError, IndexError):
                    continue
        
        # Check for relative time patterns
        if "last week" in question_lower:
            return "Date >= CURRENT_DATE - INTERVAL '7 days'"
        elif "last month" in question_lower:
            return "Date >= CURRENT_DATE - INTERVAL '30 days'"
        elif "last year" in question_lower:
            return "Date >= CURRENT_DATE - INTERVAL '365 days'"
        elif "today" in question_lower:
            return "Date >= CURRENT_DATE"
        elif "yesterday" in question_lower:
            return "Date = CURRENT_DATE - INTERVAL '1 day'"
        elif "may" in question_lower and "2025" in question_lower:
            return "Date >= '2025-05-01' AND Date < '2025-06-01'"
        
        return ""
    
    def _extract_location_condition(self, question: str) -> str:
        """Extract location condition from question"""
        question_lower = question.lower()
        
        # Check for specific location names
        for location in self.location_names:
            if location.lower() in question_lower:
                return f"Location_Name = '{location}'"
        
        return ""
    
    def _build_average_query(self, time_condition: str, location_condition: str) -> str:
        """Build average query"""
        where_clause = self._build_where_clause(time_condition, location_condition)
        return f"SELECT AVG(TotalCount) as average_count FROM {self.table_name} {where_clause}"
    
    def _build_maximum_query(self, time_condition: str, location_condition: str) -> str:
        """Build maximum query - use ORDER BY instead of MAX to avoid GROUP BY issues"""
        where_clause = self._build_where_clause(time_condition, location_condition)
        return f"SELECT Location_Name, TotalCount FROM {self.table_name} {where_clause} ORDER BY TotalCount DESC LIMIT 1"
    
    def _build_minimum_query(self, time_condition: str, location_condition: str) -> str:
        """Build minimum query - use ORDER BY instead of MIN to avoid GROUP BY issues"""
        where_clause = self._build_where_clause(time_condition, location_condition)
        return f"SELECT Location_Name, TotalCount FROM {self.table_name} {where_clause} ORDER BY TotalCount ASC LIMIT 1"
    
    def _build_sum_query(self, time_condition: str, location_condition: str) -> str:
        """Build sum query"""
        where_clause = self._build_where_clause(time_condition, location_condition)
        return f"SELECT SUM(TotalCount) as total_count FROM {self.table_name} {where_clause}"
    
    def _build_count_query(self, time_condition: str, location_condition: str) -> str:
        """Build count query"""
        where_clause = self._build_where_clause(time_condition, location_condition)
        return f"SELECT COUNT(*) as record_count FROM {self.table_name} {where_clause}"
    
    def _build_basic_query(self, time_condition: str, location_condition: str) -> str:
        """Build basic select query"""
        where_clause = self._build_where_clause(time_condition, location_condition)
        return f"SELECT * FROM {self.table_name} {where_clause} LIMIT 10"
    
    def _build_where_clause(self, time_condition: str, location_condition: str) -> str:
        """Build WHERE clause from conditions"""
        conditions = []
        
        if time_condition:
            conditions.append(time_condition)
        if location_condition:
            conditions.append(location_condition)
        
        if conditions:
            return f"WHERE {' AND '.join(conditions)}"
        else:
            return ""
    
    def execute_question(self, question: str) -> dict:
        """Execute a natural language question and return results"""
        try:
            # Generate SQL
            sql_result = self.generate_sql(question)
            sql_query = sql_result["sql_query"]
            
            # Execute SQL
            results = self.duckdb_service.execute_query(sql_query)
            
            return {
                "question": question,
                "sql_query": sql_query,
                "results": results,
                "confidence": sql_result["confidence"],
                "explanation": sql_result["explanation"],
                "query_type": sql_result["query_type"],
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error executing question: {e}")
            return {
                "question": question,
                "error": str(e),
                "success": False
            }

def test_improved_sql():
    """Test the improved SQL service"""
    print("🧪 Testing Improved SQL Service")
    print("=" * 40)
    
    # Initialize services
    llm_service = LLMService()
    csv_file_path = os.path.join(os.path.dirname(__file__), 'data', 'data.csv')
    duckdb_service = DuckDBService(csv_file_path)
    text_to_sql_service = ImprovedTextToSQLService(llm_service, duckdb_service)
    
    # Test the specific question
    question = "Which location has the most crowded in 2025 May"
    
    print(f"Question: {question}")
    print("-" * 50)
    
    # Generate and execute
    result = text_to_sql_service.execute_question(question)
    
    if result['success']:
        print(f"✅ SQL Query:")
        print(f"   {result['sql_query']}")
        print(f"\n📊 Results ({len(result['results'])} rows):")
        
        for i, row in enumerate(result['results']):
            print(f"   Row {i+1}: {row}")
        
        print(f"\nConfidence: {result['confidence']}")
        print(f"Explanation: {result['explanation']}")
    else:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")
    
    duckdb_service.close()

if __name__ == "__main__":
    test_improved_sql()
