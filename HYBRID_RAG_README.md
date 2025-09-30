# Hybrid RAG System for Crowd Flow Analysis

This document describes the hybrid RAG (Retrieval-Augmented Generation) system that supports both **statistical queries** and **trend analysis** for crowd flow data.

## Overview

The hybrid system intelligently routes queries to the most appropriate processing method:

- **Statistical Queries**: Handled by DuckDB + Text-to-SQL for precise numerical analysis
- **Trend Queries**: Handled by FAISS + TimesFM embeddings for semantic similarity and pattern analysis

## Architecture

```
User Question
     ↓
Query Classifier (LLM-based or rule-based)
     ↓
┌─────────────────┬─────────────────┐
│ Statistical     │ Trend           │
│ Queries         │ Queries         │
│                 │                 │
│ DuckDB          │ FAISS +         │
│ + Text-to-SQL   │ TimesFM         │
│                 │ + RAG           │
└─────────────────┴─────────────────┘
     ↓
Unified Response Generator
     ↓
Natural Language Answer
```

## Features

### Statistical Query Support
- **Text-to-SQL**: Converts natural language to SQL queries
- **DuckDB Integration**: Direct CSV querying with proper type casting
- **Aggregation Support**: Averages, sums, counts, min/max operations
- **Time-based Filtering**: Date range queries and relative time periods
- **Location Filtering**: Specific location-based queries

### Trend Query Support
- **Semantic Search**: FAISS-based similarity search
- **Time Series Analysis**: TimesFM embeddings for temporal patterns
- **Pattern Recognition**: Identifies trends and changes over time
- **Comparative Analysis**: Location-to-location comparisons

## API Endpoints

### New Hybrid Endpoint
```http
POST /api/ask-hybrid
Content-Type: application/json

{
  "question": "What was the average crowd count last week?",
  "top_k": 5
}
```

### System Information
```http
GET /api/system-info
```

## Query Classification

The system automatically classifies queries as:

### Statistical Queries
- Keywords: average, sum, count, maximum, minimum, total, etc.
- Examples:
  - "What was the average crowd count in the past 5 days?"
  - "Which location had the most people last week?"
  - "How many records do we have for Market Street?"

### Trend Queries
- Keywords: trend, compare, pattern, change, over time, etc.
- Examples:
  - "Compare the recent trend of Park Street vs Market Street"
  - "How has the crowd flow changed over time?"
  - "What patterns do you see in the data?"

## Data Schema

The system works with CSV data having the following schema (all fields stored as TEXT, properly cast in DuckDB):

| Field | Type | Description |
|-------|------|-------------|
| Location_code | TEXT | Location identifier |
| Location_Name | TEXT | Human-readable location name |
| Date | TIMESTAMP | Date and time (YYYY/MM/DD HH:MM:SS+00) |
| TotalCount | INTEGER | Main crowd count metric |
| Hour | INTEGER | Hour of day |
| Day | TEXT | Day name |
| DayNo | INTEGER | Day number |
| Week | TEXT | Week identifier |
| LastWeek | INTEGER | Previous week count |
| Previous4DayTimeAvg | INTEGER | 4-day average |
| ObjectId | INTEGER | Unique record ID |
| LastYear | INTEGER | Same time last year |
| Previous52DayTimeAvg | INTEGER | 52-day average |

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp env_template.txt .env
# Edit .env with your OpenAI API key
```

3. Run the server:
```bash
python run.py
```

## Usage Examples

### Statistical Queries

```python
# Average crowd count
{
  "question": "What was the average crowd count in the past 5 days?"
}

# Location comparison
{
  "question": "Which location had the most people last week?"
}

# Specific location query
{
  "question": "What is the total count for Park Street today?"
}
```

### Trend Queries

```python
# Location comparison
{
  "question": "Compare the recent trend of Park Street vs Market Street"
}

# Pattern analysis
{
  "question": "What patterns do you see in the data?"
}

# Time-based trends
{
  "question": "How has the crowd flow changed over time?"
}
```

## Testing

Run the test script to verify functionality:

```bash
python test_hybrid_rag.py
```

Or run the example:

```bash
python example_hybrid_usage.py
```

## Configuration

The system uses the following configuration:

- **DuckDB**: In-memory database for statistical queries
- **FAISS**: Vector similarity search for trend analysis
- **TimesFM**: Time series embeddings
- **OpenAI GPT-4o-mini**: LLM for query classification and response generation

## Performance

- **Statistical Queries**: Fast SQL execution on indexed data
- **Trend Queries**: Optimized with FAISS indexing and caching
- **Hybrid Routing**: Minimal overhead with intelligent classification

## Error Handling

The system includes comprehensive error handling:

- **Fallback Classification**: Rule-based classification when LLM is unavailable
- **SQL Validation**: Basic SQL query validation before execution
- **Graceful Degradation**: Fallback responses when services fail
- **Detailed Logging**: Comprehensive logging for debugging

## Future Enhancements

- **Query Caching**: Cache frequent queries for improved performance
- **Advanced Analytics**: More sophisticated statistical functions
- **Real-time Updates**: Live data streaming capabilities
- **Custom Models**: Fine-tuned models for specific use cases
