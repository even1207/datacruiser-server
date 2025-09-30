"""
LLM service for generating responses
"""

import os
from typing import List
import logging

from openai import OpenAI
from ..models.data_models import FootfallRecord

logger = logging.getLogger(__name__)


class LLMService:
    """Service for LLM operations"""
    
    def __init__(self):
        self.client: OpenAI = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                logger.warning("OpenAI API key not configured")
                return
            
            self.client = OpenAI(api_key=openai_api_key)
            logger.info("✅ OpenAI client initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
    
    def generate_response(self, query: str, similar_records: List[FootfallRecord]) -> str:
        """Generate LLM response with error handling"""
        try:
            if self.client is None:
                return "OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file."
            
            # Check if no records found
            if not similar_records:
                return "I'm sorry, I couldn't find any records matching your criteria. This could be because:\n1. No data exists for the specified date\n2. No data exists for the specified location\n3. The date format might not be recognized\n\nPlease try rephrasing your question or checking if the date exists in our database."
            
            # Build context from similar records
            context_parts = []
            for i, record in enumerate(similar_records):
                context_parts.append(
                    f"{i+1}. {record.location_name} ({record.date}): "
                    f"TotalCount={record.total_count}, "
                    f"LastWeek={record.last_week}, "
                    f"LastYear={record.last_year}, "
                    f"Similarity={record.similarity_score:.3f}"
                )
            
            context = "\n".join(context_parts)
            
            # Analyze the records to provide better context
            dates = [record.date.split(' ')[0] for record in similar_records]
            locations = [record.location_name for record in similar_records]
            unique_dates = set(dates)
            unique_locations = set(locations)
            
            # Build context information
            context_info = []
            if len(unique_dates) == 1:
                context_info.append(f"All records are from {list(unique_dates)[0]}")
            elif len(unique_dates) <= 3:
                context_info.append(f"Records from {len(unique_dates)} dates: {', '.join(sorted(unique_dates))}")
            else:
                context_info.append(f"Records span {len(unique_dates)} different dates")
            
            if len(unique_locations) == 1:
                context_info.append(f"All records are from {list(unique_locations)[0]}")
            else:
                context_info.append(f"Records from {len(unique_locations)} different locations")
            
            context_summary = "\n".join(context_info)
            
            prompt = f"""
You are a professional footfall data analysis assistant. User query: "{query}"

Context: {context_summary}

Here are the retrieved relevant historical records:
{context}

Instructions:
1. Answer the user's question based on this data
2. Focus on TotalCount as the main indicator of footfall
3. If the user asked about "busiest locations", rank them by TotalCount
4. If the user asked about a specific date, only consider records from that date
5. If comparing locations, ensure you're comparing records from the same time period
6. Be professional but easy to understand
7. If the data doesn't fully answer the question, explain what insights can be drawn

Answer in English.
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            return self._generate_fallback_response(similar_records)
    
    def _generate_fallback_response(self, similar_records: List[FootfallRecord]) -> str:
        """Generate fallback response when LLM fails"""
        if not similar_records:
            return "I'm sorry, I couldn't find any relevant data to answer your question."
        
        locations = [record.location_name for record in similar_records[:3]]
        return f"Based on the available data, here's what I found: {len(similar_records)} relevant records were retrieved. The locations include {', '.join(locations)}. Please check your OpenAI API configuration for detailed analysis."
    
    def is_available(self) -> bool:
        """Check if LLM service is available"""
        return self.client is not None

    def generate_dataset_response(self, zero_shot_prompt: str, cot_prompt: str) -> str:
        """Generate an answer for uploaded datasets using structured prompts."""
        if self.client is None:
            return (
                "OpenAI API key not configured. Returning baseline analysis only."
            )

        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a senior data analyst. Use the provided dataset context to"
                        " answer questions accurately and concisely."
                    )
                },
                {"role": "user", "content": zero_shot_prompt},
                {"role": "assistant", "content": "I will analyse the evidence step by step."},
                {"role": "user", "content": cot_prompt}
            ]

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.2,
                max_tokens=700
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error generating dataset response: {e}")
            return "I could not complete the analysis due to an LLM error."
