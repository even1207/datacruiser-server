"""
RAG service for retrieval and similarity search
"""

import json
import numpy as np
import faiss
import random
import gc
import torch
import re
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import logging

from ..config import Config
from ..models.data_models import FootfallRecord, QueryParams, DataStats
from ..utils.cache_utils import CacheManager
from ..utils.file_utils import FileUtils
from .model_service import ModelService

logger = logging.getLogger(__name__)


class RAGService:
    """Service for RAG operations"""
    
    def __init__(self, model_service: ModelService, llm_service=None):
        self.model_service = model_service
        self.llm_service = llm_service  # Add LLM service for question analysis
        self.cache_manager = CacheManager()
        self.records: List[FootfallRecord] = []
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.data_stats: Optional[DataStats] = None
    
    def initialize_data(self) -> bool:
        """Initialize data with aggressive caching"""
        # Check if we can load everything from cache
        if self.cache_manager.is_cache_valid():
            logger.info("🚀 Loading all data from cache...")
            
            if self._load_all_from_cache():
                logger.info("✅ All data loaded from cache successfully!")
                return True
            else:
                logger.warning("⚠️ Partial cache load failed, will regenerate")
        
        try:
            # Load and process data
            if not self._load_and_process_data():
                return False
            
            # Generate embeddings
            if not self._generate_embeddings():
                return False
            
            # Create FAISS index
            if not self._create_faiss_index():
                return False
            
            # Save cache metadata
            self.cache_manager.save_cache_metadata()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize data: {str(e)}")
            return False
    
    def _load_all_from_cache(self) -> bool:
        """Load all data from cache"""
        try:
            # Load data
            records_data, data_stats = self.cache_manager.load_data_from_cache()
            if records_data is None or data_stats is None:
                return False
            
            # Convert dict records to FootfallRecord objects
            self.records = [FootfallRecord.from_dict(record_dict) for record_dict in records_data]
            self.data_stats = data_stats
            
            # Load embeddings
            embeddings = self.cache_manager.load_embeddings_from_cache()
            if embeddings is None:
                return False
            
            self.embeddings = embeddings
            
            # Load FAISS index
            faiss_index = self.cache_manager.load_faiss_index_from_cache()
            if faiss_index is None:
                return False
            
            self.faiss_index = faiss_index
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading from cache: {e}")
            return False
    
    def _load_and_process_data(self) -> bool:
        """Load and process data"""
        try:
            # Load JSON data
            json_path = Config.DATA_FILE_PATH
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"JSON file not found: {json_path}")
            
            all_records = FileUtils.load_json_data(json_path)
            
            # Process records
            self.records, self.data_stats = FileUtils.process_records(all_records)
            
            # Save processed data to cache
            self.cache_manager.save_data_to_cache(
                [record.to_dict() for record in self.records],
                self.data_stats.to_dict()
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading and processing data: {e}")
            return False
    
    def _generate_embeddings(self) -> bool:
        """Generate embeddings for all records"""
        try:
            logger.info("🔮 Generating embeddings...")
            embeddings_list = []
            batch_size = Config.BATCH_SIZE
            
            for i in range(0, len(self.records), batch_size):
                batch_end = min(i + batch_size, len(self.records))
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(self.records) + batch_size - 1)//batch_size} (records {i}-{batch_end})")
                
                batch_embeddings = []
                for j in range(i, batch_end):
                    try:
                        record = self.records[j]
                        series = record.get_numerical_features()
                        
                        # Handle NaN values
                        series = np.nan_to_num(series, nan=0.0)
                        
                        # Generate embedding
                        emb = self.model_service.generate_embedding(series)
                        batch_embeddings.append(emb)
                        
                        # Force garbage collection every few records
                        if j % 50 == 0:
                            gc.collect()
                            
                    except Exception as e:
                        logger.warning(f"Error processing record {j}: {str(e)}")
                        # Use deterministic fallback
                        fallback_emb = np.random.RandomState(j).normal(0, 1, Config.EMBEDDING_DIMENSION).astype(np.float32)
                        batch_embeddings.append(fallback_emb)
                
                embeddings_list.extend(batch_embeddings)
                
                # Clear memory after each batch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            self.embeddings = np.stack(embeddings_list).astype("float32")
            logger.info(f"✅ Generated embeddings: {self.embeddings.shape}")
            
            # Save embeddings to cache
            self.cache_manager.save_embeddings_to_cache(self.embeddings)
            
            return True
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return False
    
    def _create_faiss_index(self) -> bool:
        """Create FAISS index"""
        try:
            logger.info("🔍 Creating FAISS index...")
            dim = self.embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(self.embeddings)
            
            logger.info(f"✅ FAISS index created with {self.faiss_index.ntotal} vectors")
            
            # Save FAISS index to cache
            self.cache_manager.save_faiss_index_to_cache(self.faiss_index)
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating FAISS index: {e}")
            return False
    
    def search_similar_records(self, query_params: QueryParams, top_k: int = 5) -> List[FootfallRecord]:
        """Search for similar records"""
        if self.faiss_index is None or not self.model_service.is_initialized():
            logger.warning("System not properly initialized, using fallback")
            return self._get_random_records_fallback(top_k)
        
        try:
            query_series = query_params.get_numerical_features()
            query_series = np.nan_to_num(query_series, nan=0.0)
            
            # Generate query embedding
            q_emb = self.model_service.generate_embedding(query_series)
            q_emb = q_emb.astype("float32").reshape(1, -1)
            
            # Search
            scores, ids = self.faiss_index.search(q_emb, top_k)
            
            similar_records = []
            for i, record_id in enumerate(ids[0]):
                if 0 <= record_id < len(self.records):
                    record = self.records[record_id]
                    record.similarity_score = float(scores[0][i])
                    similar_records.append(record)
            
            logger.info(f"✅ Found {len(similar_records)} similar records")
            return similar_records
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return self._get_random_records_fallback(top_k)
    
    def _get_random_records_fallback(self, top_k: int = 5) -> List[FootfallRecord]:
        """Safe random fallback"""
        if not self.records:
            return []
        
        try:
            random_records = random.sample(self.records, min(top_k, len(self.records)))
            for record in random_records:
                # Ensure record is a FootfallRecord object, not a dict
                if isinstance(record, dict):
                    record = FootfallRecord.from_dict(record)
                record.similarity_score = 0.5
            
            logger.info(f"🎲 Using random fallback: returned {len(random_records)} records")
            return random_records
            
        except Exception as e:
            logger.error(f"Error in fallback: {e}")
            return []
    
    def get_default_query_params(self) -> QueryParams:
        """Get default query parameters"""
        if self.data_stats:
            # Check if data_stats is a DataStats object or a dict
            if hasattr(self.data_stats, 'get_default_params'):
                default_dict = self.data_stats.get_default_params()
            else:
                # data_stats is a dict loaded from cache
                default_dict = {
                    "TotalCount": self.data_stats.get('TotalCount', {}).get('median', 1500),
                    "LastWeek": self.data_stats.get('LastWeek', {}).get('median', 1500),
                    "Previous4DayTimeAvg": self.data_stats.get('Previous4DayTimeAvg', {}).get('median', 1500),
                    "LastYear": self.data_stats.get('LastYear', {}).get('median', 1500),
                    "Previous52DayTimeAvg": self.data_stats.get('Previous52DayTimeAvg', {}).get('median', 1500)
                }
        else:
            default_dict = {
                "TotalCount": 1500,
                "LastWeek": 1500,
                "Previous4DayTimeAvg": 1500,
                "LastYear": 1500,
                "Previous52DayTimeAvg": 1500
            }
        
        return QueryParams.from_dict(default_dict)
    
    def is_initialized(self) -> bool:
        """Check if RAG service is initialized"""
        return (len(self.records) > 0 and 
                self.embeddings is not None and 
                self.faiss_index is not None)
    
    def analyze_question_intent(self, question: str) -> Dict[str, Any]:
        """Analyze question to understand user intent and requirements"""
        try:
            # Use LLM to analyze the question
            analysis_prompt = f"""
Analyze this footfall data question and extract the key requirements:

Question: "{question}"

Please analyze and respond with a JSON object containing:
1. "question_type": What type of question is this? (e.g., "busiest_locations", "comparison", "trend_analysis", "specific_date", "general_analysis")
2. "time_scope": What time scope is being asked about? (e.g., "specific_date", "date_range", "general", "latest", "historical")
3. "target_date": If a specific date is mentioned, extract it in YYYY/MM/DD format, otherwise null
4. "location_focus": Is the question about specific locations or general? (e.g., "specific_location", "all_locations", "location_comparison")
5. "metric_focus": What metric is most important? (e.g., "TotalCount", "LastWeek", "LastYear", "comparison")
6. "requires_same_date": Should the answer focus on records from the same date? (true/false)
7. "analysis_type": What kind of analysis is needed? (e.g., "ranking", "comparison", "summary", "trend")

Respond with valid JSON only, no additional text.
"""

            if hasattr(self, 'llm_service') and self.llm_service and self.llm_service.is_available():
                response = self.llm_service.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": analysis_prompt}],
                    temperature=0.1,
                    max_tokens=300
                )
                
                analysis_text = response.choices[0].message.content.strip()
                # Clean up the response to ensure it's valid JSON
                if analysis_text.startswith("```json"):
                    analysis_text = analysis_text[7:]
                if analysis_text.endswith("```"):
                    analysis_text = analysis_text[:-3]
                
                analysis = json.loads(analysis_text)
                logger.info(f"Question analysis: {analysis}")
                return analysis
            else:
                # Fallback analysis without LLM
                return self._fallback_question_analysis(question)
                
        except Exception as e:
            logger.error(f"Error analyzing question: {e}")
            return self._fallback_question_analysis(question)
    
    def _fallback_question_analysis(self, question: str) -> Dict[str, Any]:
        """Fallback question analysis without LLM"""
        question_lower = question.lower()
        
        # Simple keyword-based analysis
        if any(word in question_lower for word in ['busiest', 'busy', 'highest', 'top']):
            question_type = "busiest_locations"
        elif any(word in question_lower for word in ['compare', 'comparison', 'vs', 'versus']):
            question_type = "comparison"
        elif any(word in question_lower for word in ['trend', 'change', 'increase', 'decrease']):
            question_type = "trend_analysis"
        else:
            question_type = "general_analysis"
        
        # Check for specific dates
        date_patterns = [
            r'(\d{4})[-\s/](\d{1,2})[-\s/](\d{1,2})',
            r'(\d{1,2})[-\s/](\d{1,2})[-\s/](\d{4})'
        ]
        
        target_date = None
        for pattern in date_patterns:
            match = re.search(pattern, question)
            if match:
                groups = match.groups()
                try:
                    if len(groups[0]) == 4:  # Year first
                        year, month, day = groups[0], groups[1], groups[2]
                    else:  # Month first
                        month, day, year = groups[0], groups[1], groups[2]
                    
                    normalized_date = f"{year}/{month.zfill(2)}/{day.zfill(2)}"
                    datetime.strptime(normalized_date, "%Y/%m/%d")
                    target_date = normalized_date
                    break
                except (ValueError, IndexError):
                    continue
        
        return {
            "question_type": question_type,
            "time_scope": "specific_date" if target_date else "general",
            "target_date": target_date,
            "location_focus": "all_locations",
            "metric_focus": "TotalCount",
            "requires_same_date": target_date is not None,
            "analysis_type": "ranking" if question_type == "busiest_locations" else "summary"
        }
    
    def filter_records_by_date(self, records: List[FootfallRecord], target_date: str) -> List[FootfallRecord]:
        """Filter records by specific date"""
        if not target_date:
            return records
        
        filtered_records = []
        for record in records:
            # Extract date part from the full datetime string
            record_date = record.date.split(' ')[0]  # Get YYYY/MM/DD part
            if record_date == target_date:
                filtered_records.append(record)
        
        return filtered_records
    
    def search_similar_records_with_context(self, question: str, top_k: int = 5) -> List[FootfallRecord]:
        """Search for similar records with intelligent context-aware filtering"""
        if self.faiss_index is None or not self.model_service.is_initialized():
            logger.warning("System not properly initialized, using fallback")
            return self._get_random_records_fallback(top_k)
        
        try:
            # First, understand the question intent
            question_analysis = self.analyze_question_intent(question)
            logger.info(f"Question analysis: {question_analysis}")
            
            # Get default query params
            query_params = self.get_default_query_params()
            query_series = query_params.get_numerical_features()
            query_series = np.nan_to_num(query_series, nan=0.0)
            
            # Generate query embedding
            q_emb = self.model_service.generate_embedding(query_series)
            q_emb = q_emb.astype("float32").reshape(1, -1)
            
            # Determine search strategy based on question analysis
            if question_analysis.get("question_type") == "busiest_locations":
                # For busiest locations, we need more diverse records to compare
                search_k = min(top_k * 5, 100)
            else:
                search_k = min(top_k * 3, 50)
            
            scores, ids = self.faiss_index.search(q_emb, search_k)
            
            # Get candidate records
            candidate_records = []
            for i, record_id in enumerate(ids[0]):
                if 0 <= record_id < len(self.records):
                    record = self.records[record_id]
                    record.similarity_score = float(scores[0][i])
                    candidate_records.append(record)
            
            # Apply intelligent filtering based on question analysis
            filtered_records = self._apply_intelligent_filtering(candidate_records, question_analysis)
            
            if not filtered_records:
                logger.warning(f"No records found matching question criteria")
                return []
            
            # Sort and return top_k
            filtered_records.sort(key=lambda x: x.similarity_score, reverse=True)
            similar_records = filtered_records[:top_k]
            
            logger.info(f"✅ Found {len(similar_records)} relevant records")
            return similar_records
            
        except Exception as e:
            logger.error(f"Error in context-aware similarity search: {str(e)}")
            return self._get_random_records_fallback(top_k)
    
    def _apply_intelligent_filtering(self, records: List[FootfallRecord], analysis: Dict[str, Any]) -> List[FootfallRecord]:
        """Apply intelligent filtering based on question analysis"""
        filtered_records = records.copy()
        
        # Filter by specific date if required
        if analysis.get("requires_same_date") and analysis.get("target_date"):
            target_date = analysis["target_date"]
            filtered_records = self.filter_records_by_date(filtered_records, target_date)
            logger.info(f"After date filtering: {len(filtered_records)} records")
        
        # For busiest locations questions, ensure we have diverse locations from the same time period
        if analysis.get("question_type") == "busiest_locations":
            filtered_records = self._ensure_diverse_locations(filtered_records, analysis)
        
        # For comparison questions, ensure we have comparable records
        if analysis.get("question_type") == "comparison":
            filtered_records = self._ensure_comparable_records(filtered_records, analysis)
        
        return filtered_records
    
    def _ensure_diverse_locations(self, records: List[FootfallRecord], analysis: Dict[str, Any]) -> List[FootfallRecord]:
        """Ensure we have diverse locations for comparison"""
        if not records:
            return records
        
        # Group by location
        location_groups = {}
        for record in records:
            if record.location_name not in location_groups:
                location_groups[record.location_name] = []
            location_groups[record.location_name].append(record)
        
        # If we have a specific date, get the best record per location for that date
        if analysis.get("requires_same_date") and analysis.get("target_date"):
            diverse_records = []
            for location, location_records in location_groups.items():
                # Get the record with highest TotalCount for this location
                best_record = max(location_records, key=lambda x: x.total_count)
                diverse_records.append(best_record)
            return diverse_records
        
        # Otherwise, get diverse locations with good similarity scores
        diverse_records = []
        for location, location_records in location_groups.items():
            # Get the record with best similarity score for this location
            best_record = max(location_records, key=lambda x: x.similarity_score)
            diverse_records.append(best_record)
        
        return diverse_records
    
    def _ensure_comparable_records(self, records: List[FootfallRecord], analysis: Dict[str, Any]) -> List[FootfallRecord]:
        """Ensure we have comparable records for comparison questions"""
        if not records:
            return records
        
        # For comparison, we want records from similar time periods
        if analysis.get("requires_same_date"):
            return records  # Already filtered by date
        
        # Group by date and get representative records
        date_groups = {}
        for record in records:
            date_key = record.date.split(' ')[0]  # Get date part
            if date_key not in date_groups:
                date_groups[date_key] = []
            date_groups[date_key].append(record)
        
        # Get best record from each date group
        comparable_records = []
        for date, date_records in date_groups.items():
            best_record = max(date_records, key=lambda x: x.similarity_score)
            comparable_records.append(best_record)
        
        return comparable_records
