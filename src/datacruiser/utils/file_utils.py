"""
File handling utilities
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple
import logging

from ..config import Config
from ..models.data_models import FootfallRecord, DataStats

logger = logging.getLogger(__name__)


class FileUtils:
    """File handling utilities"""
    
    @staticmethod
    def load_json_data(file_path: str) -> List[Dict[str, Any]]:
        """Load JSON data from file"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"JSON file not found: {file_path}")

            logger.info("📂 Loading JSON data from file...")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            logger.info(f"✅ Loaded {len(data)} records from JSON")
            return data

        except Exception as e:
            logger.error(f"❌ Failed to load JSON data: {e}")
            raise
    
    @staticmethod
    def calculate_data_statistics(records: List[Dict[str, Any]]) -> DataStats:
        """Calculate data statistics with safety checks"""
        try:
            total_counts = []
            last_weeks = []
            prev_4days = []
            last_years = []
            prev_52days = []

            for record in records[:1000]:  # Limit to first 1000 for safety
                try:
                    total_counts.append(float(record.get("TotalCount", 0)))
                    last_weeks.append(float(record.get("LastWeek", 0)))
                    prev_4days.append(float(record.get("Previous4DayTimeAvg", 0)))
                    last_years.append(float(record.get("LastYear", 0)))
                    prev_52days.append(float(record.get("Previous52DayTimeAvg", 0)))
                except:
                    continue

            if total_counts:
                stats_dict = {
                    "TotalCount": {
                        "min": np.min(total_counts),
                        "max": np.max(total_counts),
                        "mean": np.mean(total_counts),
                        "median": np.median(total_counts)
                    },
                    "LastWeek": {
                        "min": np.min(last_weeks) if last_weeks else 0,
                        "max": np.max(last_weeks) if last_weeks else 0,
                        "mean": np.mean(last_weeks) if last_weeks else 0,
                        "median": np.median(last_weeks) if last_weeks else 0
                    },
                    "Previous4DayTimeAvg": {
                        "min": np.min(prev_4days) if prev_4days else 0,
                        "max": np.max(prev_4days) if prev_4days else 0,
                        "mean": np.mean(prev_4days) if prev_4days else 0,
                        "median": np.median(prev_4days) if prev_4days else 0
                    },
                    "LastYear": {
                        "min": np.min(last_years) if last_years else 0,
                        "max": np.max(last_years) if last_years else 0,
                        "mean": np.mean(last_years) if last_years else 0,
                        "median": np.median(last_years) if last_years else 0
                    },
                    "Previous52DayTimeAvg": {
                        "min": np.min(prev_52days) if prev_52days else 0,
                        "max": np.max(prev_52days) if prev_52days else 0,
                        "mean": np.mean(prev_52days) if prev_52days else 0,
                        "median": np.median(prev_52days) if prev_52days else 0
                    }
                }

                logger.info(f"📊 Data statistics calculated: TotalCount range [{stats_dict['TotalCount']['min']:.0f}, {stats_dict['TotalCount']['max']:.0f}], mean: {stats_dict['TotalCount']['mean']:.0f}")
                return DataStats.from_dict(stats_dict)
            else:
                logger.warning("No valid data for statistics calculation")
                return DataStats.from_dict({})

        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return DataStats.from_dict({})
    
    @staticmethod
    def process_records(records: List[Dict[str, Any]]) -> Tuple[List[FootfallRecord], DataStats]:
        """Process raw records into FootfallRecord objects and calculate statistics"""
        try:
            # Limit records for safety
            max_records = min(len(records), Config.MAX_RECORDS)
            if len(records) > max_records:
                logger.info(f"⚠️ Limiting to first {max_records} records for stability")
                records = records[:max_records]
            
            # Convert to FootfallRecord objects
            footfall_records = []
            for record in records:
                try:
                    footfall_record = FootfallRecord.from_dict(record)
                    footfall_records.append(footfall_record)
                except Exception as e:
                    logger.warning(f"Error processing record: {e}")
                    continue
            
            # Calculate statistics
            stats = FileUtils.calculate_data_statistics(records)
            
            return footfall_records, stats

        except Exception as e:
            logger.error(f"Error processing records: {e}")
            return [], DataStats.from_dict({})
