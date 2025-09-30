"""
Data models for footfall analysis
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class FootfallRecord:
    """Represents a single footfall data record"""
    
    location_name: str
    date: str
    total_count: float
    last_week: float
    previous_4day_time_avg: float
    last_year: float
    previous_52day_time_avg: float
    similarity_score: Optional[float] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FootfallRecord':
        """Create FootfallRecord from dictionary"""
        return cls(
            location_name=data.get('Location_Name', ''),
            date=data.get('Date', ''),
            total_count=float(data.get('TotalCount', 0)),
            last_week=float(data.get('LastWeek', 0)),
            previous_4day_time_avg=float(data.get('Previous4DayTimeAvg', 0)),
            last_year=float(data.get('LastYear', 0)),
            previous_52day_time_avg=float(data.get('Previous52DayTimeAvg', 0)),
            similarity_score=data.get('similarity_score')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'Location_Name': self.location_name,
            'Date': self.date,
            'TotalCount': self.total_count,
            'LastWeek': self.last_week,
            'Previous4DayTimeAvg': self.previous_4day_time_avg,
            'LastYear': self.last_year,
            'Previous52DayTimeAvg': self.previous_52day_time_avg,
            'similarity_score': self.similarity_score
        }
    
    def get_numerical_features(self) -> np.ndarray:
        """Get numerical features as numpy array"""
        return np.array([
            self.total_count,
            self.last_week,
            self.previous_4day_time_avg,
            self.last_year,
            self.previous_52day_time_avg
        ], dtype=np.float32)


@dataclass
class DataStats:
    """Statistics for the dataset"""
    
    total_count_stats: Dict[str, float]
    last_week_stats: Dict[str, float]
    previous_4day_stats: Dict[str, float]
    last_year_stats: Dict[str, float]
    previous_52day_stats: Dict[str, float]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataStats':
        """Create DataStats from dictionary"""
        return cls(
            total_count_stats=data.get('TotalCount', {}),
            last_week_stats=data.get('LastWeek', {}),
            previous_4day_stats=data.get('Previous4DayTimeAvg', {}),
            last_year_stats=data.get('LastYear', {}),
            previous_52day_stats=data.get('Previous52DayTimeAvg', {})
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'TotalCount': self.total_count_stats,
            'LastWeek': self.last_week_stats,
            'Previous4DayTimeAvg': self.previous_4day_stats,
            'LastYear': self.last_year_stats,
            'Previous52DayTimeAvg': self.previous_52day_stats
        }
    
    def get_default_params(self) -> Dict[str, float]:
        """Get default query parameters based on statistics"""
        return {
            'TotalCount': self.total_count_stats.get('median', 1500),
            'LastWeek': self.last_week_stats.get('median', 1500),
            'Previous4DayTimeAvg': self.previous_4day_stats.get('median', 1500),
            'LastYear': self.last_year_stats.get('median', 1500),
            'Previous52DayTimeAvg': self.previous_52day_stats.get('median', 1500)
        }


@dataclass
class QueryParams:
    """Parameters for querying the system"""
    
    total_count: float
    last_week: float
    previous_4day_time_avg: float
    last_year: float
    previous_52day_time_avg: float
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueryParams':
        """Create QueryParams from dictionary"""
        return cls(
            total_count=float(data.get('TotalCount', 0)),
            last_week=float(data.get('LastWeek', 0)),
            previous_4day_time_avg=float(data.get('Previous4DayTimeAvg', 0)),
            last_year=float(data.get('LastYear', 0)),
            previous_52day_time_avg=float(data.get('Previous52DayTimeAvg', 0))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'TotalCount': self.total_count,
            'LastWeek': self.last_week,
            'Previous4DayTimeAvg': self.previous_4day_time_avg,
            'LastYear': self.last_year,
            'Previous52DayTimeAvg': self.previous_52day_time_avg
        }
    
    def get_numerical_features(self) -> np.ndarray:
        """Get numerical features as numpy array"""
        return np.array([
            self.total_count,
            self.last_week,
            self.previous_4day_time_avg,
            self.last_year,
            self.previous_52day_time_avg
        ], dtype=np.float32)
