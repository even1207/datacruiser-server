"""
Model classes for the RAG-based Footfall Analysis API
"""

from .data_models import FootfallRecord, DataStats, QueryParams
from .embedding_models import EmbeddingModel, TimesFMModel, FallbackEmbeddingModel

__all__ = [
    'FootfallRecord',
    'DataStats', 
    'QueryParams',
    'EmbeddingModel',
    'TimesFMModel',
    'FallbackEmbeddingModel'
]
