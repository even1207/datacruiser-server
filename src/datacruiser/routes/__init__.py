"""
Route handlers for the RAG-based Footfall Analysis API
"""

from .api_routes import api_bp
from .health_routes import health_bp
from .cache_routes import cache_bp
from .air_quality_routes import air_quality_bp

__all__ = [
    'api_bp',
    'health_bp',
    'cache_bp',
    'air_quality_bp'
]
