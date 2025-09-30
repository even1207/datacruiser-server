"""
Health check and system status routes
"""

from flask import Blueprint, jsonify, g
from typing import Dict, Any
import logging

from ..utils.cache_utils import CacheManager

logger = logging.getLogger(__name__)

health_bp = Blueprint('health', __name__)


@health_bp.route("/", methods=["GET"])
def health_check():
    """Health check endpoint"""
    try:
        # Get cache status
        cache_manager = CacheManager()
        cache_status = {
            "cache_valid": cache_manager.is_cache_valid(),
            "cache_files": {
                "data": cache_manager.is_cache_valid(),
                "embeddings": cache_manager.is_cache_valid(),
                "faiss_index": cache_manager.is_cache_valid(),
                "stats": cache_manager.is_cache_valid(),
                "metadata": cache_manager.is_cache_valid()
            }
        }
        
        # Get system info (this would need to be passed from the main app)
        # For now, return basic status
        return jsonify({
            "service": "RAG-based Footfall Analysis API (Refactored Version)",
            "status": "running",
            "cache_status": cache_status
        })
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return jsonify({
            "service": "RAG-based Footfall Analysis API (Refactored Version)",
            "status": "error",
            "error": str(e)
        }), 500


@health_bp.route("/data/info", methods=["GET"])
def data_info():
    """Data info endpoint"""
    try:
        # This would need access to the RAG service instance
        # For now, return basic info
        cache_manager = CacheManager()
        cache_info = cache_manager.get_cache_status()
        
        return jsonify({
            "cache_info": cache_info,
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Error getting data info: {e}")
        return jsonify({"error": "Failed to get data info", "success": False}), 500
