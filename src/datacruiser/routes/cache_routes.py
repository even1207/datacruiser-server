"""
Cache management routes
"""

from flask import Blueprint, jsonify
import logging

from ..utils.cache_utils import CacheManager

logger = logging.getLogger(__name__)

cache_bp = Blueprint('cache', __name__)


@cache_bp.route("/clear", methods=["POST"])
def clear_cache():
    """Clear cache endpoint"""
    try:
        cache_manager = CacheManager()
        removed_files = cache_manager.clear_cache()
        
        logger.info(f"🗑️ Cleared cache files: {removed_files}")
        
        return jsonify({
            "message": "Cache cleared successfully",
            "removed_files": removed_files,
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return jsonify({"error": "Failed to clear cache", "success": False}), 500


@cache_bp.route("/status", methods=["GET"])
def cache_status():
    """Get cache status"""
    try:
        cache_manager = CacheManager()
        status = cache_manager.get_cache_status()
        
        return jsonify({
            **status,
            "success": True
        })
        
    except Exception as e:
        logger.error(f"Error getting cache status: {e}")
        return jsonify({"error": "Failed to get cache status", "success": False}), 500
