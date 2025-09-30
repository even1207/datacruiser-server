"""
Cache management utilities
"""

import os
import json
import pickle
import hashlib
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging

from ..config import Config

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching for the RAG system"""
    
    def __init__(self):
        self.cache_dir = Config.CACHE_DIR
        self.ensure_cache_directory()
    
    def ensure_cache_directory(self):
        """Ensure cache directory exists"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            logger.info(f"📁 Created cache directory: {self.cache_dir}")
    
    def get_file_hash(self, filepath: str) -> str:
        """Get MD5 hash of a file"""
        if not os.path.exists(filepath):
            return ""

        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def is_cache_valid(self) -> bool:
        """Check if cache is valid by comparing file hashes"""
        try:
            if not os.path.exists(Config.CACHE_METADATA_FILE):
                return False

            with open(Config.CACHE_METADATA_FILE, 'r') as f:
                cache_metadata = json.load(f)

            # Check if data file has changed
            if not os.path.exists(Config.DATA_FILE_PATH):
                return False

            current_hash = self.get_file_hash(Config.DATA_FILE_PATH)
            cached_hash = cache_metadata.get("data_file_hash", "")

            if current_hash != cached_hash:
                logger.info("📊 Data file has changed, cache invalid")
                return False

            # Check if all cache files exist
            cache_files = [
                Config.EMBEDDINGS_CACHE_FILE,
                Config.FAISS_INDEX_CACHE_FILE,
                Config.DATA_CACHE_FILE,
                Config.STATS_CACHE_FILE
            ]

            for cache_file in cache_files:
                if not os.path.exists(cache_file):
                    logger.info(f"❌ Cache file missing: {cache_file}")
                    return False

            logger.info("✅ Cache is valid")
            return True

        except Exception as e:
            logger.error(f"Error checking cache validity: {e}")
            return False
    
    def save_cache_metadata(self):
        """Save cache metadata including file hashes"""
        try:
            metadata = {
                "data_file_hash": self.get_file_hash(Config.DATA_FILE_PATH),
                "created_at": datetime.now().isoformat(),
                "cache_version": "1.0"
            }

            with open(Config.CACHE_METADATA_FILE, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info("💾 Cache metadata saved")

        except Exception as e:
            logger.error(f"Error saving cache metadata: {e}")
    
    def save_model_to_cache(self, model):
        """Save model to cache"""
        try:
            # Only save the fallback model to cache (TimesFM model is too large)
            if hasattr(model, 'embed'):
                with open(Config.MODEL_CACHE_FILE, 'wb') as f:
                    pickle.dump(model, f)
                logger.info("💾 Model saved to cache")
            else:
                logger.info("⚠️ TimesFM model too large for caching, skipping")

        except Exception as e:
            logger.error(f"❌ Failed to save model to cache: {e}")
    
    def load_model_from_cache(self):
        """Load model from cache if available"""
        try:
            if os.path.exists(Config.MODEL_CACHE_FILE):
                logger.info("📦 Loading model from cache...")
                with open(Config.MODEL_CACHE_FILE, 'rb') as f:
                    model = pickle.load(f)
                logger.info("✅ Model loaded from cache successfully")
                return model
            else:
                logger.info("🔄 No model cache found, will initialize fresh")
                return None

        except Exception as e:
            logger.error(f"❌ Failed to load model from cache: {e}")
            return None
    
    def save_data_to_cache(self, records, data_stats):
        """Save processed data to cache"""
        try:
            with open(Config.DATA_CACHE_FILE, 'wb') as f:
                pickle.dump(records, f)

            with open(Config.STATS_CACHE_FILE, 'wb') as f:
                pickle.dump(data_stats, f)

            logger.info("💾 Processed data saved to cache")

        except Exception as e:
            logger.error(f"❌ Failed to save data to cache: {e}")
    
    def load_data_from_cache(self):
        """Load processed data from cache"""
        try:
            if os.path.exists(Config.DATA_CACHE_FILE) and os.path.exists(Config.STATS_CACHE_FILE):
                logger.info("📦 Loading processed data from cache...")

                with open(Config.DATA_CACHE_FILE, 'rb') as f:
                    records = pickle.load(f)

                with open(Config.STATS_CACHE_FILE, 'rb') as f:
                    data_stats = pickle.load(f)

                logger.info(f"✅ Loaded {len(records)} records from cache")
                return records, data_stats
            else:
                logger.info("🔄 No data cache found, will process fresh")
                return None, None

        except Exception as e:
            logger.error(f"❌ Failed to load data from cache: {e}")
            return None, None
    
    def save_embeddings_to_cache(self, embeddings):
        """Save embeddings to cache"""
        try:
            if embeddings is not None:
                np.save(Config.EMBEDDINGS_CACHE_FILE, embeddings)
                logger.info("💾 Embeddings saved to cache")

        except Exception as e:
            logger.error(f"❌ Failed to save embeddings to cache: {e}")
    
    def load_embeddings_from_cache(self):
        """Load embeddings from cache"""
        try:
            if os.path.exists(Config.EMBEDDINGS_CACHE_FILE):
                logger.info("📦 Loading embeddings from cache...")
                embeddings = np.load(Config.EMBEDDINGS_CACHE_FILE)
                logger.info(f"✅ Loaded embeddings from cache: {embeddings.shape}")
                return embeddings
            else:
                logger.info("🔄 No embeddings cache found, will generate fresh")
                return None

        except Exception as e:
            logger.error(f"❌ Failed to load embeddings from cache: {e}")
            return None
    
    def save_faiss_index_to_cache(self, faiss_index):
        """Save FAISS index to cache"""
        try:
            if faiss_index is not None:
                with open(Config.FAISS_INDEX_CACHE_FILE, 'wb') as f:
                    pickle.dump(faiss_index, f)
                logger.info("💾 FAISS index saved to cache")

        except Exception as e:
            logger.error(f"❌ Failed to save FAISS index to cache: {e}")
    
    def load_faiss_index_from_cache(self):
        """Load FAISS index from cache"""
        try:
            if os.path.exists(Config.FAISS_INDEX_CACHE_FILE):
                logger.info("📦 Loading FAISS index from cache...")
                with open(Config.FAISS_INDEX_CACHE_FILE, 'rb') as f:
                    faiss_index = pickle.load(f)
                logger.info(f"✅ Loaded FAISS index from cache: {faiss_index.ntotal} vectors")
                return faiss_index
            else:
                logger.info("🔄 No FAISS index cache found, will create fresh")
                return None

        except Exception as e:
            logger.error(f"❌ Failed to load FAISS index from cache: {e}")
            return None
    
    def clear_cache(self):
        """Clear all cache files"""
        try:
            cache_files = [
                Config.MODEL_CACHE_FILE,
                Config.EMBEDDINGS_CACHE_FILE,
                Config.FAISS_INDEX_CACHE_FILE,
                Config.DATA_CACHE_FILE,
                Config.STATS_CACHE_FILE,
                Config.CACHE_METADATA_FILE
            ]

            removed_files = []
            for cache_file in cache_files:
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                    removed_files.append(cache_file)

            logger.info(f"🗑️ Cleared cache files: {removed_files}")
            return removed_files

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return []
    
    def get_cache_status(self):
        """Get detailed cache status"""
        try:
            cache_files = {
                "model": Config.MODEL_CACHE_FILE,
                "embeddings": Config.EMBEDDINGS_CACHE_FILE,
                "faiss_index": Config.FAISS_INDEX_CACHE_FILE,
                "data": Config.DATA_CACHE_FILE,
                "stats": Config.STATS_CACHE_FILE,
                "metadata": Config.CACHE_METADATA_FILE
            }

            status = {}
            total_size = 0

            for name, filepath in cache_files.items():
                if os.path.exists(filepath):
                    size = os.path.getsize(filepath)
                    status[name] = {
                        "exists": True,
                        "size_bytes": size,
                        "size_mb": round(size / (1024 * 1024), 2),
                        "modified": datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
                    }
                    total_size += size
                else:
                    status[name] = {"exists": False}

            return {
                "cache_valid": self.is_cache_valid(),
                "cache_directory": self.cache_dir,
                "files": status,
                "total_size_mb": round(total_size / (1024 * 1024), 2)
            }

        except Exception as e:
            logger.error(f"Error getting cache status: {e}")
            return {"error": "Failed to get cache status"}
