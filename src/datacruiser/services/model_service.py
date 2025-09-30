"""
Model service for handling embedding models
"""

import torch
import numpy as np
from typing import Optional
import logging

from ..config import Config
from ..models.embedding_models import EmbeddingModel, TimesFMModel, FallbackEmbeddingModel
from ..utils.cache_utils import CacheManager
from ..utils.device_utils import DeviceDetector
from transformers import TimesFmModelForPrediction

logger = logging.getLogger(__name__)


class ModelService:
    """Service for managing embedding models"""
    
    def __init__(self):
        self.model: Optional[EmbeddingModel] = None
        self.cache_manager = CacheManager()
        self.device_info = DeviceDetector.detect_device()
    
    def initialize_model(self) -> bool:
        """Initialize the embedding model with caching"""
        # Try to load from cache first
        cached_model = self.cache_manager.load_model_from_cache()
        if cached_model is not None:
            self.model = cached_model
            return True
        
        try:
            # Always use CPU mode for stability (avoid segfaults)
            logger.info("🚀 Loading TimesFM model on CPU (safe mode)...")
            
            try:
                timesfm_model = TimesFMModel(
                    TimesFmModelForPrediction.from_pretrained(
                        Config.TIMESFM_MODEL_NAME,
                        torch_dtype=torch.float32,
                        device_map="cpu",
                        low_cpu_mem_usage=True,
                        trust_remote_code=True
                    )
                )
                
                self.model = timesfm_model
                logger.info(f"✅ TimesFM model loaded successfully on CPU ({self.device_info['cpu_count']} cores)")
                # Don't cache TimesFM model as it's too large
                return True
                
            except Exception as timesfm_error:
                logger.error(f"❌ Failed to load TimesFM model: {str(timesfm_error)}")
                logger.info("🔄 Using fallback embedding method...")
                return self._init_fallback_embedding()
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize model: {str(e)}")
            return self._init_fallback_embedding()
    
    def _init_fallback_embedding(self) -> bool:
        """Initialize a simple fallback embedding method"""
        try:
            self.model = FallbackEmbeddingModel()
            logger.info("✅ Fallback embedding model initialized")
            
            # Save fallback model to cache
            self.cache_manager.save_model_to_cache(self.model)
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize fallback model: {e}")
            return False
    
    def generate_embedding(self, series: np.ndarray) -> np.ndarray:
        """Generate embedding with robust error handling"""
        if self.model is None:
            raise ValueError("Model not initialized")
        
        try:
            # Generate embedding
            emb = self.model.embed(series)
            
            # Normalize safely
            norm = np.linalg.norm(emb)
            if norm > 1e-12:
                emb = emb / norm
            
            return emb.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # Return a deterministic hash-based embedding as last resort
            hash_val = hash(tuple(series.tolist())) % (2**31)
            np.random.seed(hash_val)
            return np.random.normal(0, 1, Config.EMBEDDING_DIMENSION).astype(np.float32)
    
    def is_initialized(self) -> bool:
        """Check if model is initialized"""
        return self.model is not None
