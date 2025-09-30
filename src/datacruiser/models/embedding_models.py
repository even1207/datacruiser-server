"""
Embedding models for the RAG system
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class EmbeddingModel(ABC):
    """Abstract base class for embedding models"""
    
    @abstractmethod
    def embed(self, series: np.ndarray) -> np.ndarray:
        """Generate embedding for a time series"""
        pass


class TimesFMModel(EmbeddingModel):
    """TimesFM model wrapper"""
    
    def __init__(self, model):
        self.model = model
    
    def embed(self, series: np.ndarray) -> np.ndarray:
        """Generate embedding using TimesFM model"""
        try:
            series_reshaped = series.reshape(1, -1)
            x = torch.tensor(series_reshaped, dtype=torch.float32, device="cpu")
            f = torch.tensor([0], dtype=torch.long, device="cpu")

            with torch.no_grad():
                out = self.model(
                    past_values=x,
                    freq=f,
                    output_hidden_states=True,
                    return_dict=True
                )
                last_hidden = out.hidden_states[-1][0]
                emb = last_hidden.mean(dim=0).float()
                return emb.cpu().numpy()

        except Exception as e:
            logger.warning(f"TimesFM embedding failed: {e}")
            # Fallback to statistical features
            return self._statistical_fallback(series)
    
    def _statistical_fallback(self, series: np.ndarray) -> np.ndarray:
        """Fallback to statistical features"""
        features = np.array([
            series.mean(),
            series.std(),
            series.min(),
            series.max(),
            np.median(series)
        ])
        # Expand to 512 dimensions by repeating
        expanded = np.tile(features, 102)[:512]
        return expanded.astype(np.float32)


class FallbackEmbeddingModel(EmbeddingModel):
    """Simple fallback embedding model"""
    
    def __init__(self):
        self.device = torch.device("cpu")
        # Simple linear layer for fallback
        self.linear = torch.nn.Linear(5, 512)
        self.linear.eval()
    
    def embed(self, series: np.ndarray) -> np.ndarray:
        """Generate embedding using fallback model"""
        with torch.no_grad():
            x = torch.tensor(series, dtype=torch.float32).unsqueeze(0)
            return self.linear(x).squeeze(0).numpy()
