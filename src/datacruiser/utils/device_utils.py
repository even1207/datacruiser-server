"""
Device detection and management utilities
"""

import os
import torch
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class DeviceDetector:
    """Device detection and management"""
    
    @staticmethod
    def detect_device() -> Dict[str, Any]:
        """Detect and return device information"""
        # Detect CUDA
        cuda_available = torch.cuda.is_available()
        device_type = "cuda" if cuda_available else "cpu"

        device_info = {
            "type": device_type,
            "cuda_available": cuda_available,
            "gpu_count": torch.cuda.device_count() if cuda_available else 0,
            "cpu_count": os.cpu_count()
        }

        if cuda_available:
            device_info["gpu_name"] = torch.cuda.get_device_name(0)
            device_info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        logger.info(f"🔍 Device detection: {device_info}")
        return device_info
    
    @staticmethod
    def clear_gpu_memory():
        """Clear GPU memory if available"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("🧹 GPU memory cleared")
