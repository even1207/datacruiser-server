"""
DataCruiser - RAG-based Footfall Analysis API

A Flask-based API for analyzing footfall data using RAG (Retrieval-Augmented Generation)
with TimesFM embeddings and FAISS similarity search.
"""

__version__ = "1.0.0"
__author__ = "DataCruiser Team"
__description__ = "RAG-based Footfall Analysis API"

from .app_factory import create_app, initialize_system

__all__ = [
    'create_app',
    'initialize_system'
]
