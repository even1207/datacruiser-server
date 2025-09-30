"""
Configuration module for the RAG-based Footfall Analysis API
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Base configuration class"""
    
    # Flask configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5090))
    
    # API Keys
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    # Model configuration
    TIMESFM_MODEL_NAME = "google/timesfm-2.0-500m-pytorch"
    EMBEDDING_DIMENSION = 512
    MAX_RECORDS = 5000
    BATCH_SIZE = 10
    
    # Cache configuration
    CACHE_DIR = "cache"
    MODEL_CACHE_FILE = os.path.join(CACHE_DIR, "timesfm_model.pkl")
    EMBEDDINGS_CACHE_FILE = os.path.join(CACHE_DIR, "embeddings.npy")
    FAISS_INDEX_CACHE_FILE = os.path.join(CACHE_DIR, "faiss_index.pkl")
    DATA_CACHE_FILE = os.path.join(CACHE_DIR, "processed_data.pkl")
    STATS_CACHE_FILE = os.path.join(CACHE_DIR, "data_stats.pkl")
    CACHE_METADATA_FILE = os.path.join(CACHE_DIR, "cache_metadata.json")
    
    # Data configuration
    DATA_FILE_PATH = os.path.join("data", "data.json")
    UPLOAD_BASE_DIR = os.getenv('UPLOAD_BASE_DIR', os.path.join('data', 'uploads'))
    UPLOAD_CHUNK_SIZE = int(os.getenv('UPLOAD_CHUNK_SIZE', 50000))
    MAX_UPLOAD_FILES = int(os.getenv('MAX_UPLOAD_FILES', 10))
    SAMPLE_ROWS_PER_DATASET = int(os.getenv('SAMPLE_ROWS_PER_DATASET', 20))
    
    # Environment variables for stability
    KMP_DUPLICATE_LIB_OK = "TRUE"
    OMP_NUM_THREADS = "1"
    TOKENIZERS_PARALLELISM = "false"
    
    # Logging configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    @classmethod
    def init_environment(cls):
        """Initialize environment variables for stability"""
        os.environ["KMP_DUPLICATE_LIB_OK"] = cls.KMP_DUPLICATE_LIB_OK
        os.environ["OMP_NUM_THREADS"] = cls.OMP_NUM_THREADS
        os.environ["TOKENIZERS_PARALLELISM"] = cls.TOKENIZERS_PARALLELISM
