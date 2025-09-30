"""
Application factory for the RAG-based Footfall Analysis API
"""

import os
import signal
import sys
import logging
from flask import Flask, g

from .config import Config
from .services.model_service import ModelService
from .services.rag_service import RAGService
from .services.llm_service import LLMService
from .services.hybrid_rag_service import HybridRAGService
from .services.air_quality_service import AirQualityService
from .services.uploaded_dataset_service import UploadedDatasetService
from .routes import api_bp, health_bp, cache_bp, air_quality_bp

# Initialize environment
Config.init_environment()

# Configure logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """Create and configure the Flask application"""
    app = Flask(__name__)
    app.config.from_object(Config)

    # Register blueprints
    app.register_blueprint(health_bp)
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(cache_bp, url_prefix='/cache')
    app.register_blueprint(air_quality_bp, url_prefix='/air-quality')

    # Initialize services
    model_service = ModelService()
    llm_service = LLMService()
    rag_service = RAGService(model_service, llm_service)
    dataset_service = UploadedDatasetService()

    # Initialize hybrid service
    csv_file_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'data',
        'data.csv'
    )
    hybrid_service = HybridRAGService(model_service, llm_service, csv_file_path)

    # Initialize air quality service
    air_quality_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'data',
        'nsw_106_sensors_2025_09_05'
    )
    air_quality_service = AirQualityService(air_quality_data_dir)

    # Store services in app context
    app.model_service = model_service
    app.rag_service = rag_service
    app.llm_service = llm_service
    app.hybrid_service = hybrid_service
    app.air_quality_service = air_quality_service
    app.dataset_service = dataset_service

    # Add request context processors
    @app.before_request
    def before_request():
        g.rag_service = rag_service
        g.llm_service = llm_service
        g.model_service = model_service
        g.hybrid_service = hybrid_service
        g.air_quality_service = air_quality_service
        g.dataset_service = dataset_service

    return app


def initialize_system(app: Flask) -> bool:
    """Initialize the RAG system"""
    logger.info("🚀 Initializing RAG-based Footfall Analysis System (Refactored Version)...")

    try:
        # Initialize model
        if not app.model_service.initialize_model():
            logger.error("❌ Failed to initialize model")
            return False

        # Initialize data
        if not app.rag_service.initialize_data():
            logger.error("❌ Failed to initialize data")
            return False

        # Initialize hybrid service data
        if not app.hybrid_service.initialize_data():
            logger.error("❌ Failed to initialize hybrid service data")
            return False

        # Initialize air quality service data
        try:
            app.air_quality_service.process_csv_files()
            logger.info("✅ Air quality data processed successfully")
        except Exception as e:
            logger.error(f"❌ Failed to process air quality data: {e}")
            return False

        logger.info("✅ System initialization completed successfully!")
        logger.info(f"📊 System loaded with {len(app.rag_service.records)} records")

        if app.rag_service.cache_manager.is_cache_valid():
            logger.info("💾 System initialized from cache - no processing needed!")
        else:
            logger.info("🔄 System initialized fresh - cache will be used next time")

        return True

    except Exception as e:
        logger.error(f"❌ System initialization failed: {str(e)}")
        return False


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""

    def signal_handler(sig, frame):
        logger.info("🛑 Received interrupt signal, cleaning up...")
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Main application entry point"""
    try:
        # Setup signal handlers
        setup_signal_handlers()

        # Create app
        app = create_app()

        # Initialize system
        if not initialize_system(app):
            logger.error("❌ System initialization failed, but starting server anyway...")

        logger.info("🚀 Starting RAG-based Footfall Analysis API Server (Refactored Version)...")
        logger.info("🔗 Available endpoints:")
        logger.info("  - GET  /           : Health check and system status")
        logger.info("  - POST /api/ask    : Ask questions about footfall data (original RAG)")
        logger.info("  - POST /api/ask-hybrid: Ask questions (hybrid: statistical + trend)")
        logger.info("  - GET  /api/system-info: Get system information")
        logger.info("  - GET  /data/info  : Get data and cache information")
        logger.info("  - POST /cache/clear: Clear all cache files")
        logger.info("  - GET  /cache/status: Get detailed cache status")

        app.run(
            host=Config.HOST,
            port=Config.PORT,
            debug=Config.DEBUG,
            threaded=True
        )

    except Exception as e:
        logger.error(f"❌ Failed to start server: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

