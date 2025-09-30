#!/usr/bin/env python3
"""
Simple server runner for testing without OpenAI API key
"""

import sys
import os
import logging

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from datacruiser.app_factory import create_app, initialize_system

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run the server with simplified initialization"""
    try:
        logger.info("🚀 Starting DataCruiser RAG Server (Simple Mode)...")
        
        # Create app
        app = create_app()
        
        # Initialize system (this will take time on first run)
        logger.info("⏳ Initializing system (this may take a few minutes)...")
        if initialize_system(app):
            logger.info("✅ System initialized successfully!")
        else:
            logger.warning("⚠️ System initialization had issues, but continuing...")
        
        logger.info("🔗 Server starting on http://localhost:5080")
        logger.info("📋 Available endpoints:")
        logger.info("  - GET  /           : Health check")
        logger.info("  - POST /api/ask    : Ask questions")
        logger.info("  - GET  /data/info : Data information")
        logger.info("  - POST /cache/clear: Clear cache")
        logger.info("  - GET  /cache/status: Cache status")
        
        # Start server
        app.run(
            host="0.0.0.0",
            port=5080,
            debug=False,
            threaded=True
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to start server: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
