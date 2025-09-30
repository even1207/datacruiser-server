#!/usr/bin/env python3
"""
Main entry point for DataCruiser API Server
"""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from datacruiser.app_factory import main

if __name__ == "__main__":
    main()
