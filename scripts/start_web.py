#!/usr/bin/env python3
"""
Start script for EA Importer Web Interface.

This script can be used to start the web interface independently
without using the CLI commands.
"""

import sys
import os
from pathlib import Path

# Add src to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

import uvicorn
from ea_importer.core.config import get_settings
from ea_importer.core.logging import setup_logging, get_logger


def main():
    """Start the web interface."""
    
    # Setup logging
    setup_logging(log_level="INFO")
    logger = get_logger(__name__)
    
    # Get settings
    settings = get_settings()
    
    # Ensure directories exist
    settings.ensure_directories()
    
    logger.info("Starting EA Importer Web Interface")
    logger.info(f"Host: {settings.web_host}")
    logger.info(f"Port: {settings.web_port}")
    logger.info(f"Data root: {settings.data_root}")
    
    try:
        # Start the web server
        uvicorn.run(
            "ea_importer.web:app",
            host=settings.web_host,
            port=settings.web_port,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("Web interface stopped by user")
    except Exception as e:
        logger.error(f"Failed to start web interface: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()