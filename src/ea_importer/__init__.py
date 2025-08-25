"""
EA Importer - Australian Enterprise Agreement Ingestion & Corpus Building System

A comprehensive system for ingesting, processing, clustering, and analyzing 
Australian Enterprise Agreements (EAs) to build a versioned, auditable, 
query-ready corpus.

Key Features:
- Batch PDF ingestion with OCR fallback
- Intelligent clause segmentation and fingerprinting
- Automated clustering into agreement families
- Human-in-the-loop review workflows
- Rates and rules extraction
- Instance management with overlays
- QA smoke testing with synthetic scenarios
- Version control and corpus locking
- Web interface for review and management
- Comprehensive CLI for all operations

Usage:
    from ea_importer import EAImporter
    
    # Initialize system
    system = EAImporter()
    
    # Process PDFs
    system.ingest_pdfs("path/to/pdfs")
    
    # Run clustering
    system.cluster_documents()
    
    # Build families
    system.build_families()
"""

__version__ = "1.0.0"
__author__ = "EA Importer Team"
__license__ = "MIT"

# Import main components
from .core import get_settings, setup_logging, get_logger
from .models import Base, PDFDocument, ClauseSegment, DocumentFingerprint
from .database import get_database, get_db_session, setup_database

# Version info
VERSION_INFO = {
    "version": __version__,
    "author": __author__,
    "license": __license__,
    "python_requires": ">=3.8",
}

# Export key components
__all__ = [
    # Version info
    "__version__",
    "VERSION_INFO",
    
    # Core components
    "get_settings",
    "setup_logging", 
    "get_logger",
    
    # Data models
    "Base",
    "PDFDocument",
    "ClauseSegment",
    "DocumentFingerprint",
    
    # Database
    "get_database",
    "get_db_session",
    "setup_database",
]


def get_version_info():
    """Get detailed version information."""
    return VERSION_INFO.copy()


def quick_setup():
    """
    Quick setup for new users.
    Initializes logging and database.
    """
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)
    logger.info(f"EA Importer v{__version__} - Quick Setup")
    
    # Setup database
    try:
        setup_database()
        logger.info("Database setup completed")
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        raise
    
    logger.info("Quick setup completed successfully")


# Initialize logging on import
setup_logging()