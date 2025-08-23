"""
EA Importer - Australian Enterprise Agreement Ingestion & Corpus Builder

A comprehensive system for ingesting, processing, clustering, and querying
Australian Enterprise Agreements (EAs) with human-in-the-loop validation.
"""

__version__ = "0.1.0"
__author__ = "EA Importer Team"
__description__ = "Australian Enterprise Agreement Ingestion & Corpus Builder"

from .core.config import Settings, get_settings
from .core.logging import setup_logging

# Version info
VERSION_INFO = {
    "version": __version__,
    "description": __description__,
    "author": __author__,
}

__all__ = [
    "Settings",
    "get_settings", 
    "setup_logging",
    "VERSION_INFO",
]