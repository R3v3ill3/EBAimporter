"""
Utilities package for EA Importer system.
Contains core processing utilities for PDF handling, text cleaning, segmentation, and fingerprinting.
"""

# Import main utility classes
from .pdf_processor import PDFProcessor, PDFProcessingError, OCRNotAvailableError, process_pdf_batch
from .text_cleaner import TextCleaner, TextCleaningError, clean_text_batch, clean_document_batch
from .text_segmenter import TextSegmenter, SegmentationError, NumberingPattern, segment_documents_batch
from .fingerprinter import (
    Fingerprinter, LSHIndex, TextPreprocessor, FingerprintingError,
    fingerprint_documents_batch, save_fingerprints_batch
)

# Export all main classes and functions
__all__ = [
    # PDF Processing
    'PDFProcessor',
    'PDFProcessingError', 
    'OCRNotAvailableError',
    'process_pdf_batch',
    
    # Text Cleaning
    'TextCleaner',
    'TextCleaningError',
    'clean_text_batch',
    'clean_document_batch',
    
    # Text Segmentation
    'TextSegmenter',
    'SegmentationError',
    'NumberingPattern',
    'segment_documents_batch',
    
    # Fingerprinting
    'Fingerprinter',
    'LSHIndex',
    'TextPreprocessor',
    'FingerprintingError',
    'fingerprint_documents_batch',
    'save_fingerprints_batch',
]


def get_processor_pipeline():
    """
    Get a pre-configured processing pipeline with all utilities.
    
    Returns:
        Dictionary with initialized processor instances
    """
    return {
        'pdf_processor': PDFProcessor(),
        'text_cleaner': TextCleaner(),
        'text_segmenter': TextSegmenter(),
        'fingerprinter': Fingerprinter(),
    }


def validate_dependencies():
    """
    Validate that all required dependencies are available.
    
    Returns:
        Dictionary with dependency status
    """
    dependencies = {
        'pdf_processing': False,
        'ocr': False,
        'nlp': False,
        'fingerprinting': False,
        'embeddings': False
    }
    
    # Check PDF processing
    try:
        import pdfplumber
        dependencies['pdf_processing'] = True
    except ImportError:
        pass
    
    # Check OCR
    try:
        import pytesseract
        import PIL
        dependencies['ocr'] = True
    except ImportError:
        pass
    
    # Check NLP
    try:
        import spacy
        dependencies['nlp'] = True
    except ImportError:
        pass
    
    # Check fingerprinting
    try:
        import datasketch
        dependencies['fingerprinting'] = True
    except ImportError:
        pass
    
    # Check embeddings
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        dependencies['embeddings'] = True
    except (ImportError, OSError):
        pass
    
    return dependencies


def get_system_info():
    """
    Get comprehensive system information for diagnostics.
    
    Returns:
        Dictionary with system information
    """
    dependencies = validate_dependencies()
    
    info = {
        'utilities_available': {
            'PDFProcessor': True,
            'TextCleaner': True, 
            'TextSegmenter': True,
            'Fingerprinter': True,
        },
        'dependencies': dependencies,
        'optional_features': {
            'ocr_processing': dependencies['ocr'],
            'ml_segmentation': dependencies['nlp'],
            'minhash_fingerprinting': dependencies['fingerprinting'],
            'semantic_embeddings': dependencies['embeddings'],
        }
    }
    
    return info