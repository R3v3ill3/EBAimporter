"""
Centralized logging configuration for EA Importer system.
Provides structured logging with proper formatting and context.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from .config import get_settings


class ContextFilter(logging.Filter):
    """Add context information to log records"""
    
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.context = context or {}
    
    def filter(self, record):
        for key, value in self.context.items():
            setattr(record, key, value)
        return True


class ColoredFormatter(logging.Formatter):
    """Add colors to console log output"""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        
        return super().format(record)


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[Path] = None,
    enable_colors: bool = True,
    context: Optional[Dict[str, Any]] = None
) -> logging.Logger:
    """
    Set up comprehensive logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for file logging
        enable_colors: Whether to enable colored console output
        context: Additional context to include in all log messages
    
    Returns:
        Configured root logger
    """
    settings = get_settings()
    
    # Determine log level
    if log_level is None:
        log_level = settings.log_level.value
    
    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Console format
    if enable_colors and sys.stdout.isatty():
        console_format = ColoredFormatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Rotating file handler to prevent huge log files
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)  # File gets all messages
        
        # File format (no colors)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-20s | %(pathname)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    # Add context filter if provided
    if context:
        context_filter = ContextFilter(context)
        for handler in logger.handlers:
            handler.addFilter(context_filter)
    
    # Set specific logger levels for third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('selenium').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name"""
    return logging.getLogger(name)


def setup_component_logger(
    component_name: str,
    log_level: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> logging.Logger:
    """
    Set up logging for a specific component.
    
    Args:
        component_name: Name of the component (e.g., 'pdf_processor', 'clustering')
        log_level: Optional override log level
        context: Additional context for this component
    
    Returns:
        Component-specific logger
    """
    settings = get_settings()
    
    # Create component-specific log file
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"{component_name}.log"
    
    # Merge context
    component_context = {"component": component_name}
    if context:
        component_context.update(context)
    
    # Set up logging
    setup_logging(
        log_level=log_level,
        log_file=log_file,
        context=component_context
    )
    
    return get_logger(f"ea_importer.{component_name}")


def log_function_call(func):
    """Decorator to log function calls with parameters and execution time"""
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        
        # Log function entry
        func_name = f"{func.__qualname__}"
        logger.debug(f"Entering {func_name} with args={args}, kwargs={kwargs}")
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"Completed {func_name} in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error in {func_name} after {execution_time:.3f}s: {e}")
            raise
    
    return wrapper


def log_pipeline_stage(stage_name: str):
    """Decorator to log pipeline stage execution"""
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger("ea_importer.pipeline")
            
            logger.info(f"Starting pipeline stage: {stage_name}")
            start_time = datetime.now()
            
            try:
                result = func(*args, **kwargs)
                duration = datetime.now() - start_time
                logger.info(f"Completed pipeline stage: {stage_name} (duration: {duration})")
                return result
            except Exception as e:
                duration = datetime.now() - start_time
                logger.error(f"Failed pipeline stage: {stage_name} after {duration}: {e}")
                raise
        
        return wrapper
    return decorator


# Initialize default logging on module import
if not logging.getLogger().handlers:
    setup_logging()