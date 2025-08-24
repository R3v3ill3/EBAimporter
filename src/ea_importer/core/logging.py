"""
Logging configuration for the EA Importer system.
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

from .config import get_settings


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[Path] = None,
    rich_console: bool = True
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file to write logs to
        rich_console: Whether to use Rich console handler for better formatting
    
    Returns:
        Configured logger instance
    """
    settings = get_settings()
    
    # Use provided log level or fall back to settings
    level = log_level or settings.log_level
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if rich_console:
        console = Console(stderr=True)
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=True,
            rich_tracebacks=True,
            tracebacks_show_locals=settings.debug
        )
        console_handler.setLevel(getattr(logging, level.upper()))
    else:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(getattr(logging, level.upper()))
        formatter = logging.Formatter(settings.log_format)
        console_handler.setFormatter(formatter)
    
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Set up specific logger for our application
    app_logger = logging.getLogger("ea_importer")
    
    # Configure third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    
    # Database loggers
    if not settings.debug:
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy.dialects").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy.orm").setLevel(logging.WARNING)
    
    app_logger.info(f"Logging configured at {level} level")
    
    return app_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module."""
    return logging.getLogger(f"ea_importer.{name}")


class LoggerMixin:
    """Mixin to add logging capabilities to classes."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(self.__class__.__module__ + "." + self.__class__.__name__)