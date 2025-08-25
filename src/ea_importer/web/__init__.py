"""
EA Importer Web Interface Module

This module provides a comprehensive web-based user interface for human-in-the-loop
review workflows in the EA Importer system.

Key Components:
- FastAPI application with templated HTML interface
- Clustering review and approval workflows
- Family management and gold text editing
- Instance and overlay diff visualization
- QA testing result review
- System monitoring and administration

Usage:
    from ea_importer.web import create_app, run_server
    
    # Create application
    app = create_app()
    
    # Run development server
    run_server(host="localhost", port=8080, debug=True)
"""

from .app import create_app, run_server, app

__all__ = [
    'create_app',
    'run_server', 
    'app'
]