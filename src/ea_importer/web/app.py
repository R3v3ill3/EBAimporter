"""
EA Importer Web Interface - FastAPI Application

Provides web-based human-in-the-loop review workflows for:
- Document clustering review and approval
- Family building and gold text selection
- Overlay diff visualization and approval
- QA test result review
- System monitoring and management
"""

from fastapi import FastAPI, HTTPException, Depends, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn
from typing import List, Dict, Any, Optional
import json
import logging
from pathlib import Path
from datetime import datetime

from ..core.config import get_settings
from ..core.logging import get_logger
from ..database import get_db_session, setup_database
from ..models import (
    DocumentDB, ClauseDB, FingerprintDB, 
    ClusterCandidateDB, FamilyDB, InstanceDB, OverlayDB
)
from .routes import (
    clustering_router, 
    family_router, 
    instance_router,
    qa_router,
    monitoring_router
)

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="EA Importer Web Interface",
    description="Human-in-the-loop review system for Enterprise Agreement processing",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Ensure database is initialized at startup (creates tables if missing)
@app.on_event("startup")
async def on_startup() -> None:
    try:
        logger.info("Initializing database on startup")
        setup_database()
        logger.info("Database ready")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        # Do not raise here to allow the app to start and show error pages

# Add middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up templates and static files
templates_dir = Path(__file__).parent / "templates"
static_dir = Path(__file__).parent / "static"

templates = Jinja2Templates(directory=str(templates_dir))

# Mount static files
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Include routers
app.include_router(clustering_router, prefix="/clustering", tags=["clustering"])
app.include_router(family_router, prefix="/families", tags=["families"])
app.include_router(instance_router, prefix="/instances", tags=["instances"])
app.include_router(qa_router, prefix="/qa", tags=["qa"])
app.include_router(monitoring_router, prefix="/monitoring", tags=["monitoring"])


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard showing system overview"""
    
    try:
        with get_db_session() as session:
            # Get system statistics
            stats = {
                'documents': session.query(DocumentDB).count(),
                'clauses': session.query(ClauseDB).count(),
                'fingerprints': session.query(FingerprintDB).count(),
                'cluster_candidates': session.query(ClusterCandidateDB).count(),
                'families': session.query(FamilyDB).count(),
                'instances': session.query(InstanceDB).count(),
                'overlays': session.query(OverlayDB).count()
            }
            
            # Get recent activity
            recent_docs = session.query(DocumentDB).order_by(
                DocumentDB.created_at.desc()
            ).limit(5).all()
            
            recent_families = session.query(FamilyDB).order_by(
                FamilyDB.created_at.desc()
            ).limit(5).all()
            
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "stats": stats,
            "recent_docs": recent_docs,
            "recent_families": recent_families,
            "db_error": None,
            "page_title": "EA Importer Dashboard"
        })
        
    except Exception as e:
        logger.error(f"Error loading dashboard: {e}")
        # Degraded mode: render dashboard with placeholders
        empty_stats = {
            'documents': 0,
            'clauses': 0,
            'fingerprints': 0,
            'cluster_candidates': 0,
            'families': 0,
            'instances': 0,
            'overlays': 0
        }
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "stats": empty_stats,
            "recent_docs": [],
            "recent_families": [],
            "db_error": str(e),
            "page_title": "EA Importer Dashboard"
        })


@app.get("/documents", response_class=HTMLResponse)
async def documents_list(request: Request, page: int = 1, per_page: int = 20):
    """List all processed documents with pagination"""
    
    try:
        with get_db_session() as session:
            offset = (page - 1) * per_page
            
            documents = session.query(DocumentDB).order_by(
                DocumentDB.created_at.desc()
            ).offset(offset).limit(per_page).all()
            
            total_count = session.query(DocumentDB).count()
            total_pages = (total_count + per_page - 1) // per_page
            
        return templates.TemplateResponse("documents/list.html", {
            "request": request,
            "documents": documents,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages,
            "total_count": total_count,
            "page_title": "Documents"
        })
        
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to load documents")


@app.get("/documents/{document_id}", response_class=HTMLResponse)
async def document_detail(request: Request, document_id: int):
    """Show detailed view of a document"""
    
    try:
        with get_db_session() as session:
            document = session.query(DocumentDB).filter(
                DocumentDB.id == document_id
            ).first()
            
            if not document:
                raise HTTPException(status_code=404, detail="Document not found")
            
            # Get related data
            clauses = session.query(ClauseDB).filter(
                ClauseDB.document_id == document_id
            ).order_by(ClauseDB.clause_number).all()
            
            fingerprint = session.query(FingerprintDB).filter(
                FingerprintDB.document_id == document_id
            ).first()
            
        return templates.TemplateResponse("documents/detail.html", {
            "request": request,
            "document": document,
            "clauses": clauses,
            "fingerprint": fingerprint,
            "page_title": f"Document: {document.file_name}"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading document {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to load document")


@app.post("/upload")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    auto_process: bool = Form(False)
):
    """Upload a new document for processing"""
    
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Save uploaded file
        settings = get_settings()
        upload_dir = Path(settings.paths.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / file.filename
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"Uploaded file: {file.filename} ({len(content)} bytes)")
        
        # If auto_process is enabled, trigger processing
        if auto_process:
            # This would typically be an async task
            # For now, we'll just return the upload confirmation
            pass
        
        return JSONResponse({
            "success": True,
            "message": f"File {file.filename} uploaded successfully",
            "file_path": str(file_path),
            "auto_process": auto_process
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload file")


@app.get("/api/health")
async def health_check():
    """Health check endpoint for monitoring"""
    
    try:
        # Test database connection
        with get_db_session() as session:
            session.execute("SELECT 1")
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "database": "connected"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
        )


@app.get("/api/stats")
async def get_stats():
    """Get system statistics as JSON"""
    
    try:
        with get_db_session() as session:
            stats = {
                'documents': session.query(DocumentDB).count(),
                'clauses': session.query(ClauseDB).count(),
                'fingerprints': session.query(FingerprintDB).count(),
                'cluster_candidates': session.query(ClusterCandidateDB).count(),
                'families': session.query(FamilyDB).count(),
                'instances': session.query(InstanceDB).count(),
                'overlays': session.query(OverlayDB).count(),
                'timestamp': datetime.utcnow().isoformat()
            }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Custom 404 error handler"""
    return templates.TemplateResponse(
        "errors/404.html", 
        {"request": request, "page_title": "Page Not Found"},
        status_code=404
    )


@app.exception_handler(500)
async def server_error_handler(request: Request, exc: HTTPException):
    """Custom 500 error handler"""
    return templates.TemplateResponse(
        "errors/500.html",
        {"request": request, "page_title": "Server Error"},
        status_code=500
    )


def create_app(config_override: Optional[Dict[str, Any]] = None) -> FastAPI:
    """
    Application factory for creating configured FastAPI instances.
    
    Args:
        config_override: Optional configuration overrides
        
    Returns:
        Configured FastAPI application
    """
    if config_override:
        # Apply configuration overrides
        pass
    
    return app


def run_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    debug: bool = False,
    reload: bool = False
):
    """
    Run the web server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        debug: Enable debug mode
        reload: Enable auto-reload
    """
    logger.info(f"Starting EA Importer web server on {host}:{port}")
    
    uvicorn.run(
        "ea_importer.web.app:app",
        host=host,
        port=port,
        debug=debug,
        reload=reload,
        log_level="info" if not debug else "debug"
    )


if __name__ == "__main__":
    run_server(debug=True, reload=True)