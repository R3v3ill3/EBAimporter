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
from fastapi.background import BackgroundTasks
import uvicorn
from typing import List, Dict, Any, Optional
import json
import logging
from pathlib import Path
from datetime import datetime
from sqlalchemy import text

from ..core.config import get_settings
from ..core.logging import get_logger
from ..database import get_db_session, setup_database
from ..models import (
    DocumentDB, ClauseDB, FingerprintDB, 
    ClusterCandidateDB, FamilyDB, InstanceDB, OverlayDB
)
from ..utils.pdf_processor import PDFProcessor
from ..utils.fingerprinter import Fingerprinter
from ..utils.text_segmenter import TextSegmenter
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


def _format_bytes(num_bytes: int) -> str:
    # Human readable file size
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def _dir_size_bytes(path: Path) -> int:
    try:
        if not path.exists():
            return 0
        total = 0
        for p in path.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except Exception:
                    continue
        return total
    except Exception:
        return 0


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
        
        # Live system data
        settings = get_settings()
        data_dir = Path(settings.paths.data_dir)
        upload_dir = Path(settings.paths.upload_dir)
        storage_bytes = _dir_size_bytes(data_dir)
        upload_count = len([p for p in upload_dir.glob("**/*") if p.is_file()]) if upload_dir.exists() else 0

        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "stats": stats,
            "recent_docs": recent_docs,
            "recent_families": recent_families,
            "db_error": None,
            "page_title": "EA Importer Dashboard",
            "storage_bytes": storage_bytes,
            "storage_pretty": _format_bytes(storage_bytes),
            "upload_dir": str(upload_dir),
            "upload_file_count": upload_count,
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
        settings = get_settings()
        data_dir = Path(settings.paths.data_dir)
        storage_bytes = _dir_size_bytes(data_dir)
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "stats": empty_stats,
            "recent_docs": [],
            "recent_families": [],
            "db_error": str(e),
            "page_title": "EA Importer Dashboard",
            "storage_bytes": storage_bytes,
            "storage_pretty": _format_bytes(storage_bytes),
            "upload_dir": str(Path(settings.paths.upload_dir)),
            "upload_file_count": 0,
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


def _process_uploaded_document(document_id: Optional[int], file_path: str) -> None:
    """Background processing hook for uploaded PDFs."""
    logger.info(f"Starting background processing for {file_path} (doc_id={document_id})")
    settings = get_settings()
    processor = PDFProcessor()
    fingerprinter = Fingerprinter()
    segmenter = TextSegmenter()
    try:
        pdf_doc = processor.process_pdf(Path(file_path))
        ea_id = pdf_doc.metadata.get('ea_id')
        # Persist updates
        with get_db_session() as session:
            doc: Optional[DocumentDB] = None
            if document_id:
                doc = session.query(DocumentDB).filter(DocumentDB.id == document_id).first()
            if doc is None:
                # Create if missing
                doc = DocumentDB(
                    ea_id=ea_id or f"EA-{Path(file_path).stem}",
                    file_path=str(file_path),
                    original_filename=Path(file_path).name,
                )
                session.add(doc)
                session.flush()
            # Update fields
            try:
                size_bytes = Path(file_path).stat().st_size
            except Exception:
                size_bytes = None
            doc.ea_id = ea_id or doc.ea_id
            doc.file_size_bytes = size_bytes
            doc.has_text_layer = True  # heuristic; refine if needed
            doc.ocr_used = bool(pdf_doc.metadata.get('ocr_used'))
            doc.total_pages = pdf_doc.total_pages
            # Segment into clauses and persist
            try:
                segments = segmenter.segment_document(pdf_doc, ea_id=doc.ea_id)
                order_index = 0
                for seg in segments:
                    order_index += 1
                    clause = ClauseDB(
                        document_id=doc.id,
                        clause_id=str(seg.clause_id),
                        clause_number=str(seg.clause_id),
                        heading=seg.heading or None,
                        text=seg.text,
                        path_json=seg.path,
                        level=len(seg.path) if seg.path else 0,
                        order_index=order_index,
                        hash_sha256=seg.hash_sha256,
                        token_count=seg.token_count,
                        char_count=len(seg.text) if seg.text else 0,
                        page_spans_json=seg.page_spans,
                    )
                    session.add(clause)
                doc.total_clauses = len(segments)
            except Exception as se:
                logger.warning(f"Segmentation failed for {file_path}: {se}")
            doc.status = "completed"
            doc.processed_at = datetime.utcnow()
            # Fingerprint
            try:
                fp = fingerprinter.fingerprint_document(pdf_doc, include_embeddings=False)
                fp_db = FingerprintDB(
                    document_id=doc.id,
                    minhash_signature=fp.minhash_signature,
                    embedding_vector=None,
                    minhash_permutations=fp.minhash_permutations,
                    ngram_size=fp.ngram_size,
                )
                session.add(fp_db)
            except Exception as fe:
                logger.warning(f"Fingerprinting failed for {file_path}: {fe}")
    except Exception as e:
        logger.error(f"Background processing failed for {file_path}: {e}")
        # Best-effort status update
        try:
            with get_db_session() as session:
                if document_id:
                    doc = session.query(DocumentDB).filter(DocumentDB.id == document_id).first()
                    if doc:
                        doc.status = "failed"
        except Exception:
            pass


@app.post("/documents/{document_id}/process")
async def process_document_endpoint(document_id: int, background_tasks: BackgroundTasks):
    """Trigger background processing or reprocessing for a document."""
    try:
        with get_db_session() as session:
            doc = session.query(DocumentDB).filter(DocumentDB.id == document_id).first()
            if not doc:
                raise HTTPException(status_code=404, detail="Document not found")
            # Update status to processing
            doc.status = "processing"
            file_path = doc.file_path
        background_tasks.add_task(_process_uploaded_document, document_id, file_path)
        return {"success": True, "message": "Processing started", "document_id": document_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start processing for document {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to start processing")


@app.post("/upload")
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    auto_process: bool = Form(False)
):
    """Upload a new document for processing"""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            return JSONResponse(status_code=400, content={"success": False, "message": "Only PDF files are supported"})
        
        settings = get_settings()
        upload_dir = Path(settings.paths.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / file.filename
        
        # Stream upload to disk with size guard
        max_bytes = int(settings.processing.max_file_size_mb) * 1024 * 1024
        bytes_written = 0
        try:
            with open(file_path, "wb") as out_f:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    bytes_written += len(chunk)
                    if bytes_written > max_bytes:
                        try:
                            out_f.flush()
                        except Exception:
                            pass
                        try:
                            out_f.close()
                        except Exception:
                            pass
                        try:
                            file_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                        return JSONResponse(status_code=400, content={
                            "success": False,
                            "message": f"File exceeds max size of {settings.processing.max_file_size_mb} MB"
                        })
                    out_f.write(chunk)
        finally:
            await file.close()
        
        logger.info(f"Uploaded file: {file.filename} ({bytes_written} bytes)")
        
        # Best-effort: create document record if DB is available
        created_document_id: Optional[int] = None
        try:
            with get_db_session() as session:
                doc = DocumentDB(
                    ea_id=f"PENDING-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                    file_path=str(file_path),
                    original_filename=file.filename,
                    file_size_bytes=bytes_written,
                    status="pending",
                )
                session.add(doc)
                session.flush()
                created_document_id = doc.id
        except Exception as db_e:
            logger.warning(f"Upload DB record creation skipped due to error: {db_e}")
        
        # If auto_process is enabled, trigger processing (stub/hook)
        if auto_process:
            background_tasks.add_task(_process_uploaded_document, created_document_id, str(file_path))
        
        return JSONResponse({
            "success": True,
            "message": f"File {file.filename} uploaded successfully",
            "file_path": str(file_path),
            "file_size_bytes": bytes_written,
            "auto_process": auto_process,
            "document_id": created_document_id,
        })
        
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        # Return JSON to avoid HTML error handler for XHR
        return JSONResponse(status_code=500, content={
            "success": False,
            "message": "Failed to upload file",
            "error": str(e)
        })


@app.get("/api/health")
async def health_check():
    """Health check endpoint for monitoring"""
    
    try:
        # Test database connection
        with get_db_session() as session:
            session.execute(text("SELECT 1"))
        
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


@app.get("/api/system")
async def get_system_status():
    """Return system-level health and resource information."""
    settings = get_settings()
    data_dir = Path(settings.paths.data_dir)
    upload_dir = Path(settings.paths.upload_dir)

    # DB health
    db_ok = False
    db_error: Optional[str] = None
    try:
        with get_db_session() as session:
            session.execute(text("SELECT 1"))
            db_ok = True
    except Exception as e:
        db_error = str(e)

    storage_bytes = _dir_size_bytes(data_dir)
    upload_files = 0
    if upload_dir.exists():
        upload_files = len([p for p in upload_dir.glob("**/*") if p.is_file()])

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "database": {
            "connected": db_ok,
            "error": db_error,
        },
        "storage": {
            "data_dir": str(data_dir),
            "bytes_used": storage_bytes,
            "pretty": _format_bytes(storage_bytes),
        },
        "uploads": {
            "path": str(upload_dir),
            "file_count": upload_files,
        },
    }

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
    # If the request likely expects JSON (e.g., XHR), return JSON
    try:
        accept = request.headers.get("accept", "")
        if "application/json" in accept:
            return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})
    except Exception:
        pass
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