"""
Web Interface for EA Importer - Human-in-the-loop review and approval.
"""

from fastapi import FastAPI, Request, Depends, HTTPException, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from typing import List, Optional
import json
from pathlib import Path

from ..core.config import get_settings
from ..core.logging import get_logger
from ..database import get_db_session, get_database_manager
from ..models import AgreementFamily, FamilyClause, AgreementInstance
from ..pipeline.clustering import ClusteringResult
from ..utils.version_control import create_version_manager

# Create FastAPI app
app = FastAPI(
    title="EA Importer Web Interface",
    description="Human-in-the-loop review and approval for Enterprise Agreement processing",
    version="1.0.0"
)

# Set up templates and static files
templates_dir = Path(__file__).parent / "templates"
static_dir = Path(__file__).parent / "static"

templates = Jinja2Templates(directory=str(templates_dir))

# Create directories if they don't exist
templates_dir.mkdir(exist_ok=True)
static_dir.mkdir(exist_ok=True)

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

settings = get_settings()
logger = get_logger(__name__)


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, db: Session = Depends(get_db_session)):
    """Main dashboard page."""
    
    # Get statistics
    try:
        total_families = db.query(AgreementFamily).count()
        total_instances = db.query(AgreementInstance).count()
        total_clauses = db.query(FamilyClause).count()
        
        # Get recent families
        recent_families = db.query(AgreementFamily).order_by(
            AgreementFamily.created_at.desc()
        ).limit(5).all()
        
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {e}")
        total_families = total_instances = total_clauses = 0
        recent_families = []
    
    # Get version information
    version_manager = create_version_manager()
    versions = version_manager.list_versions()
    
    context = {
        "request": request,
        "total_families": total_families,
        "total_instances": total_instances,
        "total_clauses": total_clauses,
        "recent_families": recent_families,
        "versions": versions[:3],  # Show latest 3 versions
        "page_title": "Dashboard"
    }
    
    return templates.TemplateResponse("dashboard.html", context)


@app.get("/families", response_class=HTMLResponse)
async def list_families(request: Request, db: Session = Depends(get_db_session)):
    """List all EA families."""
    
    try:
        families = db.query(AgreementFamily).order_by(
            AgreementFamily.created_at.desc()
        ).all()
        
        # Add clause counts
        family_data = []
        for family in families:
            clause_count = db.query(FamilyClause).filter(
                FamilyClause.family_id == family.id
            ).count()
            
            family_data.append({
                'family': family,
                'clause_count': clause_count
            })
            
    except Exception as e:
        logger.error(f"Failed to get families: {e}")
        family_data = []
    
    context = {
        "request": request,
        "family_data": family_data,
        "page_title": "EA Families"
    }
    
    return templates.TemplateResponse("families.html", context)


@app.get("/families/{family_id}", response_class=HTMLResponse)
async def view_family(request: Request, family_id: str, db: Session = Depends(get_db_session)):
    """View details of a specific family."""
    
    try:
        family = db.query(AgreementFamily).filter(
            AgreementFamily.id == family_id
        ).first()
        
        if not family:
            raise HTTPException(status_code=404, detail="Family not found")
        
        # Get clauses
        clauses = db.query(FamilyClause).filter(
            FamilyClause.family_id == family.id
        ).order_by(FamilyClause.clause_id).all()
        
        # Get instances
        instances = db.query(AgreementInstance).filter(
            AgreementInstance.family_id == family.id
        ).all()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get family {family_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    
    context = {
        "request": request,
        "family": family,
        "clauses": clauses,
        "instances": instances,
        "page_title": f"Family: {family.title}"
    }
    
    return templates.TemplateResponse("family_detail.html", context)


@app.get("/clustering", response_class=HTMLResponse)
async def clustering_review(request: Request):
    """Review clustering results."""
    
    # Look for clustering reports
    reports_dir = settings.reports_dir / "clusters"
    clustering_reports = []
    
    if reports_dir.exists():
        for report_dir in reports_dir.iterdir():
            if report_dir.is_dir():
                clusters_file = report_dir / "clusters.json"
                if clusters_file.exists():
                    try:
                        with open(clusters_file, 'r') as f:
                            report_data = json.load(f)
                        clustering_reports.append({
                            'run_id': report_data.get('run_id', report_dir.name),
                            'algorithm': report_data.get('algorithm', 'unknown'),
                            'num_clusters': report_data.get('num_clusters', 0),
                            'num_documents': report_data.get('num_documents', 0),
                            'timestamp': report_data.get('timestamp', ''),
                            'path': str(report_dir)
                        })
                    except Exception as e:
                        logger.warning(f"Failed to read clustering report {clusters_file}: {e}")
    
    # Sort by timestamp (newest first)
    clustering_reports.sort(key=lambda x: x['timestamp'], reverse=True)
    
    context = {
        "request": request,
        "clustering_reports": clustering_reports,
        "page_title": "Clustering Review"
    }
    
    return templates.TemplateResponse("clustering.html", context)


@app.get("/clustering/{run_id}", response_class=HTMLResponse)
async def view_clustering_result(request: Request, run_id: str):
    """View specific clustering result."""
    
    report_dir = settings.reports_dir / "clusters" / run_id
    clusters_file = report_dir / "clusters.json"
    
    if not clusters_file.exists():
        raise HTTPException(status_code=404, detail="Clustering report not found")
    
    try:
        with open(clusters_file, 'r') as f:
            report_data = json.load(f)
        
        # Load family candidates CSV if available
        candidates_file = report_dir / "family_candidates.csv"
        candidates = []
        
        if candidates_file.exists():
            import pandas as pd
            candidates_df = pd.read_csv(candidates_file)
            candidates = candidates_df.to_dict('records')
        
    except Exception as e:
        logger.error(f"Failed to load clustering report: {e}")
        raise HTTPException(status_code=500, detail="Failed to load clustering report")
    
    context = {
        "request": request,
        "report_data": report_data,
        "candidates": candidates,
        "run_id": run_id,
        "page_title": f"Clustering Result: {run_id}"
    }
    
    return templates.TemplateResponse("clustering_detail.html", context)


@app.post("/clustering/{run_id}/approve")
async def approve_clustering(
    request: Request,
    run_id: str,
    approved_clusters: str = Form(...),
    db: Session = Depends(get_db_session)
):
    """Approve clustering results and create families."""
    
    try:
        # Parse approved clusters
        cluster_ids = [cid.strip() for cid in approved_clusters.split(',') if cid.strip()]
        
        # Load clustering result
        report_dir = settings.reports_dir / "clusters" / run_id
        clusters_file = report_dir / "clusters.json"
        
        with open(clusters_file, 'r') as f:
            report_data = json.load(f)
        
        # Create families for approved clusters
        # This would integrate with the family builder
        # For now, just log the approval
        logger.info(f"Approved clusters for {run_id}: {cluster_ids}")
        
        # Redirect back to clustering review
        return RedirectResponse(url="/clustering", status_code=303)
        
    except Exception as e:
        logger.error(f"Failed to approve clustering: {e}")
        raise HTTPException(status_code=500, detail="Failed to approve clustering")


@app.get("/versions", response_class=HTMLResponse)
async def list_versions(request: Request):
    """List all corpus versions."""
    
    version_manager = create_version_manager()
    versions = version_manager.list_versions()
    
    context = {
        "request": request,
        "versions": versions,
        "page_title": "Corpus Versions"
    }
    
    return templates.TemplateResponse("versions.html", context)


@app.post("/versions/{version}/lock")
async def lock_version(version: str):
    """Lock a corpus version."""
    
    version_manager = create_version_manager()
    success = version_manager.lock_corpus_version(version)
    
    if success:
        return {"status": "success", "message": f"Version {version} locked successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to lock version")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    
    # Check database connection
    try:
        db_manager = get_database_manager()
        db_healthy = db_manager.test_connection()
    except Exception:
        db_healthy = False
    
    # Check data directories
    data_dirs_exist = all([
        settings.data_root.exists(),
        settings.eas_dir.exists(),
        settings.reports_dir.exists()
    ])
    
    status = "healthy" if db_healthy and data_dirs_exist else "unhealthy"
    
    return {
        "status": status,
        "database": "healthy" if db_healthy else "unhealthy",
        "data_directories": "healthy" if data_dirs_exist else "unhealthy",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.web_host, port=settings.web_port)