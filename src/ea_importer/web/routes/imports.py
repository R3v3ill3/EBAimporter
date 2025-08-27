"""
Imports routes for CSV/URL batch import in the web interface.
Provides pages and JSON endpoints to start and monitor batch import jobs.
"""

from __future__ import annotations

import asyncio
import csv
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.background import BackgroundTasks

from ...core.logging import get_logger
from ...core.config import get_settings
from ...database import get_db_session
from ...models import BatchImportJob, BatchImportResult
from ...utils.csv_batch_importer import CSVBatchImporter


logger = get_logger(__name__)
router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def imports_index(request: Request):
    """Render the imports index page with upload forms."""
    settings = get_settings()
    return request.app.state.templates.TemplateResponse(
        "imports/index.html",
        {
            "request": request,
            "page_title": "CSV Batch Import",
            "default_concurrency": 5,
            "auto_process_default": True,
            "upload_dir": str(settings.paths.upload_dir),
        },
    )


def _create_job(job_name: Optional[str], source_type: str, source_path: str, settings: Dict[str, Any]) -> int:
    """Create a BatchImportJob row and return its ID."""
    with get_db_session() as session:
        job = BatchImportJob(
            job_name=job_name or "Web_Import",
            source_type=source_type,
            source_path=source_path,
            status="running",
            total_items=0,
            processed_items=0,
            successful_items=0,
            failed_items=0,
            settings=settings,
        )
        session.add(job)
        session.flush()
        return job.id


def _background_run_import(csv_path: str, job_id: int, job_name: Optional[str], auto_process: bool, max_concurrent: int) -> None:
    """Run the async importer in a background thread/process context."""
    async def run() -> None:
        importer = CSVBatchImporter(max_concurrent=max_concurrent)
        try:
            await importer.import_from_csv(
                csv_file_path=csv_path,
                job_name=job_name,
                auto_process=auto_process,
                resume_job_id=job_id,
            )
        except Exception as e:
            logger.error(f"Background import failed for job {job_id}: {e}")

    try:
        asyncio.run(run())
    except RuntimeError:
        # If already in an event loop, schedule a task
        asyncio.create_task(run())


@router.post("/upload")
async def upload_csv(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    job_name: Optional[str] = Form(None),
    auto_process: bool = Form(True),
    max_concurrent: int = Form(5),
):
    """Handle CSV file upload and start a batch job in the background."""
    # Save uploaded file to a temp path
    suffix = ".csv" if not file.filename.endswith(".csv") else ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        csv_path = tmp.name
        content = await file.read()
        tmp.write(content)

    # Create DB job row up front, then run importer with resume_job_id
    job_id = _create_job(
        job_name=job_name,
        source_type="csv",
        source_path=csv_path,
        settings={"auto_process": auto_process, "max_concurrent": max_concurrent},
    )

    background_tasks.add_task(_background_run_import, csv_path, job_id, job_name, auto_process, max_concurrent)
    return RedirectResponse(url=f"/imports/jobs/{job_id}", status_code=303)


@router.post("/urls")
async def submit_urls(
    request: Request,
    background_tasks: BackgroundTasks,
    urls_text: str = Form(...),
    job_name: Optional[str] = Form(None),
    auto_process: bool = Form(True),
    max_concurrent: int = Form(5),
):
    """Handle pasted URL list and start a batch job."""
    urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
    if not urls:
        raise HTTPException(status_code=400, detail="No URLs provided")

    # Write temp CSV with [url,title]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="", encoding="utf-8") as tmp:
        writer = csv.writer(tmp)
        writer.writerow(["url", "title"])
        for i, u in enumerate(urls, start=1):
            writer.writerow([u, f"Document_{i}"])
        csv_path = tmp.name

    # Create job and schedule processing
    job_id = _create_job(
        job_name=job_name,
        source_type="url_list",
        source_path=csv_path,
        settings={"auto_process": auto_process, "max_concurrent": max_concurrent},
    )
    background_tasks.add_task(_background_run_import, csv_path, job_id, job_name, auto_process, max_concurrent)
    return RedirectResponse(url=f"/imports/jobs/{job_id}", status_code=303)


@router.get("/jobs", response_class=HTMLResponse)
async def list_jobs(request: Request, limit: int = 50):
    with get_db_session() as session:
        jobs = (
            session.query(BatchImportJob)
            .order_by(BatchImportJob.created_at.desc())
            .limit(limit)
            .all()
        )
    return request.app.state.templates.TemplateResponse(
        "imports/jobs.html",
        {"request": request, "jobs": jobs, "page_title": "Import Jobs"},
    )


@router.get("/jobs/{job_id}", response_class=HTMLResponse)
async def job_detail(request: Request, job_id: int):
    with get_db_session() as session:
        job = session.query(BatchImportJob).filter(BatchImportJob.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        recent_failed = (
            session.query(BatchImportResult)
            .filter(BatchImportResult.job_id == job_id, BatchImportResult.status == "failed")
            .limit(20)
            .all()
        )
        recent_success = (
            session.query(BatchImportResult)
            .filter(BatchImportResult.job_id == job_id, BatchImportResult.status == "success")
            .limit(20)
            .all()
        )
    return request.app.state.templates.TemplateResponse(
        "imports/job_detail.html",
        {
            "request": request,
            "job": job,
            "recent_failed": recent_failed,
            "recent_success": recent_success,
            "page_title": f"Job {job_id}",
        },
    )


@router.get("/api/jobs/{job_id}")
async def job_status(job_id: int):
    with get_db_session() as session:
        job = session.query(BatchImportJob).filter(BatchImportJob.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return {
            "id": job.id,
            "job_name": job.job_name,
            "status": job.status,
            "total_items": job.total_items,
            "processed_items": job.processed_items,
            "successful_items": job.successful_items,
            "failed_items": job.failed_items,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        }


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: int):
    with get_db_session() as session:
        job = session.query(BatchImportJob).filter(BatchImportJob.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status != "running":
            return JSONResponse({"ok": False, "message": "Job is not running"}, status_code=400)
        job.status = "cancelled"
        session.flush()
        return {"ok": True}


@router.post("/jobs/{job_id}/retry-failed")
async def retry_failed(job_id: int, background_tasks: BackgroundTasks, job_name: Optional[str] = Form(None)):
    # Gather failed URLs
    with get_db_session() as session:
        failed = (
            session.query(BatchImportResult)
            .filter(BatchImportResult.job_id == job_id, BatchImportResult.status == "failed")
            .all()
        )
        if not failed:
            return JSONResponse({"ok": False, "message": "No failed items to retry"}, status_code=400)

    # Write temp CSV
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="", encoding="utf-8") as tmp:
        writer = csv.writer(tmp)
        writer.writerow(["url", "title"])
        for i, r in enumerate(failed, start=1):
            writer.writerow([r.source_url, f"Retry_{i}"])
        csv_path = tmp.name

    # Create new job and schedule
    new_job_id = _create_job(
        job_name=job_name or f"Retry job {job_id}",
        source_type="retry_failed",
        source_path=csv_path,
        settings={"auto_process": True},
    )
    background_tasks.add_task(_background_run_import, csv_path, new_job_id, job_name, True, 5)
    return {"ok": True, "new_job_id": new_job_id}

