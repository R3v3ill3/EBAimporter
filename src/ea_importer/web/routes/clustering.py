"""
Clustering routes for EA Importer web interface.
Handles clustering review, approval, and family generation workflows.
"""

from fastapi import APIRouter, HTTPException, Depends, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from typing import List, Dict, Any, Optional
import json
import logging
from pathlib import Path

from ...core.logging import get_logger
from ...database import get_db_session
from ...models import (
    DocumentDB, ClusterCandidateDB, FamilyDB, ClauseDB,
)
from ...pipeline.clustering import ClusteringEngine
from ...utils.fingerprinter import Fingerprinter
from ...pipeline.family_builder import FamilyBuilder

logger = get_logger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))


@router.get("/", response_class=HTMLResponse)
async def clustering_overview(request: Request):
    """Clustering overview showing all cluster candidates"""
    
    try:
        with get_db_session() as session:
            # Get all cluster candidates with status
            candidates = session.query(ClusterCandidateDB).order_by(
                ClusterCandidateDB.confidence_score.desc()
            ).all()
            
            # Group by confidence level
            high_confidence = [c for c in candidates if c.confidence_score >= 0.9]
            medium_confidence = [c for c in candidates if 0.7 <= c.confidence_score < 0.9]
            low_confidence = [c for c in candidates if c.confidence_score < 0.7]
            
            # Get statistics
            stats = {
                'total_candidates': len(candidates),
                'pending_review': len([c for c in candidates if c.review_status == 'pending']),
                'approved': len([c for c in candidates if c.review_status == 'approved']),
                'rejected': len([c for c in candidates if c.review_status == 'rejected']),
                'high_confidence': len(high_confidence),
                'medium_confidence': len(medium_confidence),
                'low_confidence': len(low_confidence)
            }
            
        return templates.TemplateResponse("clustering/overview.html", {
            "request": request,
            "candidates": candidates,
            "high_confidence": high_confidence,
            "medium_confidence": medium_confidence,
            "low_confidence": low_confidence,
            "stats": stats,
            "page_title": "Clustering Review"
        })
        
    except Exception as e:
        logger.error(f"Error loading clustering overview: {e}")
        raise HTTPException(status_code=500, detail="Failed to load clustering overview")


@router.get("/candidate/{candidate_id}", response_class=HTMLResponse)
async def cluster_candidate_detail(request: Request, candidate_id: int):
    """Detailed view of a cluster candidate for review"""
    
    try:
        with get_db_session() as session:
            candidate = session.query(ClusterCandidateDB).filter(
                ClusterCandidateDB.id == candidate_id
            ).first()
            
            if not candidate:
                raise HTTPException(status_code=404, detail="Cluster candidate not found")
            
            # Get documents in this cluster
            document_ids = candidate.document_ids
            documents = session.query(DocumentDB).filter(
                DocumentDB.id.in_(document_ids)
            ).all()
            
            # Get sample clauses from each document for comparison
            sample_clauses = {}
            for doc in documents:
                clauses = session.query(ClauseDB).filter(
                    ClauseDB.document_id == doc.id
                ).order_by(ClauseDB.clause_number).limit(5).all()
                sample_clauses[doc.id] = clauses
            
            # Calculate similarity matrix for visualization
            similarity_matrix = _calculate_similarity_matrix(documents)
            
        return templates.TemplateResponse("clustering/candidate_detail.html", {
            "request": request,
            "candidate": candidate,
            "documents": documents,
            "sample_clauses": sample_clauses,
            "similarity_matrix": similarity_matrix,
            "page_title": f"Cluster Candidate {candidate.id}"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading cluster candidate {candidate_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to load cluster candidate")


@router.post("/candidate/{candidate_id}/review")
async def review_cluster_candidate(
    candidate_id: int,
    action: str = Form(...),
    notes: str = Form(""),
    family_name: str = Form("")
):
    """Review and approve/reject a cluster candidate"""
    
    try:
        if action not in ['approve', 'reject']:
            raise HTTPException(status_code=400, detail="Invalid action")
        
        with get_db_session() as session:
            candidate = session.query(ClusterCandidateDB).filter(
                ClusterCandidateDB.id == candidate_id
            ).first()
            
            if not candidate:
                raise HTTPException(status_code=404, detail="Cluster candidate not found")
            
            # Update review status
            candidate.review_status = 'approved' if action == 'approve' else 'rejected'
            candidate.review_notes = notes
            candidate.reviewed_at = datetime.utcnow()
            
            # If approved, create family
            if action == 'approve':
                if not family_name:
                    family_name = f"Family-{candidate.id}"
                
                family_builder = FamilyBuilder()
                family = family_builder.create_family_from_candidate(
                    candidate, family_name
                )
                
                logger.info(f"Created family {family.id} from candidate {candidate_id}")
            
            session.commit()
            
        return JSONResponse({
            "success": True,
            "action": action,
            "candidate_id": candidate_id,
            "message": f"Cluster candidate {action}d successfully"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reviewing cluster candidate {candidate_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to review cluster candidate")


@router.post("/bulk-review")
async def bulk_review_candidates(
    request: Request,
    candidate_ids: str = Form(...),
    action: str = Form(...),
    notes: str = Form("")
):
    """Bulk approve/reject multiple cluster candidates"""
    
    try:
        if action not in ['approve', 'reject']:
            raise HTTPException(status_code=400, detail="Invalid action")
        
        # Parse candidate IDs
        try:
            ids = [int(id.strip()) for id in candidate_ids.split(',') if id.strip()]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid candidate IDs")
        
        processed_count = 0
        
        with get_db_session() as session:
            for candidate_id in ids:
                candidate = session.query(ClusterCandidateDB).filter(
                    ClusterCandidateDB.id == candidate_id
                ).first()
                
                if candidate and candidate.review_status == 'pending':
                    candidate.review_status = 'approved' if action == 'approve' else 'rejected'
                    candidate.review_notes = notes
                    candidate.reviewed_at = datetime.utcnow()
                    
                    # If approved, create family
                    if action == 'approve':
                        family_builder = FamilyBuilder()
                        family = family_builder.create_family_from_candidate(
                            candidate, f"Family-{candidate.id}"
                        )
                    
                    processed_count += 1
            
            session.commit()
        
        return JSONResponse({
            "success": True,
            "action": action,
            "processed_count": processed_count,
            "message": f"Bulk {action}d {processed_count} cluster candidates"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in bulk review: {e}")
        raise HTTPException(status_code=500, detail="Failed to process bulk review")


@router.get("/api/similarity/{candidate_id}")
async def get_similarity_data(candidate_id: int):
    """Get similarity data for cluster visualization"""
    
    try:
        with get_db_session() as session:
            candidate = session.query(ClusterCandidateDB).filter(
                ClusterCandidateDB.id == candidate_id
            ).first()
            
            if not candidate:
                raise HTTPException(status_code=404, detail="Cluster candidate not found")
            
            # Get documents and calculate similarity
            document_ids = candidate.document_ids
            documents = session.query(DocumentDB).filter(
                DocumentDB.id.in_(document_ids)
            ).all()
            
            similarity_data = _calculate_detailed_similarity(documents)
            
        return {
            "candidate_id": candidate_id,
            "documents": [{"id": doc.id, "name": doc.file_name} for doc in documents],
            "similarity_matrix": similarity_data["matrix"],
            "similarity_scores": similarity_data["scores"],
            "clustering_metrics": similarity_data["metrics"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting similarity data for {candidate_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get similarity data")


@router.post("/recalculate")
async def recalculate_clusters(
    similarity_threshold: float = Form(0.8),
    algorithm: str = Form("threshold"),
    min_cluster_size: int = Form(2)
):
    """Recalculate clusters with new parameters"""
    
    try:
        clustering_engine = ClusteringEngine()
        fingerprinter = Fingerprinter()

        # Get all documents that have fingerprints
        with get_db_session() as session:
            docs = session.query(DocumentDB).all()
            if not docs:
                raise HTTPException(status_code=400, detail="No documents found")

            # Load fingerprints for documents
            from ...models import FingerprintDB as FPDB
            fps_map = {}
            for doc in docs:
                fp_row = session.query(FPDB).filter(FPDB.document_id == doc.id).first()
                if not fp_row:
                    continue
                # Build pydantic DocumentFingerprint
                from ...models import DocumentFingerprint as FP
                fps_map[doc.ea_id] = FP(
                    ea_id=doc.ea_id,
                    minhash_signature=fp_row.minhash_signature,
                    embedding_vector=None,
                )

            if not fps_map:
                raise HTTPException(status_code=400, detail="No fingerprints found")

            fps_list = list(fps_map.values())
            clusters = clustering_engine.cluster_documents(fps_list, algorithm)

            # Clear existing candidates and create new ones
            session.query(ClusterCandidateDB).delete()

            # Map EA IDs to DB document IDs
            ea_to_id = {d.ea_id: d.id for d in docs}

            created = 0
            for _, ea_ids in clusters.items():
                doc_ids = [ea_to_id.get(ea) for ea in ea_ids if ea in ea_to_id]
                if not doc_ids:
                    continue
                # Compute confidence as average pairwise MinHash similarity
                sims = []
                for i in range(len(ea_ids)):
                    for j in range(i + 1, len(ea_ids)):
                        fp1 = fps_map.get(ea_ids[i])
                        fp2 = fps_map.get(ea_ids[j])
                        if fp1 and fp2:
                            sims.append(fingerprinter.calculate_similarity(fp1, fp2))
                confidence = sum(sims) / len(sims) if sims else 0.0
                session.add(ClusterCandidateDB(
                    document_ids=doc_ids,
                    confidence_score=float(confidence),
                    review_status='pending',
                ))
                created += 1

            session.commit()

        logger.info(f"Recalculated clusters: {created} candidates created")

        return JSONResponse({
            "success": True,
            "clusters_created": created,
            "algorithm": algorithm,
            "parameters": {
                "similarity_threshold": similarity_threshold,
                "min_cluster_size": min_cluster_size
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recalculating clusters: {e}")
        raise HTTPException(status_code=500, detail="Failed to recalculate clusters")


def _calculate_similarity_matrix(documents: List[DocumentDB]) -> List[List[float]]:
    """Calculate similarity matrix using stored fingerprints."""
    from ...database import get_db_session
    from ...models import FingerprintDB as FPDB
    from ...models import DocumentFingerprint as FP

    fingerprinter = Fingerprinter()
    matrix: List[List[float]] = []
    fps: List[Optional[FP]] = []

    with get_db_session() as session:
        for doc in documents:
            fp_row = session.query(FPDB).filter(FPDB.document_id == doc.id).first()
            if fp_row:
                fps.append(FP(ea_id=doc.ea_id, minhash_signature=fp_row.minhash_signature, embedding_vector=None))
            else:
                fps.append(None)

    n = len(documents)
    for i in range(n):
        row: List[float] = []
        for j in range(n):
            if i == j:
                row.append(1.0)
                continue
            if fps[i] and fps[j]:
                sim = fingerprinter.calculate_similarity(fps[i], fps[j])
            else:
                sim = 0.0
            row.append(round(float(sim), 3))
        matrix.append(row)

    return matrix


def _calculate_detailed_similarity(documents: List[DocumentDB]) -> Dict[str, Any]:
    """Calculate detailed similarity data for API response"""
    
    matrix = _calculate_similarity_matrix(documents)
    
    # Calculate summary statistics
    all_scores = [score for row in matrix for score in row if score < 1.0]
    
    metrics = {
        "avg_similarity": sum(all_scores) / len(all_scores) if all_scores else 0,
        "min_similarity": min(all_scores) if all_scores else 0,
        "max_similarity": max(all_scores) if all_scores else 0,
        "document_count": len(documents)
    }
    
    return {
        "matrix": matrix,
        "scores": all_scores,
        "metrics": metrics
    }


__all__ = ['router']