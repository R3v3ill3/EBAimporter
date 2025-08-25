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
    ClusterCandidate, ClusterConfidence
)
from ...pipeline.clustering import ClusteringEngine
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
        
        # Get all documents
        with get_db_session() as session:
            documents = session.query(DocumentDB).all()
            
            if not documents:
                raise HTTPException(status_code=400, detail="No documents found")
            
            # Recalculate clusters
            cluster_results = clustering_engine.cluster_documents(
                documents,
                algorithm=algorithm,
                similarity_threshold=similarity_threshold,
                min_cluster_size=min_cluster_size
            )
            
            # Clear existing candidates and create new ones
            session.query(ClusterCandidateDB).delete()
            
            for cluster in cluster_results:
                candidate = ClusterCandidateDB(
                    document_ids=cluster.document_ids,
                    confidence_score=cluster.confidence_score,
                    similarity_scores=cluster.similarity_scores,
                    clustering_algorithm=algorithm,
                    clustering_parameters={
                        "similarity_threshold": similarity_threshold,
                        "min_cluster_size": min_cluster_size
                    },
                    review_status='pending'
                )
                session.add(candidate)
            
            session.commit()
            
        logger.info(f"Recalculated clusters: {len(cluster_results)} candidates created")
        
        return JSONResponse({
            "success": True,
            "clusters_created": len(cluster_results),
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
    """Calculate similarity matrix for visualization"""
    
    # Simplified similarity calculation for demo
    # In production, this would use the actual fingerprinting system
    matrix = []
    
    for i, doc1 in enumerate(documents):
        row = []
        for j, doc2 in enumerate(documents):
            if i == j:
                similarity = 1.0
            else:
                # Placeholder similarity calculation
                # Would use actual fingerprint comparison
                similarity = 0.8 + (hash(f"{doc1.id}-{doc2.id}") % 20) / 100
            row.append(round(similarity, 3))
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