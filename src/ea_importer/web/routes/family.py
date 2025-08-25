"""
Family routes for EA Importer web interface.
Handles family management, gold text selection, and family merging workflows.
"""

from fastapi import APIRouter, HTTPException, Depends, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
from pathlib import Path

from ...core.logging import get_logger
from ...database import get_db_session
from ...models import (
    FamilyDB, DocumentDB, ClauseDB, FingerprintDB,
    InstanceDB, OverlayDB
)
from ...pipeline.family_builder import FamilyBuilder

logger = get_logger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


@router.get("/", response_class=HTMLResponse)
async def families_overview(request: Request):
    """Overview of all EA families"""
    
    try:
        with get_db_session() as session:
            families = session.query(FamilyDB).order_by(
                FamilyDB.created_at.desc()
            ).all()
            
            # Calculate family statistics
            family_stats = []
            for family in families:
                instances = session.query(InstanceDB).filter(
                    InstanceDB.family_id == family.id
                ).all()
                
                overlays = session.query(OverlayDB).filter(
                    OverlayDB.family_id == family.id
                ).all()
                
                family_stats.append({
                    'family': family,
                    'instance_count': len(instances),
                    'overlay_count': len(overlays),
                    'total_documents': len(family.document_ids)
                })
            
            # Overall statistics
            total_stats = {
                'total_families': len(families),
                'total_instances': sum(stat['instance_count'] for stat in family_stats),
                'total_overlays': sum(stat['overlay_count'] for stat in family_stats),
                'avg_family_size': sum(stat['total_documents'] for stat in family_stats) / len(families) if families else 0
            }
            
        return templates.TemplateResponse("families/overview.html", {
            "request": request,
            "family_stats": family_stats,
            "total_stats": total_stats,
            "page_title": "Family Management"
        })
        
    except Exception as e:
        logger.error(f"Error loading families overview: {e}")
        raise HTTPException(status_code=500, detail="Failed to load families overview")


@router.get("/{family_id}", response_class=HTMLResponse)
async def family_detail(request: Request, family_id: int):
    """Detailed view of a specific family"""
    
    try:
        with get_db_session() as session:
            family = session.query(FamilyDB).filter(
                FamilyDB.id == family_id
            ).first()
            
            if not family:
                raise HTTPException(status_code=404, detail="Family not found")
            
            # Get family documents
            documents = session.query(DocumentDB).filter(
                DocumentDB.id.in_(family.document_ids)
            ).all()
            
            # Get family instances
            instances = session.query(InstanceDB).filter(
                InstanceDB.family_id == family_id
            ).all()
            
            # Get family overlays
            overlays = session.query(OverlayDB).filter(
                OverlayDB.family_id == family_id
            ).all()
            
            # Get gold clauses for comparison
            gold_clauses = _get_gold_clauses(family, session)
            
            # Calculate family metrics
            metrics = _calculate_family_metrics(family, documents, instances, overlays)
            
        return templates.TemplateResponse("families/detail.html", {
            "request": request,
            "family": family,
            "documents": documents,
            "instances": instances,
            "overlays": overlays,
            "gold_clauses": gold_clauses,
            "metrics": metrics,
            "page_title": f"Family: {family.family_name}"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading family {family_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to load family")


@router.get("/{family_id}/gold-text", response_class=HTMLResponse)
async def family_gold_text_editor(request: Request, family_id: int):
    """Gold text editor for family"""
    
    try:
        with get_db_session() as session:
            family = session.query(FamilyDB).filter(
                FamilyDB.id == family_id
            ).first()
            
            if not family:
                raise HTTPException(status_code=404, detail="Family not found")
            
            # Get all clauses from family documents for comparison
            all_clauses = {}
            for doc_id in family.document_ids:
                clauses = session.query(ClauseDB).filter(
                    ClauseDB.document_id == doc_id
                ).order_by(ClauseDB.clause_number).all()
                all_clauses[doc_id] = clauses
            
            # Group clauses by clause number for comparison
            clause_groups = _group_clauses_by_number(all_clauses)
            
            # Get current gold text
            current_gold_text = family.gold_text or {}
            
        return templates.TemplateResponse("families/gold_text_editor.html", {
            "request": request,
            "family": family,
            "clause_groups": clause_groups,
            "current_gold_text": current_gold_text,
            "page_title": f"Gold Text Editor: {family.family_name}"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading gold text editor for family {family_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to load gold text editor")


@router.post("/{family_id}/gold-text")
async def update_gold_text(
    family_id: int,
    gold_text_data: str = Form(...)
):
    """Update family gold text"""
    
    try:
        # Parse gold text data
        try:
            gold_text = json.loads(gold_text_data)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid gold text JSON")
        
        with get_db_session() as session:
            family = session.query(FamilyDB).filter(
                FamilyDB.id == family_id
            ).first()
            
            if not family:
                raise HTTPException(status_code=404, detail="Family not found")
            
            # Update gold text
            family.gold_text = gold_text
            family.updated_at = datetime.utcnow()
            
            # Use FamilyBuilder to process the update
            family_builder = FamilyBuilder()
            family_builder.update_family_gold_text(family, gold_text)
            
            session.commit()
            
        logger.info(f"Updated gold text for family {family_id}")
        
        return JSONResponse({
            "success": True,
            "family_id": family_id,
            "message": "Gold text updated successfully"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating gold text for family {family_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update gold text")


@router.post("/{family_id}/merge")
async def merge_families(
    family_id: int,
    target_family_id: int = Form(...),
    merge_strategy: str = Form("intelligent")
):
    """Merge two families together"""
    
    try:
        if merge_strategy not in ['intelligent', 'manual', 'overwrite']:
            raise HTTPException(status_code=400, detail="Invalid merge strategy")
        
        with get_db_session() as session:
            source_family = session.query(FamilyDB).filter(
                FamilyDB.id == family_id
            ).first()
            
            target_family = session.query(FamilyDB).filter(
                FamilyDB.id == target_family_id
            ).first()
            
            if not source_family or not target_family:
                raise HTTPException(status_code=404, detail="One or both families not found")
            
            if source_family.id == target_family.id:
                raise HTTPException(status_code=400, detail="Cannot merge family with itself")
            
            # Use FamilyBuilder to perform merge
            family_builder = FamilyBuilder()
            merged_family = family_builder.merge_families(
                source_family, target_family, strategy=merge_strategy
            )
            
            session.commit()
            
        logger.info(f"Merged family {family_id} into family {target_family_id}")
        
        return JSONResponse({
            "success": True,
            "source_family_id": family_id,
            "target_family_id": target_family_id,
            "merged_family_id": merged_family.id,
            "merge_strategy": merge_strategy,
            "message": "Families merged successfully"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error merging families {family_id} and {target_family_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to merge families")


@router.post("/{family_id}/split")
async def split_family(
    family_id: int,
    document_ids: str = Form(...),
    new_family_name: str = Form(...)
):
    """Split documents from family into new family"""
    
    try:
        # Parse document IDs
        try:
            doc_ids = [int(id.strip()) for id in document_ids.split(',') if id.strip()]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid document IDs")
        
        with get_db_session() as session:
            family = session.query(FamilyDB).filter(
                FamilyDB.id == family_id
            ).first()
            
            if not family:
                raise HTTPException(status_code=404, detail="Family not found")
            
            # Validate document IDs belong to family
            invalid_docs = [doc_id for doc_id in doc_ids if doc_id not in family.document_ids]
            if invalid_docs:
                raise HTTPException(status_code=400, detail=f"Documents not in family: {invalid_docs}")
            
            # Use FamilyBuilder to perform split
            family_builder = FamilyBuilder()
            new_family = family_builder.split_family(
                family, doc_ids, new_family_name
            )
            
            session.commit()
            
        logger.info(f"Split {len(doc_ids)} documents from family {family_id} into new family {new_family.id}")
        
        return JSONResponse({
            "success": True,
            "original_family_id": family_id,
            "new_family_id": new_family.id,
            "new_family_name": new_family_name,
            "split_document_count": len(doc_ids),
            "message": "Family split successfully"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error splitting family {family_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to split family")


@router.delete("/{family_id}")
async def delete_family(family_id: int, confirm: bool = Form(False)):
    """Delete a family and optionally its instances"""
    
    try:
        if not confirm:
            raise HTTPException(status_code=400, detail="Deletion must be confirmed")
        
        with get_db_session() as session:
            family = session.query(FamilyDB).filter(
                FamilyDB.id == family_id
            ).first()
            
            if not family:
                raise HTTPException(status_code=404, detail="Family not found")
            
            # Delete related instances and overlays
            instances_deleted = session.query(InstanceDB).filter(
                InstanceDB.family_id == family_id
            ).delete()
            
            overlays_deleted = session.query(OverlayDB).filter(
                OverlayDB.family_id == family_id
            ).delete()
            
            # Delete family
            session.delete(family)
            session.commit()
            
        logger.info(f"Deleted family {family_id} with {instances_deleted} instances and {overlays_deleted} overlays")
        
        return JSONResponse({
            "success": True,
            "family_id": family_id,
            "instances_deleted": instances_deleted,
            "overlays_deleted": overlays_deleted,
            "message": "Family deleted successfully"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting family {family_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete family")


@router.get("/{family_id}/api/comparison")
async def get_family_comparison_data(family_id: int):
    """Get detailed comparison data for family documents"""
    
    try:
        with get_db_session() as session:
            family = session.query(FamilyDB).filter(
                FamilyDB.id == family_id
            ).first()
            
            if not family:
                raise HTTPException(status_code=404, detail="Family not found")
            
            # Get comparison data
            comparison_data = _generate_family_comparison_data(family, session)
            
        return comparison_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting comparison data for family {family_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get comparison data")


def _get_gold_clauses(family: FamilyDB, session) -> Dict[str, Any]:
    """Get gold clauses for family display"""
    
    gold_text = family.gold_text or {}
    
    # Convert gold text to display format
    gold_clauses = []
    for clause_num, clause_data in gold_text.items():
        gold_clauses.append({
            'clause_number': clause_num,
            'heading': clause_data.get('heading', ''),
            'text': clause_data.get('text', ''),
            'confidence': clause_data.get('confidence', 0.0),
            'source_doc_id': clause_data.get('source_doc_id')
        })
    
    return gold_clauses


def _calculate_family_metrics(
    family: FamilyDB, 
    documents: List[DocumentDB], 
    instances: List[InstanceDB], 
    overlays: List[OverlayDB]
) -> Dict[str, Any]:
    """Calculate metrics for family display"""
    
    metrics = {
        'document_count': len(documents),
        'instance_count': len(instances),
        'overlay_count': len(overlays),
        'total_clauses': 0,
        'gold_clause_count': len(family.gold_text or {}),
        'avg_similarity': family.similarity_stats.get('avg_similarity', 0.0) if family.similarity_stats else 0.0,
        'family_quality_score': family.quality_score or 0.0
    }
    
    return metrics


def _group_clauses_by_number(all_clauses: Dict[int, List]) -> Dict[str, List]:
    """Group clauses by clause number across documents"""
    
    clause_groups = {}
    
    for doc_id, clauses in all_clauses.items():
        for clause in clauses:
            clause_num = clause.clause_number
            if clause_num not in clause_groups:
                clause_groups[clause_num] = []
            
            clause_groups[clause_num].append({
                'document_id': doc_id,
                'clause': clause
            })
    
    return clause_groups


def _generate_family_comparison_data(family: FamilyDB, session) -> Dict[str, Any]:
    """Generate detailed comparison data for API"""
    
    # Get all documents and clauses
    documents = session.query(DocumentDB).filter(
        DocumentDB.id.in_(family.document_ids)
    ).all()
    
    all_clauses = {}
    for doc in documents:
        clauses = session.query(ClauseDB).filter(
            ClauseDB.document_id == doc.id
        ).all()
        all_clauses[doc.id] = clauses
    
    # Generate comparison matrix
    comparison_data = {
        'family_id': family.id,
        'documents': [{'id': doc.id, 'name': doc.file_name} for doc in documents],
        'clause_comparison': _generate_clause_comparison(all_clauses),
        'similarity_matrix': _calculate_document_similarity_matrix(documents),
        'gold_text': family.gold_text or {}
    }
    
    return comparison_data


def _generate_clause_comparison(all_clauses: Dict[int, List]) -> Dict[str, Any]:
    """Generate clause-by-clause comparison data"""
    
    # Group by clause number
    clause_groups = _group_clauses_by_number(all_clauses)
    
    comparison = {}
    for clause_num, clause_variants in clause_groups.items():
        comparison[clause_num] = {
            'variants': [
                {
                    'document_id': variant['document_id'],
                    'text': variant['clause'].text,
                    'heading': variant['clause'].heading
                }
                for variant in clause_variants
            ],
            'variance_score': _calculate_clause_variance(clause_variants)
        }
    
    return comparison


def _calculate_clause_variance(clause_variants: List[Dict]) -> float:
    """Calculate variance score for clause variants"""
    
    if len(clause_variants) <= 1:
        return 0.0
    
    # Simplified variance calculation based on text length differences
    texts = [variant['clause'].text for variant in clause_variants]
    lengths = [len(text) for text in texts]
    
    if not lengths:
        return 0.0
    
    avg_length = sum(lengths) / len(lengths)
    variance = sum((length - avg_length) ** 2 for length in lengths) / len(lengths)
    
    # Normalize to 0-1 scale
    return min(variance / avg_length if avg_length > 0 else 0, 1.0)


def _calculate_document_similarity_matrix(documents: List[DocumentDB]) -> List[List[float]]:
    """Calculate similarity matrix between documents"""
    
    # Simplified similarity calculation for demo
    matrix = []
    
    for i, doc1 in enumerate(documents):
        row = []
        for j, doc2 in enumerate(documents):
            if i == j:
                similarity = 1.0
            else:
                # Placeholder similarity - would use actual fingerprint comparison
                similarity = 0.7 + (hash(f"{doc1.id}-{doc2.id}") % 30) / 100
            row.append(round(similarity, 3))
        matrix.append(row)
    
    return matrix


__all__ = ['router']