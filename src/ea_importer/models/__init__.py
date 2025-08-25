"""
Data models for EA Importer system.
Includes SQLAlchemy ORM models and Pydantic domain objects.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import json

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Boolean, Float, 
    ForeignKey, JSON, LargeBinary, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from sqlalchemy.sql import func
from pydantic import BaseModel, Field, validator
import numpy as np


# SQLAlchemy Base
Base = declarative_base()


# Enums for status tracking
class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class AgreementStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"
    DRAFT = "draft"


class OverlayType(str, Enum):
    REPLACE_TEXT = "replace_text"
    APPEND_TEXT = "append_text"
    ADD_CLAUSE = "add_clause"
    REMOVE_CLAUSE = "remove_clause"
    MODIFY_RATE = "modify_rate"


# ============================================================================
# SQLAlchemy ORM Models (Database Schema)
# ============================================================================

class IngestRun(Base):
    """Track ingestion pipeline runs"""
    __tablename__ = "ingest_runs"
    
    id = Column(Integer, primary_key=True)
    started_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime)
    commit_sha = Column(String(40))
    notes = Column(Text)
    status = Column(String(20), default=ProcessingStatus.PENDING.value)
    files_processed = Column(Integer, default=0)
    files_failed = Column(Integer, default=0)
    
    # Relationships
    documents = relationship("Document", back_populates="ingest_run")


class Document(Base):
    """Represents a processed EA document"""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True)
    ea_id = Column(String(100), unique=True, nullable=False, index=True)
    file_path = Column(String(500), nullable=False)
    original_filename = Column(String(255))
    file_size_bytes = Column(Integer)
    checksum_sha256 = Column(String(64), index=True)
    
    # Processing metadata
    ingest_run_id = Column(Integer, ForeignKey("ingest_runs.id"))
    processed_at = Column(DateTime, default=func.now())
    status = Column(String(20), default=ProcessingStatus.PENDING.value)
    created_at = Column(DateTime, default=func.now())
    
    # Document metadata
    total_pages = Column(Integer)
    total_clauses = Column(Integer)
    has_text_layer = Column(Boolean)
    ocr_used = Column(Boolean, default=False)
    
    # Legal metadata
    title = Column(String(500))
    fwc_id = Column(String(100), index=True)
    jurisdiction = Column(String(100))
    effective_from = Column(DateTime)
    effective_to = Column(DateTime)
    
    # Relationships
    ingest_run = relationship("IngestRun", back_populates="documents")
    clauses = relationship("Clause", back_populates="document", cascade="all, delete-orphan")
    fingerprints = relationship("DocumentFingerprint", back_populates="document", cascade="all, delete-orphan")
    family_memberships = relationship("FamilyMember", back_populates="document")

    # Backwards compatible attribute used by web layer
    @property
    def file_name(self) -> Optional[str]:
        return self.original_filename


class Clause(Base):
    """Individual clause within a document"""
    __tablename__ = "clauses"
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    clause_id = Column(String(50), nullable=False)  # e.g., "2.3.1.a"
    clause_number = Column(String(50))  # For UI ordering compatibility
    heading = Column(String(500))
    text = Column(Text, nullable=False)
    
    # Structure
    path_json = Column(JSON)  # Hierarchical path as JSON array
    level = Column(Integer, default=0)
    order_index = Column(Integer, nullable=False)
    
    # Content metadata
    hash_sha256 = Column(String(64), index=True)
    token_count = Column(Integer)
    char_count = Column(Integer)
    page_spans_json = Column(JSON)  # List of [start_page, end_page] ranges
    
    # Timing
    effective_from = Column(DateTime)
    effective_to = Column(DateTime)
    
    # Relationships
    document = relationship("Document", back_populates="clauses")
    
    # Indexes
    __table_args__ = (
        Index("idx_document_clause", "document_id", "clause_id"),
        Index("idx_clause_hash", "hash_sha256"),
    )


class DocumentFingerprint(Base):
    """Document fingerprints for similarity detection"""
    __tablename__ = "document_fingerprints"
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    
    # Fingerprint data
    minhash_signature = Column(LargeBinary)  # Serialized MinHash
    embedding_vector = Column(LargeBinary)   # Optional: serialized embedding
    
    # Metadata
    minhash_permutations = Column(Integer)
    ngram_size = Column(Integer)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    document = relationship("Document", back_populates="fingerprints")


class AgreementFamily(Base):
    """Family of related agreements"""
    __tablename__ = "agreement_families"
    
    id = Column(Integer, primary_key=True)
    title = Column(String(500), nullable=False)
    family_name = Column(String(255))
    jurisdiction = Column(String(100))
    version = Column(String(50))
    
    # Gold standard info
    gold_document_id = Column(Integer, ForeignKey("documents.id"))
    
    # Lifecycle
    effective_from = Column(DateTime)
    effective_to = Column(DateTime)
    checksum = Column(String(64))
    created_at = Column(DateTime, default=func.now())
    locked_at = Column(DateTime)
    updated_at = Column(DateTime)
    # Web UI fields
    document_ids = Column(JSON)
    gold_text = Column(JSON)
    similarity_stats = Column(JSON)
    quality_score = Column(Float)
    
    # Relationships
    gold_document = relationship("Document", foreign_keys=[gold_document_id])
    members = relationship("FamilyMember", back_populates="family")
    clauses = relationship("FamilyClause", back_populates="family")
    rates = relationship("FamilyRate", back_populates="family")
    rules = relationship("FamilyRule", back_populates="family")


class FamilyMember(Base):
    """Membership of documents in families"""
    __tablename__ = "family_members"
    
    id = Column(Integer, primary_key=True)
    family_id = Column(Integer, ForeignKey("agreement_families.id"), nullable=False)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    
    # Similarity metrics
    similarity_score = Column(Float)
    confidence_level = Column(String(20))  # high, medium, low
    
    # Review status
    human_confirmed = Column(Boolean, default=False)
    confirmed_by = Column(String(100))
    confirmed_at = Column(DateTime)
    
    # Relationships
    family = relationship("AgreementFamily", back_populates="members")
    document = relationship("Document", back_populates="family_memberships")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint("family_id", "document_id"),
    )


class FamilyClause(Base):
    """Gold standard clauses for a family"""
    __tablename__ = "family_clauses"
    
    id = Column(Integer, primary_key=True)
    family_id = Column(Integer, ForeignKey("agreement_families.id"), nullable=False)
    clause_id = Column(String(50), nullable=False)
    heading = Column(String(500))
    text = Column(Text, nullable=False)
    
    # Structure
    path_json = Column(JSON)
    hash_sha256 = Column(String(64))
    token_count = Column(Integer)
    page_spans_json = Column(JSON)
    
    # Source tracking
    source_document_id = Column(Integer, ForeignKey("documents.id"))
    source_clause_id = Column(String(50))
    
    # Lifecycle
    effective_from = Column(DateTime)
    effective_to = Column(DateTime)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    family = relationship("AgreementFamily", back_populates="clauses")
    source_document = relationship("Document", foreign_keys=[source_document_id])


class FamilyRate(Base):
    """Standardized rates for a family"""
    __tablename__ = "family_rates"
    
    id = Column(Integer, primary_key=True)
    family_id = Column(Integer, ForeignKey("agreement_families.id"), nullable=False)
    
    # Classification
    classification = Column(String(100), nullable=False)
    level = Column(String(50))
    
    # Rate information
    base_rate = Column(Float, nullable=False)
    unit = Column(String(20), default="hourly")  # hourly, weekly, annual
    
    # Timing
    effective_from = Column(DateTime)
    effective_to = Column(DateTime)
    
    # Source
    source_clause_id = Column(Integer, ForeignKey("family_clauses.id"))
    
    # Relationships
    family = relationship("AgreementFamily", back_populates="rates")
    source_clause = relationship("FamilyClause", foreign_keys=[source_clause_id])


class FamilyRule(Base):
    """Extracted rules for a family"""
    __tablename__ = "family_rules"
    
    id = Column(Integer, primary_key=True)
    family_id = Column(Integer, ForeignKey("agreement_families.id"), nullable=False)
    
    # Rule identification
    key = Column(String(100), nullable=False)  # e.g., "overtime_weekday", "penalty_sunday"
    rule_type = Column(String(50))  # penalty, allowance, overtime, etc.
    
    # Rule data (flexible JSON structure)
    jsonb_rule = Column(JSON, nullable=False)
    
    # Source
    source_clause_id = Column(Integer, ForeignKey("family_clauses.id"))
    
    # Lifecycle
    effective_from = Column(DateTime)
    effective_to = Column(DateTime)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    family = relationship("AgreementFamily", back_populates="rules")
    source_clause = relationship("FamilyClause", foreign_keys=[source_clause_id])


class AgreementInstance(Base):
    """Specific instances of agreements (employers)"""
    __tablename__ = "agreement_instances"
    
    id = Column(Integer, primary_key=True)
    family_id = Column(Integer, ForeignKey("agreement_families.id"), nullable=False)
    
    # Instance identification
    instance_key = Column(String(100), unique=True, nullable=False)
    employer_id = Column(String(100))
    fwc_id = Column(String(100), index=True)
    
    # Timing
    commencement = Column(DateTime)
    nominal_expiry = Column(DateTime)
    status = Column(String(20), default=AgreementStatus.ACTIVE.value)
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    family = relationship("AgreementFamily", foreign_keys=[family_id])
    parameters = relationship("InstanceParameter", back_populates="instance")
    overlays = relationship("InstanceOverlay", back_populates="instance")


class InstanceParameter(Base):
    """Parameters for agreement instances"""
    __tablename__ = "instance_parameters"
    
    id = Column(Integer, primary_key=True)
    instance_id = Column(Integer, ForeignKey("agreement_instances.id"), nullable=False)
    
    key = Column(String(100), nullable=False)  # employer_name, abn, pay_steps, etc.
    value = Column(Text)  # Can be JSON for complex values
    data_type = Column(String(20), default="string")  # string, json, number, date
    
    # Relationships
    instance = relationship("AgreementInstance", back_populates="parameters")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint("instance_id", "key"),
    )


class InstanceOverlay(Base):
    """Overlays for instance-specific modifications"""
    __tablename__ = "instance_overlays"
    
    id = Column(Integer, primary_key=True)
    instance_id = Column(Integer, ForeignKey("agreement_instances.id"), nullable=False)
    # Optional link to family for UI queries
    family_id = Column(Integer, ForeignKey("agreement_families.id"))
    clause_id = Column(String(50), nullable=False)
    
    overlay_type = Column(String(20), nullable=False)  # OverlayType enum
    payload_jsonb = Column(JSON, nullable=False)
    
    # Timing
    effective_from = Column(DateTime)
    effective_to = Column(DateTime)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    instance = relationship("AgreementInstance", back_populates="overlays")


class ClusteringRun(Base):
    """Track clustering analysis runs"""
    __tablename__ = "clustering_runs"
    
    id = Column(Integer, primary_key=True)
    run_id = Column(String(100), unique=True, nullable=False)
    
    # Parameters
    algorithm = Column(String(50))
    parameters_json = Column(JSON)
    
    # Results
    total_documents = Column(Integer)
    families_created = Column(Integer)
    singletons = Column(Integer)
    
    # Timing
    started_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime)
    status = Column(String(20), default=ProcessingStatus.PENDING.value)


# ---------------------------------------------------------------------------
# Backwards-compatible alias names expected by web layer
# ---------------------------------------------------------------------------

# Provide legacy-style aliases for ORM classes so imports like
# `from ea_importer.models import DocumentDB, ClauseDB, FingerprintDB, ...`
# resolve correctly in the web app and routes.
DocumentDB = Document
ClauseDB = Clause
FingerprintDB = DocumentFingerprint
FamilyDB = AgreementFamily
InstanceDB = AgreementInstance
OverlayDB = InstanceOverlay


# ---------------------------------------------------------------------------
# Additional web-layer expectations
# ---------------------------------------------------------------------------

class ClusterConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ClusterCandidateDB(Base):
    """Persisted cluster candidate for review in the web UI.

    This provides minimal fields used by the web routes for listing, sorting,
    and reviewing candidates. It is intentionally simple and can be extended
    later if needed.
    """
    __tablename__ = "cluster_candidates"

    id = Column(Integer, primary_key=True)
    # Store referenced document IDs as JSON array of integers
    document_ids = Column(JSON, nullable=False, default=list)
    # Confidence score used for ordering in UI
    confidence_score = Column(Float, default=0.0)
    # Review workflow
    review_status = Column(String(20), default="pending")  # pending/approved/rejected
    review_notes = Column(Text)
    reviewed_at = Column(DateTime)
    created_at = Column(DateTime, default=func.now())


# ============================================================================
# Pydantic Domain Models (for API and processing)
# ============================================================================

class PDFPage(BaseModel):
    """Represents a single page from a PDF"""
    page_number: int
    text: str
    bbox: Optional[Tuple[float, float, float, float]] = None
    has_images: bool = False
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True


class PDFDocument(BaseModel):
    """Represents a complete PDF document"""
    file_path: Path
    pages: List[PDFPage] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def total_pages(self) -> int:
        return len(self.pages)
    
    @property
    def full_text(self) -> str:
        return "\n".join(page.text for page in self.pages)
    
    @property
    def page_texts(self) -> List[str]:
        return [page.text for page in self.pages]
    
    class Config:
        arbitrary_types_allowed = True


class ClauseSegment(BaseModel):
    """Represents a segmented clause"""
    ea_id: str
    clause_id: str
    path: List[str] = Field(default_factory=list)
    heading: Optional[str] = None
    text: str
    hash_sha256: Optional[str] = None
    token_count: Optional[int] = None
    page_spans: List[Tuple[int, int]] = Field(default_factory=list)
    effective_from: Optional[datetime] = None
    effective_to: Optional[datetime] = None
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Clause text cannot be empty")
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class DocumentFingerprint(BaseModel):
    """Document fingerprint for similarity detection"""
    ea_id: str
    minhash_signature: bytes
    embedding_vector: Optional[np.ndarray] = None
    minhash_permutations: int = 128
    ngram_size: int = 5
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True


class SimilarityMatrix(BaseModel):
    """Similarity matrix for clustering"""
    document_ids: List[str]
    similarity_scores: np.ndarray
    algorithm: str
    threshold: float
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True


class ClusterCandidate(BaseModel):
    """Candidate family from clustering"""
    cluster_id: str
    document_ids: List[str]
    similarity_scores: List[float]
    confidence_level: str  # high, medium, low
    suggested_title: Optional[str] = None
    
    @validator('confidence_level')
    def validate_confidence(cls, v):
        if v not in ['high', 'medium', 'low']:
            raise ValueError("Confidence level must be high, medium, or low")
        return v


class FamilyCandidate(BaseModel):
    """Family building candidate"""
    family_id: str
    title: str
    gold_document_id: str
    member_document_ids: List[str]
    similarity_matrix: Dict[str, float]
    status: str = "pending_review"


class RateExtraction(BaseModel):
    """Extracted rate information"""
    classification: str
    level: Optional[str] = None
    base_rate: float
    unit: str = "hourly"
    source_clause_id: str
    effective_from: Optional[datetime] = None
    effective_to: Optional[datetime] = None


class RuleExtraction(BaseModel):
    """Extracted rule information"""
    key: str
    rule_type: str
    rule_data: Dict[str, Any]
    source_clause_id: str
    effective_from: Optional[datetime] = None
    effective_to: Optional[datetime] = None


class QATestResult(BaseModel):
    """Quality assurance test result"""
    test_id: str
    family_id: str
    worker_scenario: Dict[str, Any]
    test_results: Dict[str, Any]
    anomalies: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    passed: bool
    execution_time_seconds: float


class VersionManifest(BaseModel):
    """Version control manifest"""
    version: str
    created_at: datetime
    locked_at: Optional[datetime] = None
    commit_sha: Optional[str] = None
    families_count: int
    instances_count: int
    checksums: Dict[str, str] = Field(default_factory=dict)
    notes: Optional[str] = None