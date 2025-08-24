"""
Database models for the EA Importer system.
"""

from datetime import datetime, date
from typing import Optional, Dict, Any, List
from enum import Enum as PyEnum

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Date, Boolean, Float,
    JSON, LargeBinary, ForeignKey, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.orm import DeclarativeBase, relationship, Session
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
import uuid

class Base(DeclarativeBase):
    pass


class OverlayType(PyEnum):
    """Types of overlays for instance-specific clause modifications."""
    REPLACE_TEXT = "replace_text"
    APPEND_TEXT = "append_text"
    ADD_CLAUSE = "add_clause"
    REMOVE_CLAUSE = "remove_clause"


class AgreementStatus(PyEnum):
    """Status of agreement instances."""
    ACTIVE = "active"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"
    PENDING = "pending"


class IngestRun(Base):
    """Track ingestion runs for auditing and provenance."""
    __tablename__ = "ingest_runs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    commit_sha = Column(String(40), nullable=True)
    notes = Column(Text, nullable=True)
    config_snapshot = Column(JSONB, nullable=True)
    
    # Statistics
    files_processed = Column(Integer, default=0)
    files_succeeded = Column(Integer, default=0)
    files_failed = Column(Integer, default=0)
    total_clauses = Column(Integer, default=0)
    
    # Relationships
    agreement_families = relationship("AgreementFamily", back_populates="ingest_run")
    
    def __repr__(self):
        return f"<IngestRun(id={self.id}, started_at={self.started_at})>"


class AgreementFamily(Base):
    """Groups of similar agreements sharing the same structure and clauses."""
    __tablename__ = "agreement_families"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(500), nullable=False)
    jurisdiction = Column(String(100), nullable=False)
    version = Column(String(50), nullable=False)
    effective_from = Column(Date, nullable=True)
    effective_to = Column(Date, nullable=True)
    checksum = Column(String(64), nullable=False)  # SHA256 of gold content
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    locked_at = Column(DateTime(timezone=True), nullable=True)
    
    # Metadata
    description = Column(Text, nullable=True)
    industry_sector = Column(String(200), nullable=True)
    coverage = Column(Text, nullable=True)
    source_documents = Column(JSONB, nullable=True)  # List of source PDFs
    
    # Foreign keys
    ingest_run_id = Column(UUID(as_uuid=True), ForeignKey("ingest_runs.id"))
    gold_document_id = Column(String(100), nullable=True)  # Reference to source EA
    
    # Relationships
    ingest_run = relationship("IngestRun", back_populates="agreement_families")
    clauses = relationship("FamilyClause", back_populates="family", cascade="all, delete-orphan")
    rates = relationship("FamilyRate", back_populates="family", cascade="all, delete-orphan")
    rules = relationship("FamilyRule", back_populates="family", cascade="all, delete-orphan")
    instances = relationship("AgreementInstance", back_populates="family")
    
    # Indexes
    __table_args__ = (
        Index("ix_family_jurisdiction_version", "jurisdiction", "version"),
        Index("ix_family_effective", "effective_from", "effective_to"),
        UniqueConstraint("title", "version", name="uq_family_title_version"),
    )
    
    def __repr__(self):
        return f"<AgreementFamily(id={self.id}, title='{self.title}', version='{self.version}')>"


class FamilyClause(Base):
    """Individual clauses within an agreement family."""
    __tablename__ = "family_clauses"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    family_id = Column(UUID(as_uuid=True), ForeignKey("agreement_families.id"), nullable=False)
    clause_id = Column(String(50), nullable=False)  # e.g., "2.3.1.a"
    heading = Column(String(500), nullable=True)
    text = Column(Text, nullable=False)
    path = Column(JSONB, nullable=False)  # Hierarchical path as array
    hash_sha256 = Column(String(64), nullable=False)
    tokens = Column(Integer, nullable=True)
    page_spans = Column(JSONB, nullable=True)  # [[start_page, end_page], ...]
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Metadata
    clause_type = Column(String(50), nullable=True)  # 'definition', 'rate', 'procedure', etc.
    importance_score = Column(Float, nullable=True)
    
    # Foreign keys
    family = relationship("AgreementFamily", back_populates="clauses")
    
    # Indexes
    __table_args__ = (
        Index("ix_clause_family_id", "family_id"),
        Index("ix_clause_hash", "hash_sha256"),
        UniqueConstraint("family_id", "clause_id", name="uq_family_clause"),
    )
    
    def __repr__(self):
        return f"<FamilyClause(family_id={self.family_id}, clause_id='{self.clause_id}')>"


class FamilyRate(Base):
    """Pay rates and classifications within an agreement family."""
    __tablename__ = "family_rates"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    family_id = Column(UUID(as_uuid=True), ForeignKey("agreement_families.id"), nullable=False)
    classification = Column(String(200), nullable=False)
    level = Column(String(50), nullable=True)
    base_rate = Column(Float, nullable=False)
    unit = Column(String(20), nullable=False, default="hour")  # hour, week, year
    effective_from = Column(Date, nullable=True)
    effective_to = Column(Date, nullable=True)
    source_clause_id = Column(UUID(as_uuid=True), ForeignKey("family_clauses.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Metadata
    description = Column(Text, nullable=True)
    conditions = Column(JSONB, nullable=True)  # Special conditions or qualifiers
    
    # Relationships
    family = relationship("AgreementFamily", back_populates="rates")
    source_clause = relationship("FamilyClause")
    
    # Indexes
    __table_args__ = (
        Index("ix_rate_family_classification", "family_id", "classification"),
        Index("ix_rate_effective", "effective_from", "effective_to"),
        CheckConstraint("base_rate > 0", name="ck_rate_positive"),
    )
    
    def __repr__(self):
        return f"<FamilyRate(family_id={self.family_id}, classification='{self.classification}', rate={self.base_rate})>"


class FamilyRule(Base):
    """Business rules extracted from agreement families."""
    __tablename__ = "family_rules"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    family_id = Column(UUID(as_uuid=True), ForeignKey("agreement_families.id"), nullable=False)
    key = Column(String(200), nullable=False)  # e.g., 'overtime_weekday', 'allowance_tool'
    jsonb_rule = Column(JSONB, nullable=False)  # Rule definition and parameters
    source_clause_id = Column(UUID(as_uuid=True), ForeignKey("family_clauses.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Metadata
    rule_type = Column(String(50), nullable=True)  # 'penalty', 'allowance', 'condition'
    priority = Column(Integer, default=0)
    description = Column(Text, nullable=True)
    
    # Relationships
    family = relationship("AgreementFamily", back_populates="rules")
    source_clause = relationship("FamilyClause")
    
    # Indexes
    __table_args__ = (
        Index("ix_rule_family_key", "family_id", "key"),
        UniqueConstraint("family_id", "key", name="uq_family_rule_key"),
    )
    
    def __repr__(self):
        return f"<FamilyRule(family_id={self.family_id}, key='{self.key}')>"


class AgreementInstance(Base):
    """Specific instances of agreements with employer and timing information."""
    __tablename__ = "agreements_instances"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    family_id = Column(UUID(as_uuid=True), ForeignKey("agreement_families.id"), nullable=False)
    employer_id = Column(String(100), nullable=True)  # Could be ABN or internal ID
    fwc_id = Column(String(50), nullable=True)  # Fair Work Commission ID
    commencement = Column(Date, nullable=True)
    nominal_expiry = Column(Date, nullable=True)
    status = Column(String(20), nullable=False, default=AgreementStatus.ACTIVE.value)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Metadata
    title = Column(String(500), nullable=True)
    employer_name = Column(String(300), nullable=True)
    coverage_description = Column(Text, nullable=True)
    
    # Relationships
    family = relationship("AgreementFamily", back_populates="instances")
    parameters = relationship("InstanceParam", back_populates="instance", cascade="all, delete-orphan")
    overlays = relationship("InstanceOverlay", back_populates="instance", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("ix_instance_family_employer", "family_id", "employer_id"),
        Index("ix_instance_fwc", "fwc_id"),
        Index("ix_instance_dates", "commencement", "nominal_expiry"),
    )
    
    def __repr__(self):
        return f"<AgreementInstance(id={self.id}, family_id={self.family_id}, employer='{self.employer_name}')>"


class InstanceParam(Base):
    """Parameters specific to agreement instances (employer details, pay steps, etc.)."""
    __tablename__ = "instance_params"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    instance_id = Column(UUID(as_uuid=True), ForeignKey("agreements_instances.id"), nullable=False)
    key = Column(String(100), nullable=False)  # e.g., 'employer_abn', 'pay_steps'
    value = Column(JSONB, nullable=False)  # JSON value (string, number, object, array)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    instance = relationship("AgreementInstance", back_populates="parameters")
    
    # Indexes
    __table_args__ = (
        Index("ix_param_instance_key", "instance_id", "key"),
        UniqueConstraint("instance_id", "key", name="uq_instance_param_key"),
    )
    
    def __repr__(self):
        return f"<InstanceParam(instance_id={self.instance_id}, key='{self.key}')>"


class InstanceOverlay(Base):
    """Overlays for instance-specific clause modifications."""
    __tablename__ = "instance_overlays"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    instance_id = Column(UUID(as_uuid=True), ForeignKey("agreements_instances.id"), nullable=False)
    clause_id = Column(String(50), nullable=False)  # References family clause
    overlay_type = Column(String(20), nullable=False)  # OverlayType enum
    payload_jsonb = Column(JSONB, nullable=False)  # Overlay content
    effective_from = Column(Date, nullable=True)
    effective_to = Column(Date, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Metadata
    description = Column(Text, nullable=True)
    applied_by = Column(String(100), nullable=True)  # User who applied overlay
    
    # Relationships
    instance = relationship("AgreementInstance", back_populates="overlays")
    
    # Indexes
    __table_args__ = (
        Index("ix_overlay_instance_clause", "instance_id", "clause_id"),
        Index("ix_overlay_effective", "effective_from", "effective_to"),
        CheckConstraint(
            f"overlay_type IN {tuple(t.value for t in OverlayType)}",
            name="ck_overlay_type"
        ),
    )
    
    def __repr__(self):
        return f"<InstanceOverlay(instance_id={self.instance_id}, clause_id='{self.clause_id}', type='{self.overlay_type}')>"


# Additional utility tables for clustering and QA

class ClusterRun(Base):
    """Track clustering operations."""
    __tablename__ = "cluster_runs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(String(100), nullable=False, unique=True)
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    algorithm = Column(String(50), nullable=False)  # e.g., 'minhash', 'hdbscan'
    parameters = Column(JSONB, nullable=False)
    num_documents = Column(Integer, nullable=False)
    num_clusters = Column(Integer, nullable=True)
    
    def __repr__(self):
        return f"<ClusterRun(run_id='{self.run_id}', algorithm='{self.algorithm}')>"


class DocumentFingerprint(Base):
    """Store document fingerprints for similarity comparison."""
    __tablename__ = "document_fingerprints"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ea_id = Column(String(100), nullable=False, unique=True)
    sha256_hash = Column(String(64), nullable=False)
    minhash_signature = Column(LargeBinary, nullable=False)  # Serialized MinHash
    embedding = Column(LargeBinary, nullable=True)  # Optional embedding vector
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Metadata
    file_path = Column(String(500), nullable=True)
    file_size = Column(Integer, nullable=True)
    num_pages = Column(Integer, nullable=True)
    num_clauses = Column(Integer, nullable=True)
    
    __table_args__ = (
        Index("ix_fingerprint_sha256", "sha256_hash"),
        Index("ix_fingerprint_ea_id", "ea_id"),
    )
    
    def __repr__(self):
        return f"<DocumentFingerprint(ea_id='{self.ea_id}', sha256='{self.sha256_hash[:16]}...')>"