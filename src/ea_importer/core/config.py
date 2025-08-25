"""
Core configuration module for EA Importer system.
Handles all configuration settings using Pydantic for validation and type safety.
"""

import os
from pathlib import Path
from typing import Optional, Dict, List, Any
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from enum import Enum


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ClusteringAlgorithm(str, Enum):
    MINHASH_THRESHOLD = "minhash_threshold"
    DBSCAN = "dbscan"
    HDBSCAN = "hdbscan"
    AGGLOMERATIVE = "agglomerative"
    ADAPTIVE = "adaptive"


class DatabaseConfig(BaseSettings):
    """Database configuration settings"""
    url: str = Field(
        default="postgresql://localhost:5432/ea_importer",
        description="Database connection URL"
    )
    echo: bool = Field(default=False, description="Enable SQLAlchemy query logging")
    pool_size: int = Field(default=10, description="Connection pool size")
    max_overflow: int = Field(default=20, description="Maximum pool overflow")
    
    class Config:
        env_prefix = "DB_"


class OCRConfig(BaseSettings):
    """OCR processing configuration"""
    dpi: int = Field(default=300, description="OCR DPI resolution")
    language: str = Field(default="eng", description="Tesseract language")
    min_text_ratio: float = Field(default=0.1, description="Minimum text ratio to consider PDF has text layer")
    timeout: int = Field(default=300, description="OCR timeout in seconds")
    
    class Config:
        env_prefix = "OCR_"


class ProcessingConfig(BaseSettings):
    """Document processing configuration"""
    max_file_size_mb: int = Field(default=100, description="Maximum PDF file size in MB")
    min_clause_length: int = Field(default=10, description="Minimum clause length in characters")
    max_clause_length: int = Field(default=10000, description="Maximum clause length in characters")
    batch_size: int = Field(default=10, description="Batch processing size")
    max_workers: int = Field(default=4, description="Maximum number of worker processes")
    
    class Config:
        env_prefix = "PROC_"


class FingerprintConfig(BaseSettings):
    """Fingerprinting and hashing configuration"""
    minhash_permutations: int = Field(default=128, description="MinHash permutations")
    ngram_size: int = Field(default=5, description="N-gram size for fingerprinting")
    shingle_size: int = Field(default=3, description="Shingle size for text processing")
    
    class Config:
        env_prefix = "FP_"


class ClusteringConfig(BaseSettings):
    """Clustering algorithm configuration"""
    algorithm: ClusteringAlgorithm = Field(default=ClusteringAlgorithm.ADAPTIVE)
    
    # Threshold-based clustering
    high_similarity_threshold: float = Field(default=0.95, description="High similarity threshold")
    medium_similarity_threshold: float = Field(default=0.90, description="Medium similarity threshold")
    low_similarity_threshold: float = Field(default=0.85, description="Low similarity threshold")
    
    # DBSCAN parameters
    dbscan_eps: float = Field(default=0.1, description="DBSCAN epsilon parameter")
    dbscan_min_samples: int = Field(default=2, description="DBSCAN minimum samples")
    
    # HDBSCAN parameters
    hdbscan_min_cluster_size: int = Field(default=2, description="HDBSCAN minimum cluster size")
    hdbscan_min_samples: Optional[int] = Field(default=None, description="HDBSCAN minimum samples")
    
    # Agglomerative clustering
    agglomerative_n_clusters: Optional[int] = Field(default=None, description="Number of clusters for agglomerative")
    agglomerative_linkage: str = Field(default="ward", description="Linkage criterion")
    
    class Config:
        env_prefix = "CLUSTER_"


class WebConfig(BaseSettings):
    """Web interface configuration"""
    host: str = Field(default="localhost", description="Web server host")
    port: int = Field(default=8080, description="Web server port")
    reload: bool = Field(default=False, description="Enable auto-reload in development")
    secret_key: str = Field(default="your-secret-key-here", description="Secret key for sessions")
    
    class Config:
        env_prefix = "WEB_"


class PathConfig(BaseSettings):
    """File and directory path configuration"""
    
    # Base directories
    data_dir: Path = Field(default=Path("data"), description="Base data directory")
    reports_dir: Path = Field(default=Path("reports"), description="Reports output directory")
    versions_dir: Path = Field(default=Path("versions"), description="Version control directory")
    
    # Subdirectories will be created relative to data_dir
    @property
    def raw_pdfs_dir(self) -> Path:
        return self.data_dir / "eas" / "raw"
    
    @property
    def text_dir(self) -> Path:
        return self.data_dir / "eas" / "text"
    
    @property
    def clauses_dir(self) -> Path:
        return self.data_dir / "eas" / "clauses"
    
    @property
    def fingerprints_dir(self) -> Path:
        return self.data_dir / "eas" / "fp"
    
    @property
    def embeddings_dir(self) -> Path:
        return self.data_dir / "eas" / "emb"
    
    @property
    def families_dir(self) -> Path:
        return self.data_dir / "families"
    
    @property
    def instances_dir(self) -> Path:
        return self.data_dir / "instances"
    
    @property
    def clusters_reports_dir(self) -> Path:
        return self.reports_dir / "clusters"
    
    @property
    def qa_reports_dir(self) -> Path:
        return self.reports_dir / "qa"
    
    class Config:
        env_prefix = "PATH_"


class QAConfig(BaseSettings):
    """Quality Assurance and testing configuration"""
    synthetic_workers_per_family: int = Field(default=20, description="Number of synthetic workers per family")
    test_scenarios_per_worker: int = Field(default=5, description="Test scenarios per worker")
    max_anomaly_rate: float = Field(default=0.02, description="Maximum acceptable anomaly rate")
    timeout_seconds: int = Field(default=600, description="QA test timeout")
    
    class Config:
        env_prefix = "QA_"


class VersionControlConfig(BaseSettings):
    """Version control and corpus locking configuration"""
    default_version_pattern: str = Field(default="%Y%m%d.v%H%M", description="Default version naming pattern")
    lock_timeout_hours: int = Field(default=24, description="Lock timeout in hours")
    checksum_algorithm: str = Field(default="sha256", description="Checksum algorithm")
    
    class Config:
        env_prefix = "VERSION_"


class Settings(BaseSettings):
    """Main application settings"""
    
    # Environment and logging
    environment: str = Field(default="development", description="Application environment")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # Component configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    fingerprint: FingerprintConfig = Field(default_factory=FingerprintConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    paths: PathConfig = Field(default_factory=PathConfig)
    qa: QAConfig = Field(default_factory=QAConfig)
    version_control: VersionControlConfig = Field(default_factory=VersionControlConfig)
    
    # Jurisdiction and context
    jurisdiction: str = Field(default="NSW + Federal (FWC/Fair Work)", description="Legal jurisdiction")
    target_version: str = Field(default="2025.08.v1", description="Target corpus version")
    minimum_clause_count: int = Field(default=60, description="Minimum acceptable clause count")
    
    @validator('database', pre=True)
    def parse_database_config(cls, v):
        if isinstance(v, dict):
            return DatabaseConfig(**v)
        return v
    
    def create_directories(self):
        """Create all necessary directories"""
        directories = [
            self.paths.data_dir,
            self.paths.raw_pdfs_dir,
            self.paths.text_dir,
            self.paths.clauses_dir,
            self.paths.fingerprints_dir,
            self.paths.embeddings_dir,
            self.paths.families_dir,
            self.paths.instances_dir,
            self.paths.reports_dir,
            self.paths.clusters_reports_dir,
            self.paths.qa_reports_dir,
            self.paths.versions_dir,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Convenience function to get settings
def get_settings() -> Settings:
    return settings


# Initialize directories on import
settings.create_directories()