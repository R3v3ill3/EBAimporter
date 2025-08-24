"""
Core configuration management using Pydantic Settings.
"""

import os
from pathlib import Path
from typing import Optional, List
from functools import lru_cache

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Database Configuration
    database_url: str = Field(
        default="postgresql://username:password@localhost:5432/ea_importer",
        env="DATABASE_URL"
    )
    db_host: str = Field(default="localhost", env="DB_HOST")
    db_port: int = Field(default=5432, env="DB_PORT")
    db_name: str = Field(default="ea_importer", env="DB_NAME")
    db_user: str = Field(default="username", env="DB_USER")
    db_password: str = Field(default="password", env="DB_PASSWORD")
    
    # Supabase Configuration (alternative)
    supabase_url: Optional[str] = Field(default=None, env="SUPABASE_URL")
    supabase_anon_key: Optional[str] = Field(default=None, env="SUPABASE_ANON_KEY")
    supabase_service_role_key: Optional[str] = Field(default=None, env="SUPABASE_SERVICE_ROLE_KEY")
    
    # File Storage
    data_root: Path = Field(default=Path("data"), env="DATA_ROOT")
    upload_max_size: str = Field(default="100MB", env="UPLOAD_MAX_SIZE")
    
    # OCR Configuration
    tesseract_cmd: str = Field(default="/usr/local/bin/tesseract", env="TESSERACT_CMD")
    ocr_language: str = Field(default="eng", env="OCR_LANGUAGE")
    ocr_dpi: int = Field(default=300, env="OCR_DPI")
    
    # ML Configuration
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    cluster_threshold_high: float = Field(default=0.95, env="CLUSTER_THRESHOLD_HIGH")
    cluster_threshold_medium: float = Field(default=0.90, env="CLUSTER_THRESHOLD_MEDIUM")
    cluster_threshold_low: float = Field(default=0.85, env="CLUSTER_THRESHOLD_LOW")
    min_clause_count: int = Field(default=60, env="MIN_CLAUSE_COUNT")
    
    # Jurisdiction Settings
    jurisdiction: str = Field(default="NSW + Federal (FWC/Fair Work)", env="JURISDICTION")
    target_version: str = Field(default="2025.08.v1", env="TARGET_VERSION")
    
    # Web Interface
    web_host: str = Field(default="0.0.0.0", env="WEB_HOST")
    web_port: int = Field(default=8000, env="WEB_PORT")
    secret_key: str = Field(default="dev-secret-key", env="SECRET_KEY")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    
    # External APIs
    fwc_base_url: str = Field(default="https://www.fwc.gov.au", env="FWC_BASE_URL")
    request_delay: float = Field(default=1.0, env="REQUEST_DELAY")
    
    # Development
    debug: bool = Field(default=False, env="DEBUG")
    testing: bool = Field(default=False, env="TESTING")
    
    @validator("data_root", pre=True)
    def validate_data_root(cls, v):
        """Ensure data_root is a Path object."""
        if isinstance(v, str):
            return Path(v)
        return v
    
    @property
    def family_thresholds(self) -> dict:
        """Get clustering thresholds as a dict."""
        return {
            "high": self.cluster_threshold_high,
            "medium": self.cluster_threshold_medium,
            "low": self.cluster_threshold_low,
        }
    
    @property
    def eas_dir(self) -> Path:
        """Get the EAs directory path."""
        return self.data_root / "eas"
    
    @property
    def raw_eas_dir(self) -> Path:
        """Get the raw EAs directory path."""
        return self.eas_dir / "raw"
    
    @property
    def text_dir(self) -> Path:
        """Get the text output directory path."""
        return self.eas_dir / "text"
    
    @property
    def clauses_dir(self) -> Path:
        """Get the clauses output directory path."""
        return self.eas_dir / "clauses"
    
    @property
    def fingerprints_dir(self) -> Path:
        """Get the fingerprints directory path."""
        return self.eas_dir / "fp"
    
    @property
    def embeddings_dir(self) -> Path:
        """Get the embeddings directory path."""
        return self.eas_dir / "emb"
    
    @property
    def families_dir(self) -> Path:
        """Get the families directory path."""
        return self.data_root / "families"
    
    @property
    def instances_dir(self) -> Path:
        """Get the instances directory path."""
        return self.data_root / "instances"
    
    @property
    def reports_dir(self) -> Path:
        """Get the reports directory path."""
        return self.data_root / "reports"
    
    @property
    def versions_dir(self) -> Path:
        """Get the versions directory path."""
        return self.data_root / "versions"
    
    def ensure_directories(self):
        """Create all required directories if they don't exist."""
        directories = [
            self.eas_dir,
            self.raw_eas_dir,
            self.text_dir,
            self.clauses_dir,
            self.fingerprints_dir,
            self.embeddings_dir,
            self.families_dir,
            self.instances_dir,
            self.reports_dir / "clusters",
            self.reports_dir / "qa",
            self.versions_dir,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()