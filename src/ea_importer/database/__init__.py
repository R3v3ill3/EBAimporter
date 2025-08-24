"""
Database connection and session management.
"""

from typing import Generator, Optional
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from ..core.config import get_settings
from ..core.logging import get_logger
from ..models import Base

logger = get_logger(__name__)


class DatabaseManager:
    """Manages database connections and sessions."""
    
    def __init__(self, database_url: Optional[str] = None):
        self.settings = get_settings()
        self.database_url = database_url or self.settings.database_url
        
        # Create engine
        engine_kwargs = {
            "echo": self.settings.debug,
            "future": True,
        }
        
        # For SQLite (testing), use special pool
        if self.database_url.startswith("sqlite"):
            engine_kwargs.update({
                "poolclass": StaticPool,
                "connect_args": {"check_same_thread": False},
            })
        
        self.engine = create_engine(self.database_url, **engine_kwargs)
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
            future=True
        )
        
        logger.info(f"Database manager initialized with URL: {self._mask_url(self.database_url)}")
    
    def _mask_url(self, url: str) -> str:
        """Mask sensitive parts of database URL for logging."""
        if "://" in url:
            scheme, rest = url.split("://", 1)
            if "@" in rest:
                creds, host_part = rest.split("@", 1)
                return f"{scheme}://***:***@{host_part}"
        return url
    
    def create_tables(self):
        """Create all tables in the database."""
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created successfully")
    
    def drop_tables(self):
        """Drop all tables in the database (use with caution!)."""
        logger.warning("Dropping all database tables...")
        Base.metadata.drop_all(bind=self.engine)
        logger.warning("All database tables dropped")
    
    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around a series of operations."""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False


@lru_cache()
def get_database_manager() -> DatabaseManager:
    """Get cached database manager instance."""
    return DatabaseManager()


def get_db_session() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI to get database sessions.
    
    Yields:
        Database session
    """
    db_manager = get_database_manager()
    session = db_manager.get_session()
    try:
        yield session
    finally:
        session.close()


# Utility functions for common database operations

def init_database(database_url: Optional[str] = None, drop_existing: bool = False):
    """
    Initialize the database with tables.
    
    Args:
        database_url: Optional database URL override
        drop_existing: Whether to drop existing tables first
    """
    db_manager = DatabaseManager(database_url) if database_url else get_database_manager()
    
    if drop_existing:
        db_manager.drop_tables()
    
    db_manager.create_tables()
    
    # Test connection
    if not db_manager.test_connection():
        raise RuntimeError("Failed to connect to database after initialization")


def get_metadata() -> MetaData:
    """Get SQLAlchemy metadata for migrations."""
    return Base.metadata