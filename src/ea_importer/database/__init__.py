"""
Database module for EA Importer system.
Handles PostgreSQL connections, sessions, and database operations.
"""

from contextlib import contextmanager
from typing import Generator, Optional, Type, TypeVar, Union
import logging
import socket

from sqlalchemy import create_engine, MetaData, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from ..core.config import get_settings
from ..models import Base

# Type variables for generic database operations
ModelType = TypeVar("ModelType", bound=Base)

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Database connection and session management.
    """
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database manager.
        
        Args:
            database_url: Optional database URL override
        """
        self.settings = get_settings()
        self.database_url = database_url or self.settings.database.url
        self.engine: Optional[Engine] = None
        self.SessionLocal: Optional[sessionmaker] = None
        
    def initialize(self) -> None:
        """Initialize database engine and session factory."""
        logger.info(f"Initializing database connection to {self._safe_url()}")

        # Build connection args (SSL, timeouts, keepalive)
        connect_args: dict = {
            "connect_timeout": self.settings.database.connect_timeout,
        }

        # TCP keepalives (libpq/psycopg2)
        if self.settings.database.keepalives:
            connect_args.update({
                "keepalives": 1,
                "keepalives_idle": self.settings.database.keepalives_idle,
                "keepalives_interval": self.settings.database.keepalives_interval,
                "keepalives_count": self.settings.database.keepalives_count,
            })
        else:
            connect_args["keepalives"] = 0

        # SSL mode
        if self.settings.database.sslmode:
            connect_args["sslmode"] = self.settings.database.sslmode

        # Prefer IPv4: resolve host and supply hostaddr (libpq uses host for SNI)
        try:
            if self.settings.database.prefer_ipv4:
                url = make_url(self.database_url)
                if url.host:
                    infos = socket.getaddrinfo(url.host, url.port or 5432, family=socket.AF_INET, type=socket.SOCK_STREAM)
                    if infos:
                        ipv4_addr = infos[0][4][0]
                        connect_args["hostaddr"] = ipv4_addr
        except Exception as e:
            logger.warning(f"IPv4 resolution failed; proceeding without hostaddr: {e}")

        # Create engine with connection pooling
        self.engine = create_engine(
            self.database_url,
            echo=self.settings.database.echo,
            pool_size=self.settings.database.pool_size,
            max_overflow=self.settings.database.max_overflow,
            pool_pre_ping=True,  # Verify connections before use
            pool_recycle=self.settings.database.pool_recycle,
            connect_args=connect_args,
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        logger.info("Database connection initialized successfully")
    
    def _safe_url(self) -> str:
        """Return database URL with password masked for logging."""
        if "://" not in self.database_url:
            return self.database_url
            
        protocol, rest = self.database_url.split("://", 1)
        
        if "@" in rest:
            auth, host_path = rest.split("@", 1)
            if ":" in auth:
                user, _ = auth.split(":", 1)
                return f"{protocol}://{user}:***@{host_path}"
        
        return self.database_url
    
    def create_tables(self) -> None:
        """Create all database tables."""
        if not self.engine:
            raise RuntimeError("Database not initialized. Call initialize() first.")
            
        logger.info("Creating database tables")
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created successfully")
    
    def drop_tables(self) -> None:
        """Drop all database tables. Use with caution!"""
        if not self.engine:
            raise RuntimeError("Database not initialized. Call initialize() first.")
            
        logger.warning("Dropping all database tables")
        Base.metadata.drop_all(bind=self.engine)
        logger.info("Database tables dropped")
    
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        if not self.engine:
            return False
            
        inspector = inspect(self.engine)
        return table_name in inspector.get_table_names()
    
    def get_table_info(self) -> dict:
        """Get information about existing tables."""
        if not self.engine:
            return {}
            
        inspector = inspect(self.engine)
        table_info = {}
        
        for table_name in inspector.get_table_names():
            columns = inspector.get_columns(table_name)
            table_info[table_name] = {
                'columns': [col['name'] for col in columns],
                'column_count': len(columns)
            }
        
        return table_info
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Context manager for database sessions.
        
        Yields:
            Database session with automatic cleanup
        """
        if not self.SessionLocal:
            raise RuntimeError("Database not initialized. Call initialize() first.")
            
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def get_raw_session(self) -> Session:
        """
        Get a raw session without context management.
        Caller is responsible for session lifecycle.
        """
        if not self.SessionLocal:
            raise RuntimeError("Database not initialized. Call initialize() first.")
            
        return self.SessionLocal()
    
    def test_connection(self) -> bool:
        """
        Test database connection.
        
        Returns:
            True if connection is successful
        """
        try:
            from sqlalchemy import text
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False


# Global database manager instance
db_manager = DatabaseManager()


def get_database() -> DatabaseManager:
    """Get the global database manager instance."""
    return db_manager


def init_database(database_url: Optional[str] = None) -> DatabaseManager:
    """
    Initialize the database with optional URL override.
    
    Args:
        database_url: Optional database URL override
        
    Returns:
        Initialized database manager
    """
    if database_url:
        db_manager.database_url = database_url
    
    db_manager.initialize()
    return db_manager


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Convenience function for getting database sessions.
    
    Yields:
        Database session
    """
    with db_manager.get_session() as session:
        yield session


# Repository pattern base class
class BaseRepository:
    """
    Base repository class for database operations.
    """
    
    def __init__(self, session: Session, model_class: Type[ModelType]):
        self.session = session
        self.model_class = model_class
    
    def create(self, **kwargs) -> ModelType:
        """Create a new record."""
        instance = self.model_class(**kwargs)
        self.session.add(instance)
        self.session.flush()  # Get ID without committing
        return instance
    
    def get_by_id(self, record_id: int) -> Optional[ModelType]:
        """Get record by ID."""
        return self.session.query(self.model_class).filter(
            self.model_class.id == record_id
        ).first()
    
    def get_all(self, limit: Optional[int] = None) -> list[ModelType]:
        """Get all records with optional limit."""
        query = self.session.query(self.model_class)
        if limit:
            query = query.limit(limit)
        return query.all()
    
    def update(self, record_id: int, **kwargs) -> Optional[ModelType]:
        """Update record by ID."""
        instance = self.get_by_id(record_id)
        if instance:
            for key, value in kwargs.items():
                setattr(instance, key, value)
            self.session.flush()
        return instance
    
    def delete(self, record_id: int) -> bool:
        """Delete record by ID."""
        instance = self.get_by_id(record_id)
        if instance:
            self.session.delete(instance)
            self.session.flush()
            return True
        return False
    
    def count(self) -> int:
        """Count total records."""
        return self.session.query(self.model_class).count()
    
    def exists(self, **kwargs) -> bool:
        """Check if record exists with given criteria."""
        query = self.session.query(self.model_class)
        for key, value in kwargs.items():
            query = query.filter(getattr(self.model_class, key) == value)
        return query.first() is not None


# Database operation utilities
def create_test_database(test_db_url: str = "sqlite:///:memory:") -> DatabaseManager:
    """
    Create an in-memory test database.
    
    Args:
        test_db_url: Test database URL (defaults to in-memory SQLite)
        
    Returns:
        Test database manager
    """
    test_db = DatabaseManager(test_db_url)
    test_db.initialize()
    test_db.create_tables()
    return test_db


def reset_database() -> None:
    """
    Reset the database by dropping and recreating all tables.
    USE WITH CAUTION - THIS WILL DELETE ALL DATA!
    """
    logger.warning("Resetting database - all data will be lost!")
    db_manager.drop_tables()
    db_manager.create_tables()
    logger.info("Database reset completed")


def setup_database() -> None:
    """
    Set up the database for first-time use.
    """
    logger.info("Setting up database for first-time use")
    
    # Initialize connection
    db_manager.initialize()
    
    # Check if tables exist
    table_info = db_manager.get_table_info()
    
    if not table_info:
        logger.info("No existing tables found, creating new database schema")
        db_manager.create_tables()
    else:
        logger.info(f"Found existing tables: {list(table_info.keys())}")
    
    # Test connection
    if db_manager.test_connection():
        logger.info("Database setup completed successfully")
    else:
        raise RuntimeError("Database connection test failed")


# Export main components
__all__ = [
    "DatabaseManager",
    "BaseRepository", 
    "db_manager",
    "get_database",
    "init_database",
    "get_db_session",
    "create_test_database",
    "reset_database",
    "setup_database",
]