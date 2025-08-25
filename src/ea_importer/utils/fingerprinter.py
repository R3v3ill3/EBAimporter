"""
Document Fingerprinting utility for EA Importer.

Handles multiple fingerprinting strategies for document similarity detection:
- SHA256 hashing for exact content matching
- MinHash signatures for approximate similarity
- Optional text embeddings for semantic similarity
- Shingle-based text preprocessing
- Efficient batch processing

Designed for large-scale document corpus analysis.
"""

import hashlib
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from pathlib import Path
import time
from datetime import datetime

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from datasketch import MinHashLSH, MinHash
    HAS_DATASKETCH = True
except ImportError:
    HAS_DATASKETCH = False

try:
    import mmh3
    HAS_MURMURHASH = True
except ImportError:
    HAS_MURMURHASH = False

# Optional: For embeddings
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

from ..core.config import get_settings
from ..core.logging import get_logger, log_function_call
from ..models import PDFDocument, ClauseSegment, DocumentFingerprint

logger = get_logger(__name__)


class FingerprintingError(Exception):
    """Custom exception for fingerprinting errors"""
    pass


class TextPreprocessor:
    """
    Text preprocessing for fingerprinting.
    """
    
    def __init__(self, ngram_size: int = 5, shingle_size: int = 3):
        """
        Initialize text preprocessor.
        
        Args:
            ngram_size: Size of character n-grams
            shingle_size: Size of word shingles
        """
        self.ngram_size = ngram_size
        self.shingle_size = shingle_size
    
    def create_shingles(self, text: str, shingle_type: str = "word") -> Set[str]:
        """
        Create shingles from text.
        
        Args:
            text: Input text
            shingle_type: Type of shingles ("word", "char", "mixed")
            
        Returns:
            Set of shingles
        """
        text = text.lower().strip()
        
        if shingle_type == "word":
            return self._create_word_shingles(text)
        elif shingle_type == "char":
            return self._create_char_shingles(text)
        elif shingle_type == "mixed":
            word_shingles = self._create_word_shingles(text)
            char_shingles = self._create_char_shingles(text)
            return word_shingles.union(char_shingles)
        else:
            raise ValueError(f"Unknown shingle type: {shingle_type}")
    
    def _create_word_shingles(self, text: str) -> Set[str]:
        """Create word-based shingles"""
        words = text.split()
        shingles = set()
        
        for i in range(len(words) - self.shingle_size + 1):
            shingle = " ".join(words[i:i + self.shingle_size])
            shingles.add(shingle)
        
        return shingles
    
    def _create_char_shingles(self, text: str) -> Set[str]:
        """Create character-based n-grams"""
        # Remove spaces for character n-grams
        text_no_spaces = text.replace(" ", "")
        shingles = set()
        
        for i in range(len(text_no_spaces) - self.ngram_size + 1):
            shingle = text_no_spaces[i:i + self.ngram_size]
            shingles.add(shingle)
        
        return shingles
    
    def preprocess_for_hashing(self, text: str) -> str:
        """
        Preprocess text for consistent hashing.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text for hashing
        """
        # Normalize whitespace and case
        text = " ".join(text.lower().split())
        
        # Remove punctuation for better similarity detection
        import string
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        return text


class Fingerprinter:
    """
    Advanced document fingerprinting with multiple similarity detection methods.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize fingerprinter.
        
        Args:
            config: Optional configuration override
        """
        self.settings = get_settings()
        self.config = config or {}
        
        # Fingerprinting parameters
        self.minhash_permutations = self.config.get('minhash_permutations', self.settings.fingerprint.minhash_permutations)
        self.ngram_size = self.config.get('ngram_size', self.settings.fingerprint.ngram_size)
        self.shingle_size = self.config.get('shingle_size', self.settings.fingerprint.shingle_size)
        
        # Initialize preprocessor
        self.preprocessor = TextPreprocessor(
            ngram_size=self.ngram_size,
            shingle_size=self.shingle_size
        )
        
        # Initialize embedding model if available
        self.embedding_model = None
        if HAS_SPACY:
            try:
                self.embedding_model = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model for embeddings")
            except OSError:
                logger.warning("spaCy model not found, embeddings disabled")
        
        # Check dependencies
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check and log available dependencies"""
        available = []
        missing = []
        
        if HAS_DATASKETCH:
            available.append("datasketch (MinHash)")
        else:
            missing.append("datasketch")
        
        if HAS_NUMPY:
            available.append("numpy")
        else:
            missing.append("numpy")
        
        if HAS_MURMURHASH:
            available.append("mmh3 (MurmurHash)")
        else:
            missing.append("mmh3")
        
        if HAS_SPACY and self.embedding_model:
            available.append("spacy (embeddings)")
        else:
            missing.append("spacy/embeddings")
        
        logger.info(f"Fingerprinting dependencies - Available: {available}")
        if missing:
            logger.warning(f"Missing optional dependencies: {missing}")
    
    @log_function_call
    def fingerprint_document(self, document: PDFDocument, include_embeddings: bool = False) -> DocumentFingerprint:
        """
        Create comprehensive fingerprint for a document.
        
        Args:
            document: PDFDocument to fingerprint
            include_embeddings: Whether to generate embeddings
            
        Returns:
            DocumentFingerprint object
        """
        ea_id = document.metadata.get('ea_id', 'UNKNOWN')
        logger.info(f"Fingerprinting document {ea_id}")
        
        # Get full text
        full_text = document.full_text
        
        if not full_text.strip():
            raise FingerprintingError(f"Document {ea_id} has no text content")
        
        # Create MinHash signature
        minhash_signature = self._create_minhash(full_text)
        
        # Create embeddings if requested
        embedding_vector = None
        if include_embeddings and self.embedding_model:
            embedding_vector = self._create_embedding(full_text)
        
        # Create fingerprint object
        fingerprint = DocumentFingerprint(
            ea_id=ea_id,
            minhash_signature=pickle.dumps(minhash_signature),
            embedding_vector=pickle.dumps(embedding_vector) if embedding_vector is not None else None,
            minhash_permutations=self.minhash_permutations,
            ngram_size=self.ngram_size,
            created_at=datetime.now()
        )
        
        logger.debug(f"Fingerprint created for {ea_id}")
        return fingerprint
    
    def _create_minhash(self, text: str) -> MinHash:
        """
        Create MinHash signature from text.
        
        Args:
            text: Input text
            
        Returns:
            MinHash object
        """
        if not HAS_DATASKETCH:
            raise FingerprintingError("datasketch library required for MinHash")
        
        # Create shingles
        shingles = self.preprocessor.create_shingles(text, "mixed")
        
        if not shingles:
            logger.warning("No shingles created from text")
            shingles = {text}  # Fallback to whole text
        
        # Create MinHash
        minhash = MinHash(num_perm=self.minhash_permutations)
        
        for shingle in shingles:
            minhash.update(shingle.encode('utf8'))
        
        return minhash
    
    def _create_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Create text embedding using spaCy.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector or None if not available
        """
        if not self.embedding_model or not HAS_NUMPY:
            return None
        
        try:
            # Truncate text if too long (spaCy has limits)
            max_chars = 1000000  # 1M characters
            if len(text) > max_chars:
                text = text[:max_chars]
                logger.warning("Text truncated for embedding generation")
            
            doc = self.embedding_model(text)
            return doc.vector
            
        except Exception as e:
            logger.warning(f"Failed to create embedding: {e}")
            return None
    
    def calculate_similarity(self, fp1: DocumentFingerprint, fp2: DocumentFingerprint, method: str = "minhash") -> float:
        """
        Calculate similarity between two document fingerprints.
        
        Args:
            fp1: First document fingerprint
            fp2: Second document fingerprint
            method: Similarity method ("minhash", "embedding", "combined")
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        if method == "minhash":
            return self._calculate_minhash_similarity(fp1, fp2)
        elif method == "embedding":
            return self._calculate_embedding_similarity(fp1, fp2)
        elif method == "combined":
            minhash_sim = self._calculate_minhash_similarity(fp1, fp2)
            embedding_sim = self._calculate_embedding_similarity(fp1, fp2)
            
            # Weighted combination (favor MinHash if embeddings not available)
            if embedding_sim is not None:
                return 0.7 * minhash_sim + 0.3 * embedding_sim
            else:
                return minhash_sim
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    def _calculate_minhash_similarity(self, fp1: DocumentFingerprint, fp2: DocumentFingerprint) -> float:
        """Calculate MinHash Jaccard similarity"""
        try:
            minhash1 = pickle.loads(fp1.minhash_signature)
            minhash2 = pickle.loads(fp2.minhash_signature)
            
            return minhash1.jaccard(minhash2)
            
        except Exception as e:
            logger.error(f"Failed to calculate MinHash similarity: {e}")
            return 0.0
    
    def _calculate_embedding_similarity(self, fp1: DocumentFingerprint, fp2: DocumentFingerprint) -> Optional[float]:
        """Calculate embedding cosine similarity"""
        if not fp1.embedding_vector or not fp2.embedding_vector or not HAS_NUMPY:
            return None
        
        try:
            emb1 = pickle.loads(fp1.embedding_vector)
            emb2 = pickle.loads(fp2.embedding_vector)
            
            # Calculate cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
            
        except Exception as e:
            logger.error(f"Failed to calculate embedding similarity: {e}")
            return None
    
    def calculate_sha256(self, text: str) -> str:
        """
        Calculate SHA256 hash of text.
        
        Args:
            text: Input text
            
        Returns:
            SHA256 hash as hexadecimal string
        """
        normalized_text = self.preprocessor.preprocess_for_hashing(text)
        return hashlib.sha256(normalized_text.encode('utf-8')).hexdigest()
    
    def fingerprint_clauses(self, clauses: List[ClauseSegment], include_embeddings: bool = False) -> Dict[str, DocumentFingerprint]:
        """
        Create fingerprints for individual clauses.
        
        Args:
            clauses: List of clause segments
            include_embeddings: Whether to generate embeddings
            
        Returns:
            Dictionary mapping clause IDs to fingerprints
        """
        logger.info(f"Fingerprinting {len(clauses)} clauses")
        
        fingerprints = {}
        
        for clause in clauses:
            try:
                # Create MinHash for clause
                minhash_signature = self._create_minhash(clause.text)
                
                # Create embeddings if requested
                embedding_vector = None
                if include_embeddings and self.embedding_model:
                    embedding_vector = self._create_embedding(clause.text)
                
                # Create fingerprint
                fingerprint = DocumentFingerprint(
                    ea_id=f"{clause.ea_id}#{clause.clause_id}",
                    minhash_signature=pickle.dumps(minhash_signature),
                    embedding_vector=pickle.dumps(embedding_vector) if embedding_vector is not None else None,
                    minhash_permutations=self.minhash_permutations,
                    ngram_size=self.ngram_size,
                    created_at=datetime.now()
                )
                
                fingerprints[clause.clause_id] = fingerprint
                
            except Exception as e:
                logger.error(f"Failed to fingerprint clause {clause.clause_id}: {e}")
        
        logger.info(f"Successfully fingerprinted {len(fingerprints)} clauses")
        return fingerprints
    
    def build_similarity_matrix(self, fingerprints: List[DocumentFingerprint], method: str = "minhash") -> Tuple[np.ndarray, List[str]]:
        """
        Build similarity matrix for a set of fingerprints.
        
        Args:
            fingerprints: List of document fingerprints
            method: Similarity calculation method
            
        Returns:
            Tuple of (similarity_matrix, document_ids)
        """
        if not HAS_NUMPY:
            raise FingerprintingError("numpy required for similarity matrix")
        
        logger.info(f"Building similarity matrix for {len(fingerprints)} documents")
        
        document_ids = [fp.ea_id for fp in fingerprints]
        n = len(fingerprints)
        similarity_matrix = np.zeros((n, n))
        
        # Calculate pairwise similarities
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    sim = self.calculate_similarity(fingerprints[i], fingerprints[j], method)
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim  # Symmetric matrix
        
        logger.info("Similarity matrix completed")
        return similarity_matrix, document_ids
    
    def find_near_duplicates(self, fingerprints: List[DocumentFingerprint], threshold: float = 0.95) -> List[Tuple[str, str, float]]:
        """
        Find near-duplicate documents based on similarity threshold.
        
        Args:
            fingerprints: List of document fingerprints
            threshold: Similarity threshold for near-duplicates
            
        Returns:
            List of (doc_id1, doc_id2, similarity) tuples
        """
        logger.info(f"Finding near-duplicates with threshold {threshold}")
        
        near_duplicates = []
        
        for i, fp1 in enumerate(fingerprints):
            for j, fp2 in enumerate(fingerprints[i+1:], i+1):
                similarity = self.calculate_similarity(fp1, fp2)
                
                if similarity >= threshold:
                    near_duplicates.append((fp1.ea_id, fp2.ea_id, similarity))
        
        logger.info(f"Found {len(near_duplicates)} near-duplicate pairs")
        return near_duplicates
    
    def save_fingerprint(self, fingerprint: DocumentFingerprint, output_dir: Path) -> Path:
        """
        Save fingerprint to disk.
        
        Args:
            fingerprint: Document fingerprint to save
            output_dir: Output directory
            
        Returns:
            Path to saved file
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create safe filename
        safe_ea_id = fingerprint.ea_id.replace('/', '_').replace('\\', '_')
        output_path = output_dir / f"{safe_ea_id}.fingerprint"
        
        with open(output_path, 'wb') as f:
            pickle.dump(fingerprint, f)
        
        return output_path
    
    def load_fingerprint(self, file_path: Path) -> DocumentFingerprint:
        """
        Load fingerprint from disk.
        
        Args:
            file_path: Path to fingerprint file
            
        Returns:
            DocumentFingerprint object
        """
        with open(file_path, 'rb') as f:
            return pickle.load(f)


class LSHIndex:
    """
    Locality-Sensitive Hashing index for fast similarity search.
    """
    
    def __init__(self, threshold: float = 0.8, num_perm: int = 128):
        """
        Initialize LSH index.
        
        Args:
            threshold: Similarity threshold for LSH
            num_perm: Number of permutations for MinHash
        """
        if not HAS_DATASKETCH:
            raise FingerprintingError("datasketch library required for LSH")
        
        self.threshold = threshold
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.fingerprints = {}  # Store fingerprints by ID
    
    def add_fingerprint(self, fingerprint: DocumentFingerprint):
        """Add fingerprint to LSH index"""
        minhash = pickle.loads(fingerprint.minhash_signature)
        self.lsh.insert(fingerprint.ea_id, minhash)
        self.fingerprints[fingerprint.ea_id] = fingerprint
    
    def query_similar(self, fingerprint: DocumentFingerprint) -> List[str]:
        """Find similar documents using LSH"""
        minhash = pickle.loads(fingerprint.minhash_signature)
        return list(self.lsh.query(minhash))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            'threshold': self.threshold,
            'num_perm': self.num_perm,
            'total_documents': len(self.fingerprints),
            'index_size': len(self.lsh.keys) if hasattr(self.lsh, 'keys') else 'unknown'
        }


# Utility functions for batch processing
def fingerprint_documents_batch(documents: List[PDFDocument], **kwargs) -> Dict[str, DocumentFingerprint]:
    """
    Fingerprint multiple documents in batch.
    
    Args:
        documents: List of PDFDocuments to fingerprint
        **kwargs: Arguments passed to Fingerprinter
        
    Returns:
        Dictionary mapping EA IDs to fingerprints
    """
    fingerprinter = Fingerprinter(**kwargs)
    results = {}
    
    for document in documents:
        ea_id = document.metadata.get('ea_id', 'UNKNOWN')
        try:
            fingerprint = fingerprinter.fingerprint_document(document)
            results[ea_id] = fingerprint
        except Exception as e:
            logger.error(f"Failed to fingerprint document {ea_id}: {e}")
    
    return results


def save_fingerprints_batch(fingerprints: Dict[str, DocumentFingerprint], output_dir: Path):
    """
    Save multiple fingerprints to disk in batch.
    
    Args:
        fingerprints: Dictionary of fingerprints to save
        output_dir: Output directory
    """
    fingerprinter = Fingerprinter()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for ea_id, fingerprint in fingerprints.items():
        try:
            fingerprinter.save_fingerprint(fingerprint, output_dir)
            logger.debug(f"Saved fingerprint for {ea_id}")
        except Exception as e:
            logger.error(f"Failed to save fingerprint for {ea_id}: {e}")


# Export main classes
__all__ = [
    'Fingerprinter',
    'LSHIndex',
    'TextPreprocessor',
    'FingerprintingError',
    'fingerprint_documents_batch',
    'save_fingerprints_batch',
]