"""
Fingerprinting utilities for document similarity using SHA256 and MinHash.
"""

import hashlib
import pickle
from typing import List, Set, Union, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from datasketch import MinHashLSH, MinHash

from ..core.logging import LoggerMixin


@dataclass
class DocumentFingerprint:
    """Represents fingerprints for a document."""
    ea_id: str
    sha256_hash: str
    minhash_signature: MinHash
    text_length: int
    num_clauses: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            'ea_id': self.ea_id,
            'sha256_hash': self.sha256_hash,
            'minhash_bytes': pickle.dumps(self.minhash_signature),
            'text_length': self.text_length,
            'num_clauses': self.num_clauses,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DocumentFingerprint':
        """Create from dictionary."""
        return cls(
            ea_id=data['ea_id'],
            sha256_hash=data['sha256_hash'],
            minhash_signature=pickle.loads(data['minhash_bytes']),
            text_length=data['text_length'],
            num_clauses=data['num_clauses'],
        )


class TextFingerprinter(LoggerMixin):
    """Generates fingerprints for text documents using various hashing methods."""
    
    def __init__(self, 
                 ngram_size: int = 5,
                 num_perm: int = 128,
                 similarity_threshold: float = 0.8):
        """
        Initialize the fingerprinter.
        
        Args:
            ngram_size: Size of character n-grams for MinHash
            num_perm: Number of permutation functions for MinHash
            similarity_threshold: Threshold for LSH similarity
        """
        self.ngram_size = ngram_size
        self.num_perm = num_perm
        self.similarity_threshold = similarity_threshold
        
        # Initialize LSH for fast similarity queries
        self.lsh = MinHashLSH(threshold=similarity_threshold, num_perm=num_perm)
        
        self.logger.info(f"Fingerprinter initialized with {ngram_size}-grams, "
                        f"{num_perm} permutations, threshold {similarity_threshold}")
    
    def compute_sha256(self, text: str) -> str:
        """
        Compute SHA256 hash of text.
        
        Args:
            text: Input text
            
        Returns:
            SHA256 hash as hex string
        """
        # Normalize text for consistent hashing
        normalized_text = text.strip().lower()
        return hashlib.sha256(normalized_text.encode('utf-8')).hexdigest()
    
    def extract_ngrams(self, text: str, n: int = None) -> Set[str]:
        """
        Extract character n-grams from text.
        
        Args:
            text: Input text
            n: N-gram size (defaults to self.ngram_size)
            
        Returns:
            Set of n-grams
        """
        n = n or self.ngram_size
        
        # Normalize text
        normalized_text = text.lower().replace('\n', ' ').replace('\t', ' ')
        # Remove extra whitespace
        normalized_text = ' '.join(normalized_text.split())
        
        # Extract n-grams
        ngrams = set()
        for i in range(len(normalized_text) - n + 1):
            ngram = normalized_text[i:i + n]
            ngrams.add(ngram)
        
        return ngrams
    
    def compute_minhash(self, text: str) -> MinHash:
        """
        Compute MinHash signature for text.
        
        Args:
            text: Input text
            
        Returns:
            MinHash signature
        """
        # Extract n-grams
        ngrams = self.extract_ngrams(text)
        
        # Create MinHash
        minhash = MinHash(num_perm=self.num_perm)
        
        # Add n-grams to MinHash
        for ngram in ngrams:
            minhash.update(ngram.encode('utf-8'))
        
        return minhash
    
    def compute_jaccard_similarity(self, minhash1: MinHash, minhash2: MinHash) -> float:
        """
        Compute Jaccard similarity between two MinHash signatures.
        
        Args:
            minhash1: First MinHash signature
            minhash2: Second MinHash signature
            
        Returns:
            Jaccard similarity (0.0 to 1.0)
        """
        return minhash1.jaccard(minhash2)
    
    def fingerprint_document(self,
                           ea_id: str,
                           full_text: str,
                           num_clauses: int = None) -> DocumentFingerprint:
        """
        Generate complete fingerprint for a document.
        
        Args:
            ea_id: Document identifier
            full_text: Complete document text
            num_clauses: Number of clauses in document
            
        Returns:
            DocumentFingerprint object
        """
        self.logger.debug(f"Fingerprinting document {ea_id}")
        
        # Compute SHA256 hash
        sha256_hash = self.compute_sha256(full_text)
        
        # Compute MinHash signature
        minhash_signature = self.compute_minhash(full_text)
        
        fingerprint = DocumentFingerprint(
            ea_id=ea_id,
            sha256_hash=sha256_hash,
            minhash_signature=minhash_signature,
            text_length=len(full_text),
            num_clauses=num_clauses or 0
        )
        
        self.logger.debug(f"Generated fingerprint for {ea_id}: "
                         f"SHA256={sha256_hash[:16]}..., text_length={len(full_text)}")
        
        return fingerprint
    
    def fingerprint_clauses(self, clauses: List[Dict]) -> List[Tuple[str, str]]:
        """
        Generate SHA256 hashes for individual clauses.
        
        Args:
            clauses: List of clause dictionaries with 'clause_id' and 'text' keys
            
        Returns:
            List of (clause_id, sha256_hash) tuples
        """
        fingerprints = []
        
        for clause in clauses:
            clause_id = clause['clause_id']
            text = clause['text']
            
            sha256_hash = self.compute_sha256(text)
            fingerprints.append((clause_id, sha256_hash))
        
        return fingerprints
    
    def add_to_lsh(self, fingerprint: DocumentFingerprint):
        """
        Add document fingerprint to LSH index for fast similarity queries.
        
        Args:
            fingerprint: Document fingerprint to add
        """
        try:
            self.lsh.insert(fingerprint.ea_id, fingerprint.minhash_signature)
            self.logger.debug(f"Added {fingerprint.ea_id} to LSH index")
        except ValueError as e:
            # Document might already be in LSH
            self.logger.warning(f"Could not add {fingerprint.ea_id} to LSH: {e}")
    
    def query_similar_documents(self, 
                              fingerprint: DocumentFingerprint,
                              exclude_self: bool = True) -> List[str]:
        """
        Query LSH for similar documents.
        
        Args:
            fingerprint: Query document fingerprint
            exclude_self: Whether to exclude the query document from results
            
        Returns:
            List of similar document IDs
        """
        try:
            similar_docs = self.lsh.query(fingerprint.minhash_signature)
            
            if exclude_self and fingerprint.ea_id in similar_docs:
                similar_docs.remove(fingerprint.ea_id)
            
            self.logger.debug(f"Found {len(similar_docs)} similar documents to {fingerprint.ea_id}")
            return similar_docs
            
        except Exception as e:
            self.logger.error(f"LSH query failed for {fingerprint.ea_id}: {e}")
            return []
    
    def compute_similarity_matrix(self, 
                                fingerprints: List[DocumentFingerprint]) -> np.ndarray:
        """
        Compute pairwise similarity matrix for a list of documents.
        
        Args:
            fingerprints: List of document fingerprints
            
        Returns:
            Symmetric similarity matrix
        """
        n = len(fingerprints)
        similarity_matrix = np.zeros((n, n))
        
        self.logger.info(f"Computing similarity matrix for {n} documents")
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    similarity = self.compute_jaccard_similarity(
                        fingerprints[i].minhash_signature,
                        fingerprints[j].minhash_signature
                    )
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
        
        return similarity_matrix
    
    def find_duplicate_documents(self, 
                               fingerprints: List[DocumentFingerprint],
                               threshold: float = 0.95) -> List[List[str]]:
        """
        Find groups of duplicate/near-duplicate documents.
        
        Args:
            fingerprints: List of document fingerprints
            threshold: Similarity threshold for considering documents as duplicates
            
        Returns:
            List of duplicate groups (each group is a list of document IDs)
        """
        duplicate_groups = []
        processed_docs = set()
        
        similarity_matrix = self.compute_similarity_matrix(fingerprints)
        
        for i, fingerprint in enumerate(fingerprints):
            if fingerprint.ea_id in processed_docs:
                continue
            
            # Find all documents similar to this one
            duplicate_group = [fingerprint.ea_id]
            processed_docs.add(fingerprint.ea_id)
            
            for j, other_fingerprint in enumerate(fingerprints):
                if (i != j and 
                    other_fingerprint.ea_id not in processed_docs and
                    similarity_matrix[i, j] >= threshold):
                    
                    duplicate_group.append(other_fingerprint.ea_id)
                    processed_docs.add(other_fingerprint.ea_id)
            
            # Only add groups with more than one document
            if len(duplicate_group) > 1:
                duplicate_groups.append(duplicate_group)
        
        self.logger.info(f"Found {len(duplicate_groups)} duplicate groups")
        return duplicate_groups
    
    def save_fingerprint(self, fingerprint: DocumentFingerprint, file_path: str):
        """
        Save fingerprint to file.
        
        Args:
            fingerprint: Fingerprint to save
            file_path: Output file path
        """
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(fingerprint.to_dict(), f)
            self.logger.debug(f"Saved fingerprint for {fingerprint.ea_id} to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save fingerprint: {e}")
            raise
    
    def load_fingerprint(self, file_path: str) -> DocumentFingerprint:
        """
        Load fingerprint from file.
        
        Args:
            file_path: Input file path
            
        Returns:
            Loaded fingerprint
        """
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            return DocumentFingerprint.from_dict(data)
        except Exception as e:
            self.logger.error(f"Failed to load fingerprint from {file_path}: {e}")
            raise


class ClauseFingerprinter(LoggerMixin):
    """Specialized fingerprinting for individual clauses within documents."""
    
    def __init__(self, ngram_size: int = 3):
        """
        Initialize clause fingerprinter.
        
        Args:
            ngram_size: Size of n-grams for clause fingerprinting
        """
        self.ngram_size = ngram_size
        self.text_fingerprinter = TextFingerprinter(ngram_size=ngram_size)
    
    def fingerprint_clause(self, clause_text: str) -> Tuple[str, MinHash]:
        """
        Generate fingerprint for a single clause.
        
        Args:
            clause_text: Text content of the clause
            
        Returns:
            Tuple of (SHA256 hash, MinHash signature)
        """
        sha256_hash = self.text_fingerprinter.compute_sha256(clause_text)
        minhash_signature = self.text_fingerprinter.compute_minhash(clause_text)
        
        return sha256_hash, minhash_signature
    
    def compare_clauses(self, clause1_text: str, clause2_text: str) -> float:
        """
        Compare similarity between two clauses.
        
        Args:
            clause1_text: First clause text
            clause2_text: Second clause text
            
        Returns:
            Jaccard similarity score
        """
        _, minhash1 = self.fingerprint_clause(clause1_text)
        _, minhash2 = self.fingerprint_clause(clause2_text)
        
        return self.text_fingerprinter.compute_jaccard_similarity(minhash1, minhash2)
    
    def find_similar_clauses(self,
                           target_clause: str,
                           candidate_clauses: List[Tuple[str, str]],
                           threshold: float = 0.8) -> List[Tuple[str, float]]:
        """
        Find clauses similar to a target clause.
        
        Args:
            target_clause: Target clause text
            candidate_clauses: List of (clause_id, clause_text) tuples
            threshold: Similarity threshold
            
        Returns:
            List of (clause_id, similarity_score) tuples for similar clauses
        """
        _, target_minhash = self.fingerprint_clause(target_clause)
        similar_clauses = []
        
        for clause_id, clause_text in candidate_clauses:
            _, candidate_minhash = self.fingerprint_clause(clause_text)
            similarity = self.text_fingerprinter.compute_jaccard_similarity(
                target_minhash, candidate_minhash
            )
            
            if similarity >= threshold:
                similar_clauses.append((clause_id, similarity))
        
        # Sort by similarity (descending)
        similar_clauses.sort(key=lambda x: x[1], reverse=True)
        
        return similar_clauses


def create_text_fingerprinter(**kwargs) -> TextFingerprinter:
    """Factory function to create a text fingerprinter."""
    return TextFingerprinter(**kwargs)


def create_clause_fingerprinter(**kwargs) -> ClauseFingerprinter:
    """Factory function to create a clause fingerprinter."""
    return ClauseFingerprinter(**kwargs)