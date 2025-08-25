"""
Clustering Engine for EA Importer - Document family detection.
"""

import logging
import numpy as np
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

try:
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from ..core.config import get_settings, ClusteringAlgorithm
from ..core.logging import get_logger, log_function_call
from ..models import DocumentFingerprint, ClusterCandidate
from ..utils import Fingerprinter

logger = get_logger(__name__)


class ClusteringEngine:
    """Advanced clustering engine for document family detection."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.settings = get_settings()
        self.config = config or {}
        
        self.algorithm = self.config.get('algorithm', self.settings.clustering.algorithm)
        self.high_threshold = self.config.get('high_threshold', self.settings.clustering.high_similarity_threshold)
        self.medium_threshold = self.config.get('medium_threshold', self.settings.clustering.medium_similarity_threshold)
        
        self.fingerprinter = Fingerprinter()
    
    @log_function_call
    def cluster_documents(self, fingerprints: List[DocumentFingerprint], algorithm: Optional[str] = None) -> Dict[str, List[str]]:
        """Cluster documents based on similarity."""
        if not fingerprints:
            return {}
        
        algorithm = algorithm or self.algorithm
        logger.info(f"Clustering {len(fingerprints)} documents using {algorithm}")
        
        # Build similarity matrix
        similarity_matrix, doc_ids = self.fingerprinter.build_similarity_matrix(fingerprints)
        
        # Apply clustering
        if algorithm == ClusteringAlgorithm.MINHASH_THRESHOLD:
            clusters = self._threshold_clustering(similarity_matrix, doc_ids)
        elif algorithm == ClusteringAlgorithm.DBSCAN and HAS_SKLEARN:
            clusters = self._dbscan_clustering(similarity_matrix, doc_ids)
        else:
            clusters = self._threshold_clustering(similarity_matrix, doc_ids)
        
        logger.info(f"Clustering completed: {len(clusters)} clusters found")
        return clusters
    
    def _threshold_clustering(self, similarity_matrix: np.ndarray, doc_ids: List[str]) -> Dict[str, List[str]]:
        """MinHash threshold-based clustering"""
        clusters = {}
        assigned = set()
        cluster_counter = 1
        
        n = len(doc_ids)
        
        for i in range(n):
            if doc_ids[i] in assigned:
                continue
                
            cluster_docs = [doc_ids[i]]
            assigned.add(doc_ids[i])
            
            # Find similar documents
            for j in range(i + 1, n):
                if doc_ids[j] in assigned:
                    continue
                
                if similarity_matrix[i, j] >= self.high_threshold:
                    cluster_docs.append(doc_ids[j])
                    assigned.add(doc_ids[j])
            
            cluster_id = f"FAMILY_{cluster_counter:03d}" if len(cluster_docs) > 1 else f"SINGLETON_{cluster_counter:03d}"
            clusters[cluster_id] = cluster_docs
            cluster_counter += 1
        
        return clusters
    
    def _dbscan_clustering(self, similarity_matrix: np.ndarray, doc_ids: List[str]) -> Dict[str, List[str]]:
        """DBSCAN clustering"""
        distance_matrix = 1 - similarity_matrix
        
        dbscan = DBSCAN(eps=0.1, min_samples=2, metric='precomputed')
        cluster_labels = dbscan.fit_predict(distance_matrix)
        
        clusters = {}
        noise_counter = 1
        
        for i, label in enumerate(cluster_labels):
            doc_id = doc_ids[i]
            
            if label == -1:
                cluster_id = f"OUTLIER_{noise_counter:03d}"
                clusters[cluster_id] = [doc_id]
                noise_counter += 1
            else:
                cluster_id = f"FAMILY_{label + 1:03d}"
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(doc_id)
        
        return clusters
    
    def generate_cluster_candidates(self, clusters: Dict[str, List[str]], fingerprints: List[DocumentFingerprint]) -> List[ClusterCandidate]:
        """Generate cluster candidates with confidence scores"""
        candidates = []
        fp_dict = {fp.ea_id: fp for fp in fingerprints}
        
        for cluster_id, doc_ids in clusters.items():
            if len(doc_ids) < 1:
                continue
            
            # Calculate similarities within cluster
            similarities = []
            cluster_fps = [fp_dict[doc_id] for doc_id in doc_ids if doc_id in fp_dict]
            
            if len(cluster_fps) > 1:
                for i in range(len(cluster_fps)):
                    for j in range(i + 1, len(cluster_fps)):
                        sim = self.fingerprinter.calculate_similarity(cluster_fps[i], cluster_fps[j])
                        similarities.append(sim)
            
            # Determine confidence
            if similarities:
                avg_similarity = np.mean(similarities)
                confidence = "high" if avg_similarity >= self.high_threshold else "medium" if avg_similarity >= self.medium_threshold else "low"
            else:
                confidence = "singleton" if len(doc_ids) == 1 else "unknown"
            
            candidate = ClusterCandidate(
                cluster_id=cluster_id,
                document_ids=doc_ids,
                similarity_scores=similarities,
                confidence_level=confidence,
                suggested_title=f"Agreement Family ({len(doc_ids)} documents)"
            )
            
            candidates.append(candidate)
        
        return candidates
    
    def save_clustering_results(self, clusters: Dict[str, List[str]], output_dir: Path, run_id: Optional[str] = None) -> Path:
        """Save clustering results"""
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        run_dir = output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        clusters_file = run_dir / "clusters.json"
        with open(clusters_file, 'w') as f:
            json.dump({
                'run_id': run_id,
                'created_at': datetime.now().isoformat(),
                'clusters': clusters,
                'statistics': {
                    'total_clusters': len(clusters),
                    'total_documents': sum(len(docs) for docs in clusters.values()),
                }
            }, f, indent=2)
        
        logger.info(f"Clustering results saved to {run_dir}")
        return clusters_file


__all__ = ['ClusteringEngine']
