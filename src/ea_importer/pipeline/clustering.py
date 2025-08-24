"""
Clustering engine for grouping similar EAs into families.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import hdbscan

from ..core.config import get_settings
from ..core.logging import LoggerMixin
from ..utils.fingerprinter import DocumentFingerprint, TextFingerprinter


class ClusteringAlgorithm(Enum):
    """Available clustering algorithms."""
    MINHASH_THRESHOLD = "minhash_threshold"
    DBSCAN = "dbscan"
    HDBSCAN = "hdbscan"
    AGGLOMERATIVE = "agglomerative"


@dataclass
class ClusterCandidate:
    """Represents a potential cluster/family of documents."""
    cluster_id: str
    ea_ids: List[str]
    similarity_scores: Dict[str, float]  # Pairwise similarities within cluster
    centroid_ea_id: str  # Most representative document
    confidence_score: float
    size: int
    
    def __post_init__(self):
        self.size = len(self.ea_ids)


@dataclass
class ClusteringResult:
    """Results from a clustering operation."""
    run_id: str
    algorithm: ClusteringAlgorithm
    parameters: Dict[str, Any]
    clusters: List[ClusterCandidate]
    outliers: List[str]  # EAs that couldn't be clustered
    num_documents: int
    num_clusters: int
    silhouette_score: Optional[float]
    execution_time_seconds: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['algorithm'] = self.algorithm.value
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClusteringResult':
        """Create from dictionary."""
        data['algorithm'] = ClusteringAlgorithm(data['algorithm'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class EAClusterer(LoggerMixin):
    """Clusters Enterprise Agreements into families based on similarity."""
    
    def __init__(self):
        """Initialize the clusterer."""
        self.settings = get_settings()
        self.fingerprinter = TextFingerprinter()
        
    def cluster_by_minhash_threshold(self,
                                   fingerprints: List[DocumentFingerprint],
                                   threshold: float = 0.9) -> List[ClusterCandidate]:
        """
        Cluster documents using MinHash similarity with threshold.
        
        Args:
            fingerprints: List of document fingerprints
            threshold: Similarity threshold for clustering
            
        Returns:
            List of cluster candidates
        """
        self.logger.info(f"Clustering {len(fingerprints)} documents with threshold {threshold}")
        
        # Compute similarity matrix
        similarity_matrix = self.fingerprinter.compute_similarity_matrix(fingerprints)
        
        # Group documents based on threshold
        clusters = []
        processed = set()
        
        for i, fp in enumerate(fingerprints):
            if fp.ea_id in processed:
                continue
            
            # Find all documents similar to this one
            cluster_members = [fp.ea_id]
            cluster_similarities = {}
            processed.add(fp.ea_id)
            
            for j, other_fp in enumerate(fingerprints):
                if (i != j and 
                    other_fp.ea_id not in processed and
                    similarity_matrix[i, j] >= threshold):
                    
                    cluster_members.append(other_fp.ea_id)
                    cluster_similarities[f"{fp.ea_id}_{other_fp.ea_id}"] = similarity_matrix[i, j]
                    processed.add(other_fp.ea_id)
            
            # Calculate confidence score (average similarity within cluster)
            if len(cluster_members) > 1:
                similarities = []
                for k in range(len(cluster_members)):
                    for l in range(k + 1, len(cluster_members)):
                        # Find indices in original fingerprints list
                        k_idx = next(m for m, f in enumerate(fingerprints) 
                                   if f.ea_id == cluster_members[k])
                        l_idx = next(m for m, f in enumerate(fingerprints) 
                                   if f.ea_id == cluster_members[l])
                        similarities.append(similarity_matrix[k_idx, l_idx])
                
                confidence_score = np.mean(similarities) if similarities else 1.0
            else:
                confidence_score = 1.0
            
            # Choose centroid (document with highest average similarity to others)
            if len(cluster_members) > 1:
                centroid_scores = {}
                for member in cluster_members:
                    member_idx = next(m for m, f in enumerate(fingerprints) 
                                    if f.ea_id == member)
                    other_indices = [next(m for m, f in enumerate(fingerprints) 
                                        if f.ea_id == other_member)
                                   for other_member in cluster_members 
                                   if other_member != member]
                    
                    avg_similarity = np.mean([similarity_matrix[member_idx, other_idx] 
                                            for other_idx in other_indices])
                    centroid_scores[member] = avg_similarity
                
                centroid_ea_id = max(centroid_scores.keys(), 
                                   key=lambda x: centroid_scores[x])
            else:
                centroid_ea_id = cluster_members[0]
            
            cluster = ClusterCandidate(
                cluster_id=str(uuid.uuid4()),
                ea_ids=cluster_members,
                similarity_scores=cluster_similarities,
                centroid_ea_id=centroid_ea_id,
                confidence_score=confidence_score,
                size=len(cluster_members)
            )
            clusters.append(cluster)
        
        self.logger.info(f"Created {len(clusters)} clusters using threshold method")
        return clusters
    
    def cluster_by_dbscan(self,
                         fingerprints: List[DocumentFingerprint],
                         eps: float = 0.1,
                         min_samples: int = 2) -> List[ClusterCandidate]:
        """
        Cluster documents using DBSCAN algorithm.
        
        Args:
            fingerprints: List of document fingerprints
            eps: Maximum distance between two samples for clustering
            min_samples: Minimum number of samples in a neighborhood
            
        Returns:
            List of cluster candidates
        """
        self.logger.info(f"Clustering with DBSCAN (eps={eps}, min_samples={min_samples})")
        
        # Compute similarity matrix and convert to distance matrix
        similarity_matrix = self.fingerprinter.compute_similarity_matrix(fingerprints)
        distance_matrix = 1.0 - similarity_matrix
        
        # Apply DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        return self._labels_to_clusters(fingerprints, cluster_labels, similarity_matrix)
    
    def cluster_by_hdbscan(self,
                          fingerprints: List[DocumentFingerprint],
                          min_cluster_size: int = 2,
                          min_samples: int = 1) -> List[ClusterCandidate]:
        """
        Cluster documents using HDBSCAN algorithm.
        
        Args:
            fingerprints: List of document fingerprints
            min_cluster_size: Minimum size of clusters
            min_samples: Minimum samples in core region
            
        Returns:
            List of cluster candidates
        """
        self.logger.info(f"Clustering with HDBSCAN (min_cluster_size={min_cluster_size})")
        
        # Compute similarity matrix and convert to distance matrix
        similarity_matrix = self.fingerprinter.compute_similarity_matrix(fingerprints)
        distance_matrix = 1.0 - similarity_matrix
        
        # Apply HDBSCAN
        clustering = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='precomputed'
        )
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        return self._labels_to_clusters(fingerprints, cluster_labels, similarity_matrix)
    
    def cluster_by_agglomerative(self,
                               fingerprints: List[DocumentFingerprint],
                               n_clusters: int = None,
                               linkage: str = 'average') -> List[ClusterCandidate]:
        """
        Cluster documents using Agglomerative clustering.
        
        Args:
            fingerprints: List of document fingerprints
            n_clusters: Number of clusters (if None, will be estimated)
            linkage: Linkage criterion
            
        Returns:
            List of cluster candidates
        """
        if n_clusters is None:
            # Estimate number of clusters (heuristic: sqrt of number of documents)
            n_clusters = max(2, int(np.sqrt(len(fingerprints))))
        
        self.logger.info(f"Clustering with Agglomerative (n_clusters={n_clusters}, linkage={linkage})")
        
        # Compute similarity matrix and convert to distance matrix
        similarity_matrix = self.fingerprinter.compute_similarity_matrix(fingerprints)
        distance_matrix = 1.0 - similarity_matrix
        
        # Apply Agglomerative clustering
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            metric='precomputed'
        )
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        return self._labels_to_clusters(fingerprints, cluster_labels, similarity_matrix)
    
    def _labels_to_clusters(self,
                          fingerprints: List[DocumentFingerprint],
                          cluster_labels: np.ndarray,
                          similarity_matrix: np.ndarray) -> List[ClusterCandidate]:
        """
        Convert cluster labels to ClusterCandidate objects.
        
        Args:
            fingerprints: List of document fingerprints
            cluster_labels: Cluster labels from sklearn
            similarity_matrix: Pairwise similarity matrix
            
        Returns:
            List of cluster candidates
        """
        clusters = []
        
        # Group documents by cluster label
        unique_labels = set(cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # Outliers (for DBSCAN/HDBSCAN)
                continue
            
            # Get indices of documents in this cluster
            cluster_indices = np.where(cluster_labels == label)[0]
            cluster_members = [fingerprints[i].ea_id for i in cluster_indices]
            
            if len(cluster_members) < 2:
                continue  # Skip singleton clusters
            
            # Calculate pairwise similarities within cluster
            cluster_similarities = {}
            similarities = []
            for i, idx1 in enumerate(cluster_indices):
                for j, idx2 in enumerate(cluster_indices):
                    if i < j:
                        sim = similarity_matrix[idx1, idx2]
                        similarities.append(sim)
                        cluster_similarities[f"{fingerprints[idx1].ea_id}_{fingerprints[idx2].ea_id}"] = sim
            
            # Calculate confidence score
            confidence_score = np.mean(similarities) if similarities else 1.0
            
            # Choose centroid
            centroid_scores = {}
            for idx in cluster_indices:
                other_indices = [other_idx for other_idx in cluster_indices if other_idx != idx]
                avg_similarity = np.mean([similarity_matrix[idx, other_idx] 
                                        for other_idx in other_indices]) if other_indices else 1.0
                centroid_scores[fingerprints[idx].ea_id] = avg_similarity
            
            centroid_ea_id = max(centroid_scores.keys(), key=lambda x: centroid_scores[x])
            
            cluster = ClusterCandidate(
                cluster_id=str(uuid.uuid4()),
                ea_ids=cluster_members,
                similarity_scores=cluster_similarities,
                centroid_ea_id=centroid_ea_id,
                confidence_score=confidence_score,
                size=len(cluster_members)
            )
            clusters.append(cluster)
        
        return clusters
    
    def adaptive_clustering(self,
                          fingerprints: List[DocumentFingerprint]) -> ClusteringResult:
        """
        Perform adaptive clustering using multiple algorithms and thresholds.
        
        Args:
            fingerprints: List of document fingerprints
            
        Returns:
            Best clustering result
        """
        self.logger.info("Performing adaptive clustering with multiple methods")
        
        start_time = datetime.now()
        
        # Try different clustering approaches
        clustering_attempts = [
            # High threshold for very similar documents
            (ClusteringAlgorithm.MINHASH_THRESHOLD, {'threshold': 0.95}),
            # Medium threshold for similar documents
            (ClusteringAlgorithm.MINHASH_THRESHOLD, {'threshold': 0.90}),
            # Lower threshold for related documents
            (ClusteringAlgorithm.MINHASH_THRESHOLD, {'threshold': 0.85}),
            # HDBSCAN for automatic cluster discovery
            (ClusteringAlgorithm.HDBSCAN, {'min_cluster_size': 2}),
            # DBSCAN with conservative parameters
            (ClusteringAlgorithm.DBSCAN, {'eps': 0.15, 'min_samples': 2}),
        ]
        
        best_result = None
        best_score = -1
        
        for algorithm, params in clustering_attempts:
            try:
                self.logger.info(f"Trying {algorithm.value} with params {params}")
                
                if algorithm == ClusteringAlgorithm.MINHASH_THRESHOLD:
                    clusters = self.cluster_by_minhash_threshold(fingerprints, **params)
                elif algorithm == ClusteringAlgorithm.HDBSCAN:
                    clusters = self.cluster_by_hdbscan(fingerprints, **params)
                elif algorithm == ClusteringAlgorithm.DBSCAN:
                    clusters = self.cluster_by_dbscan(fingerprints, **params)
                elif algorithm == ClusteringAlgorithm.AGGLOMERATIVE:
                    clusters = self.cluster_by_agglomerative(fingerprints, **params)
                else:
                    continue
                
                # Find outliers (documents not in any cluster)
                clustered_ea_ids = set()
                for cluster in clusters:
                    clustered_ea_ids.update(cluster.ea_ids)
                
                outliers = [fp.ea_id for fp in fingerprints if fp.ea_id not in clustered_ea_ids]
                
                # Calculate evaluation score
                score = self._evaluate_clustering(clusters, len(fingerprints), len(outliers))
                
                self.logger.info(f"{algorithm.value}: {len(clusters)} clusters, "
                               f"{len(outliers)} outliers, score: {score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_result = ClusteringResult(
                        run_id=str(uuid.uuid4()),
                        algorithm=algorithm,
                        parameters=params,
                        clusters=clusters,
                        outliers=outliers,
                        num_documents=len(fingerprints),
                        num_clusters=len(clusters),
                        silhouette_score=None,  # Could be computed if needed
                        execution_time_seconds=(datetime.now() - start_time).total_seconds(),
                        timestamp=datetime.now()
                    )
                
            except Exception as e:
                self.logger.warning(f"Clustering attempt {algorithm.value} failed: {e}")
                continue
        
        if best_result is None:
            # Fallback: create singleton clusters
            self.logger.warning("All clustering attempts failed, creating singleton clusters")
            clusters = [
                ClusterCandidate(
                    cluster_id=str(uuid.uuid4()),
                    ea_ids=[fp.ea_id],
                    similarity_scores={},
                    centroid_ea_id=fp.ea_id,
                    confidence_score=1.0,
                    size=1
                )
                for fp in fingerprints
            ]
            
            best_result = ClusteringResult(
                run_id=str(uuid.uuid4()),
                algorithm=ClusteringAlgorithm.MINHASH_THRESHOLD,
                parameters={'threshold': 1.0},
                clusters=clusters,
                outliers=[],
                num_documents=len(fingerprints),
                num_clusters=len(clusters),
                silhouette_score=None,
                execution_time_seconds=(datetime.now() - start_time).total_seconds(),
                timestamp=datetime.now()
            )
        
        self.logger.info(f"Best clustering: {best_result.algorithm.value} with score {best_score:.3f}")
        return best_result
    
    def _evaluate_clustering(self,
                           clusters: List[ClusterCandidate],
                           total_documents: int,
                           num_outliers: int) -> float:
        """
        Evaluate clustering quality.
        
        Args:
            clusters: List of clusters
            total_documents: Total number of documents
            num_outliers: Number of outlier documents
            
        Returns:
            Clustering quality score (higher is better)
        """
        if not clusters:
            return 0.0
        
        # Factors for evaluation:
        # 1. Cluster size distribution (prefer balanced clusters)
        cluster_sizes = [cluster.size for cluster in clusters]
        size_variance = np.var(cluster_sizes) if len(cluster_sizes) > 1 else 0
        size_score = 1.0 / (1.0 + size_variance / np.mean(cluster_sizes))
        
        # 2. Average confidence score
        confidence_score = np.mean([cluster.confidence_score for cluster in clusters])
        
        # 3. Coverage (documents clustered vs total)
        coverage_score = (total_documents - num_outliers) / total_documents
        
        # 4. Penalize too many or too few clusters
        num_clusters = len(clusters)
        ideal_clusters = max(2, int(np.sqrt(total_documents)))
        cluster_penalty = abs(num_clusters - ideal_clusters) / ideal_clusters
        cluster_score = max(0.1, 1.0 - cluster_penalty)
        
        # Combine scores
        overall_score = (size_score * 0.2 + 
                        confidence_score * 0.4 + 
                        coverage_score * 0.3 + 
                        cluster_score * 0.1)
        
        return overall_score
    
    def save_clustering_result(self, result: ClusteringResult, output_dir: Path):
        """
        Save clustering result to files.
        
        Args:
            result: Clustering result to save
            output_dir: Output directory
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main result
        result_file = output_dir / "clusters.json"
        with open(result_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        # Save cluster summary CSV
        summary_data = []
        for cluster in result.clusters:
            summary_data.append({
                'cluster_id': cluster.cluster_id,
                'size': cluster.size,
                'centroid_ea_id': cluster.centroid_ea_id,
                'confidence_score': cluster.confidence_score,
                'ea_ids': ','.join(cluster.ea_ids)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = output_dir / "family_candidates.csv"
        summary_df.to_csv(summary_file, index=False)
        
        # Save outliers
        if result.outliers:
            outliers_file = output_dir / "outliers.txt"
            with open(outliers_file, 'w') as f:
                for ea_id in result.outliers:
                    f.write(f"{ea_id}\n")
        
        self.logger.info(f"Clustering result saved to {output_dir}")


def create_ea_clusterer() -> EAClusterer:
    """Factory function to create an EA clusterer."""
    return EAClusterer()