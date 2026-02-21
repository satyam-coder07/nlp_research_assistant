import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class DocumentClusterer:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.kmeans_model = None
        self.feature_names = None

    def perform_kmeans(self, feature_matrix, n_clusters=5, random_state=42, max_iter=300):
        self.kmeans_model = KMeans(
            n_clusters=n_clusters, random_state=random_state,
            max_iter=max_iter, n_init=10
        )
        self.kmeans_model.fit(feature_matrix)
        return self.kmeans_model

    def find_optimal_clusters(self, feature_matrix, min_k=2, max_k=10):
        inertias = {}
        silhouette_scores = {}
        for k in range(min_k, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(feature_matrix)
            inertias[k] = kmeans.inertia_
            if k > 1:
                silhouette_scores[k] = silhouette_score(feature_matrix, labels)
        optimal_k = max(silhouette_scores, key=silhouette_scores.get)
        return optimal_k, {'inertias': inertias, 'silhouette_scores': silhouette_scores}

    def get_cluster_labels(self):
        if self.kmeans_model is None:
            raise ValueError("Model not trained.")
        return self.kmeans_model.labels_

    def get_top_terms_per_cluster(self, tfidf_matrix, feature_names, n_terms=10):
        if self.kmeans_model is None:
            raise ValueError("Model not trained.")
        self.feature_names = feature_names
        cluster_terms = {}
        centers = self.kmeans_model.cluster_centers_
        for cluster_id in range(len(centers)):
            centroid = centers[cluster_id]
            top_indices = np.argsort(centroid)[::-1][:n_terms]
            cluster_terms[cluster_id] = [
                (feature_names[idx], centroid[idx]) for idx in top_indices
            ]
        return cluster_terms

    def get_documents_per_cluster(self, labels=None):
        if labels is None:
            labels = self.get_cluster_labels()
        clusters = {}
        for doc_idx, cluster_id in enumerate(labels):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(doc_idx)
        return clusters

    def get_cluster_statistics(self):
        labels = self.get_cluster_labels()
        doc_per_cluster = self.get_documents_per_cluster(labels)
        stats = {}
        for cluster_id, doc_indices in doc_per_cluster.items():
            stats[cluster_id] = {
                'num_documents': len(doc_indices),
                'percentage': (len(doc_indices) / len(labels)) * 100
            }
        return stats
