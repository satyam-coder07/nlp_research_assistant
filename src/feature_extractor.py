import logging
from typing import List, Dict, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class FeatureExtractor:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.tfidf_vectorizer = None
        self.bow_vectorizer = None
        self.feature_names = None

    def extract_tfidf(self, documents, max_features=500, ngram_range=(1, 2),
                     min_df=2, max_df=0.85):
        num_docs = len(documents)
        if num_docs == 1:
            adaptive_min_df = 1
            adaptive_max_df = 1.0
        elif num_docs <= 3:
            adaptive_min_df = 1
            adaptive_max_df = 1.0
        else:
            adaptive_min_df = min(min_df, num_docs // 2)
            adaptive_max_df = max_df

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=adaptive_min_df,
            max_df=adaptive_max_df,
            sublinear_tf=True,
            use_idf=True,
            smooth_idf=True
        )

        tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        self.feature_names = list(feature_names)
        return tfidf_matrix.toarray(), self.feature_names

    def extract_bow(self, documents, max_features=500, ngram_range=(1, 1),
                   min_df=2, max_df=0.85):
        num_docs = len(documents)
        if num_docs == 1:
            adaptive_min_df = 1
            adaptive_max_df = 1.0
        elif num_docs <= 3:
            adaptive_min_df = 1
            adaptive_max_df = 1.0
        else:
            adaptive_min_df = min(min_df, num_docs // 2)
            adaptive_max_df = max_df

        self.bow_vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=adaptive_min_df,
            max_df=adaptive_max_df
        )

        bow_matrix = self.bow_vectorizer.fit_transform(documents)
        feature_names = self.bow_vectorizer.get_feature_names_out()
        return bow_matrix.toarray(), list(feature_names)

    def get_top_terms_per_document(self, tfidf_matrix, feature_names, n=10):
        top_terms = {}
        for doc_idx in range(tfidf_matrix.shape[0]):
            scores = tfidf_matrix[doc_idx]
            top_indices = np.argsort(scores)[::-1][:n]
            terms_scores = [
                (feature_names[idx], scores[idx])
                for idx in top_indices if scores[idx] > 0
            ]
            top_terms[doc_idx] = terms_scores
        return top_terms

    def get_overall_top_terms(self, tfidf_matrix, feature_names, n=20):
        mean_scores = np.mean(tfidf_matrix, axis=0)
        top_indices = np.argsort(mean_scores)[::-1][:n]
        return [(feature_names[idx], mean_scores[idx]) for idx in top_indices]

    def transform_new_documents(self, documents, method='tfidf'):
        if method == 'tfidf':
            if self.tfidf_vectorizer is None:
                raise ValueError("TF-IDF vectorizer not fitted.")
            return self.tfidf_vectorizer.transform(documents).toarray()
        elif method == 'bow':
            if self.bow_vectorizer is None:
                raise ValueError("BoW vectorizer not fitted.")
            return self.bow_vectorizer.transform(documents).toarray()
        else:
            raise ValueError(f"Unknown method: {method}")

    def get_feature_statistics(self, tfidf_matrix):
        return {
            'num_documents': tfidf_matrix.shape[0],
            'num_features': tfidf_matrix.shape[1],
            'sparsity': 1.0 - (np.count_nonzero(tfidf_matrix) / tfidf_matrix.size),
            'mean_tfidf': np.mean(tfidf_matrix),
            'max_tfidf': np.max(tfidf_matrix)
        }
