import logging
from typing import List, Dict, Tuple
from collections import Counter
import numpy as np


class KeywordExtractor:

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def extract_keywords_from_topics(self, topics, n_keywords=10):
        keywords = {}
        for topic_id, word_weights in topics.items():
            keywords[topic_id] = [word for word, weight in word_weights[:n_keywords]]
        return keywords

    def extract_keywords_tfidf(self, tfidf_matrix, feature_names, n=20):
        mean_scores = np.mean(tfidf_matrix, axis=0)
        top_indices = np.argsort(mean_scores)[::-1][:n]
        return [(feature_names[idx], mean_scores[idx]) for idx in top_indices]

    def generate_theme_labels(self, topic_keywords, strategy='simple'):
        theme_labels = {}
        for topic_id, keywords in topic_keywords.items():
            if strategy == 'simple':
                label = " + ".join(keywords[:3])
            elif strategy == 'combined':
                label = self._infer_theme(keywords)
            else:
                label = f"Topic {topic_id}"
            theme_labels[topic_id] = label
        return theme_labels

    def _infer_theme(self, keywords):
        theme_patterns = {
            'research': ['research', 'study', 'analysis', 'method', 'data', 'experiment'],
            'learning': ['learn', 'learning', 'knowledge', 'education', 'training', 'skill'],
            'technology': ['technology', 'system', 'software', 'algorithm', 'computer', 'digital'],
            'health': ['health', 'medical', 'patient', 'treatment', 'disease', 'clinical'],
            'business': ['business', 'market', 'company', 'financial', 'economic', 'management'],
            'social': ['social', 'people', 'community', 'society', 'cultural', 'human']
        }
        theme_scores = {}
        keywords_lower = [k.lower() for k in keywords]
        for theme, pattern_words in theme_patterns.items():
            score = sum(1 for kw in keywords_lower if any(pw in kw for pw in pattern_words))
            if score > 0:
                theme_scores[theme] = score
        if theme_scores:
            best_theme = max(theme_scores, key=theme_scores.get)
            return f"{best_theme.title()}: {' + '.join(keywords[:3])}"
        return " + ".join(keywords[:3])

    def extract_document_keywords(self, tfidf_matrix, feature_names, doc_idx, n=10):
        doc_vector = tfidf_matrix[doc_idx]
        top_indices = np.argsort(doc_vector)[::-1][:n]
        return [
            (feature_names[idx], doc_vector[idx])
            for idx in top_indices if doc_vector[idx] > 0
        ]

    def get_keyword_frequencies(self, documents):
        all_tokens = [token for doc in documents for token in doc]
        return dict(Counter(all_tokens))

    def get_top_keywords_by_frequency(self, documents, n=20):
        freq_dist = self.get_keyword_frequencies(documents)
        return sorted(freq_dist.items(), key=lambda x: x[1], reverse=True)[:n]
