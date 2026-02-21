import logging
from typing import Dict, List, Any
import numpy as np


class Evaluator:

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def evaluate_topics(self, lda_model, texts, dictionary, coherence_type='c_v'):
        from gensim.models.coherencemodel import CoherenceModel
        coherence_model = CoherenceModel(
            model=lda_model, texts=texts,
            dictionary=dictionary, coherence=coherence_type
        )
        coherence_score = coherence_model.get_coherence()
        coherence_per_topic = coherence_model.get_coherence_per_topic()
        diversity = self._calculate_topic_diversity(lda_model)
        return {
            'coherence_score': coherence_score,
            'coherence_type': coherence_type,
            'coherence_per_topic': coherence_per_topic,
            'topic_diversity': diversity,
            'num_topics': lda_model.num_topics,
            'interpretation': self._interpret_coherence(coherence_score, coherence_type)
        }

    def _calculate_topic_diversity(self, lda_model, top_n=10):
        unique_words = set()
        total_words = 0
        for topic_id in range(lda_model.num_topics):
            top_words = [word for word, _ in lda_model.show_topic(topic_id, topn=top_n)]
            unique_words.update(top_words)
            total_words += len(top_words)
        if total_words == 0:
            return 0.0
        return len(unique_words) / total_words

    def _interpret_coherence(self, score, coherence_type):
        if coherence_type == 'c_v':
            if score > 0.6:
                return f"Excellent ({score:.4f}): Topics are highly coherent and semantically meaningful."
            elif score > 0.5:
                return f"Good ({score:.4f}): Topics show good coherence with clear themes."
            elif score > 0.4:
                return f"Moderate ({score:.4f}): Topics are somewhat coherent but may have mixed themes."
            elif score > 0.3:
                return f"Fair ({score:.4f}): Topics show weak coherence; consider tuning parameters."
            else:
                return f"Poor ({score:.4f}): Topics lack coherence; significant tuning needed."
        elif coherence_type == 'u_mass':
            if score > -1:
                return f"Excellent ({score:.4f}): Topics are highly coherent."
            elif score > -2:
                return f"Good ({score:.4f}): Topics show good coherence."
            elif score > -3:
                return f"Moderate ({score:.4f}): Topics are moderately coherent."
            else:
                return f"Fair to Poor ({score:.4f}): Topics show weak coherence."
        return f"Score: {score:.4f}"

    def evaluate_clustering(self, labels, feature_matrix):
        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
        metrics = {}
        try:
            metrics['silhouette_score'] = silhouette_score(feature_matrix, labels)
            metrics['davies_bouldin_score'] = davies_bouldin_score(feature_matrix, labels)
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(feature_matrix, labels)
        except Exception as e:
            self.logger.error(f"Clustering evaluation failed: {str(e)}")
        return metrics

    def compare_summaries(self, summaries):
        comparison = {}
        for method, summary in summaries.items():
            comparison[method] = {
                'length': len(summary),
                'word_count': len(summary.split()),
                'sentence_count': len([s for s in summary.split('.') if s.strip()])
            }
        return comparison
