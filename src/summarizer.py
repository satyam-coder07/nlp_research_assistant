import logging
import re
from typing import List, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter


class ExtractiveSummarizer:

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def summarize_tfidf(self, text, num_sentences=5, min_sentence_length=10):
        sentences = self._split_sentences(text)
        if len(sentences) <= num_sentences:
            return text
        valid_sentences = [s for s in sentences if len(s.split()) >= min_sentence_length]
        if len(valid_sentences) <= num_sentences:
            return ' '.join(valid_sentences)
        vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        try:
            tfidf_matrix = vectorizer.fit_transform(valid_sentences)
            sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)
            top_indices = np.argsort(sentence_scores)[::-1][:num_sentences]
            top_indices = sorted(top_indices)
            return ' '.join([valid_sentences[idx] for idx in top_indices])
        except Exception:
            return ' '.join(valid_sentences[:num_sentences])

    def summarize_frequency(self, text, num_sentences=5, min_sentence_length=10):
        sentences = self._split_sentences(text)
        if len(sentences) <= num_sentences:
            return text
        valid_sentences = [s for s in sentences if len(s.split()) >= min_sentence_length]
        if len(valid_sentences) <= num_sentences:
            return ' '.join(valid_sentences)
        words = self._tokenize_words(text)
        word_freq = Counter(words)
        max_freq = max(word_freq.values()) if word_freq else 1
        for word in word_freq:
            word_freq[word] = word_freq[word] / max_freq
        sentence_scores = {}
        for idx, sentence in enumerate(valid_sentences):
            sentence_scores[idx] = self._score_sentence(sentence, word_freq)
        top_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
        top_indices = sorted(top_indices)
        return ' '.join([valid_sentences[idx] for idx in top_indices])

    def summarize_position_weighted(self, text, num_sentences=5, min_sentence_length=10,
                                    position_weight=0.3):
        sentences = self._split_sentences(text)
        if len(sentences) <= num_sentences:
            return text
        valid_sentences = [s for s in sentences if len(s.split()) >= min_sentence_length]
        if len(valid_sentences) <= num_sentences:
            return ' '.join(valid_sentences)
        vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        try:
            tfidf_matrix = vectorizer.fit_transform(valid_sentences)
            content_scores = np.sum(tfidf_matrix.toarray(), axis=1)
            content_scores = content_scores / np.max(content_scores)
            position_scores = np.array([
                1.0 - (i / len(valid_sentences)) for i in range(len(valid_sentences))
            ])
            final_scores = (1 - position_weight) * content_scores + position_weight * position_scores
            top_indices = np.argsort(final_scores)[::-1][:num_sentences]
            top_indices = sorted(top_indices)
            return ' '.join([valid_sentences[idx] for idx in top_indices])
        except Exception:
            return ' '.join(valid_sentences[:num_sentences])

    def _split_sentences(self, text):
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _tokenize_words(self, text):
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
            'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
            'she', 'it', 'we', 'they', 'them', 'their', 'our'
        }
        return [w for w in words if w not in stopwords and len(w) > 2]

    def _score_sentence(self, sentence, word_freq):
        words = self._tokenize_words(sentence)
        if not words:
            return 0.0
        return sum(word_freq.get(word, 0) for word in words) / len(words)

    def get_summary_statistics(self, original_text, summary):
        original_sentences = self._split_sentences(original_text)
        summary_sentences = self._split_sentences(summary)
        original_words = len(original_text.split())
        summary_words = len(summary.split())
        return {
            'original_sentences': len(original_sentences),
            'summary_sentences': len(summary_sentences),
            'original_words': original_words,
            'summary_words': summary_words,
            'compression_ratio': summary_words / original_words if original_words > 0 else 0,
            'sentence_retention': len(summary_sentences) / len(original_sentences) if original_sentences else 0
        }
