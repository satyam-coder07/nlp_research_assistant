import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel


class TopicModeler:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.lda_model = None
        self.dictionary = None
        self.corpus = None
        self.texts = None

    def prepare_corpus(self, documents):
        if not documents:
            raise ValueError("No documents provided for topic modeling.")
        total_tokens = sum(len(doc) for doc in documents)
        if total_tokens == 0:
            raise ValueError("All documents are empty after preprocessing.")
        self.dictionary = corpora.Dictionary(documents)
        num_docs = len(documents)
        if num_docs == 1:
            pass
        elif num_docs <= 3:
            self.dictionary.filter_extremes(no_below=1, no_above=1.0, keep_n=500)
        else:
            self.dictionary.filter_extremes(
                no_below=min(2, num_docs // 2), no_above=0.8, keep_n=1000
            )
        if len(self.dictionary) == 0:
            raise ValueError(
                f"No terms remaining after dictionary filtering. "
                f"Documents: {num_docs}, Total tokens before filtering: {total_tokens}."
            )
        self.corpus = [self.dictionary.doc2bow(doc) for doc in documents]
        non_empty_docs = sum(1 for doc in self.corpus if len(doc) > 0)
        if non_empty_docs == 0:
            raise ValueError("All documents became empty after dictionary processing.")
        self.texts = documents
        return self.dictionary, self.corpus

    def train_lda_model(self, documents, num_topics=5, passes=15, iterations=100,
                       alpha='auto', eta='auto', random_state=42):
        self.prepare_corpus(documents)
        self.lda_model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=num_topics,
            passes=passes,
            iterations=iterations,
            alpha=alpha,
            eta=eta,
            random_state=random_state,
            per_word_topics=True
        )
        return self.lda_model

    def get_topics(self, num_words=10):
        if self.lda_model is None:
            raise ValueError("Model not trained.")
        topics = {}
        for topic_id in range(self.lda_model.num_topics):
            topics[topic_id] = self.lda_model.show_topic(topic_id, topn=num_words)
        return topics

    def get_document_topics(self, document_corpus=None):
        if self.lda_model is None:
            raise ValueError("Model not trained.")
        corpus_to_use = document_corpus if document_corpus is not None else self.corpus
        doc_topics = []
        for doc in corpus_to_use:
            topics = self.lda_model.get_document_topics(doc)
            doc_topics.append(topics)
        return doc_topics

    def assign_dominant_topic(self, doc_topics):
        dominant_topics = []
        for doc_idx, topics in enumerate(doc_topics):
            if not topics:
                dominant_topics.append({
                    'doc_idx': doc_idx, 'dominant_topic': -1,
                    'topic_weight': 0.0, 'all_topics': []
                })
                continue
            sorted_topics = sorted(topics, key=lambda x: x[1], reverse=True)
            dominant = sorted_topics[0]
            dominant_topics.append({
                'doc_idx': doc_idx, 'dominant_topic': dominant[0],
                'topic_weight': dominant[1], 'all_topics': sorted_topics
            })
        return dominant_topics

    def calculate_coherence(self, coherence_type='c_v'):
        if self.lda_model is None or self.texts is None:
            raise ValueError("Model not trained or texts not available.")
        coherence_model = CoherenceModel(
            model=self.lda_model, texts=self.texts,
            dictionary=self.dictionary, coherence=coherence_type
        )
        return coherence_model.get_coherence()

    def get_topic_keywords(self, num_keywords=10):
        topics = self.get_topics(num_words=num_keywords)
        return {topic_id: [word for word, _ in words_weights]
                for topic_id, words_weights in topics.items()}
