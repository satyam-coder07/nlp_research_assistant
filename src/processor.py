import re
import logging
from typing import List, Dict, Any

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


class TextPreprocessor:

    def __init__(self, use_spacy=True, language='english'):
        self.logger = logging.getLogger(__name__)
        self.language = language
        self._download_nltk_resources()
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words(language))
        except Exception:
            self.stop_words = set()
        self.stop_words.update([
            'et', 'al', 'fig', 'figure', 'table', 'section', 'chapter',
            'paper', 'study', 'research', 'pp', 'vol', 'doi', 'isbn'
        ])
        self.nlp = None
        if use_spacy and SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except Exception:
                self.nlp = None

    def _download_nltk_resources(self):
        resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                except Exception:
                    pass

    def preprocess_text(self, text, remove_stopwords=True, lemmatize=True):
        cleaned = self._clean_text(text)
        sentences = self.tokenize_sentences(cleaned)
        tokens = self.tokenize_words(cleaned)
        tokens_no_stopwords = tokens
        if remove_stopwords:
            tokens_no_stopwords = self.remove_stopwords(tokens)
        lemmatized = tokens_no_stopwords
        if lemmatize:
            if self.nlp:
                lemmatized = self.lemmatize_spacy(tokens_no_stopwords)
            else:
                lemmatized = self.lemmatize_nltk(tokens_no_stopwords)
        return {
            'original_text': text,
            'cleaned_text': cleaned,
            'sentences': sentences,
            'tokens': tokens,
            'tokens_no_stopwords': tokens_no_stopwords,
            'lemmatized_tokens': lemmatized,
            'token_count': len(tokens),
            'sentence_count': len(sentences)
        }

    def _clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s\.\,\;\:\!\?]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def tokenize_sentences(self, text):
        try:
            sentences = sent_tokenize(text)
            return [s.strip() for s in sentences if s.strip()]
        except Exception:
            return [s.strip() + '.' for s in text.split('.') if s.strip()]

    def tokenize_words(self, text):
        try:
            tokens = word_tokenize(text)
            return [t for t in tokens if t.isalpha() and len(t) > 1]
        except Exception:
            return [w for w in text.split() if w.isalpha() and len(w) > 1]

    def remove_stopwords(self, tokens):
        return [token for token in tokens if token.lower() not in self.stop_words]

    def lemmatize_nltk(self, tokens):
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def lemmatize_spacy(self, tokens):
        if not self.nlp:
            return self.lemmatize_nltk(tokens)
        text = ' '.join(tokens)
        doc = self.nlp(text)
        return [token.lemma_ for token in doc if token.is_alpha]

    def preprocess_documents(self, documents):
        preprocessed = {}
        for doc_id, text in documents.items():
            preprocessed[doc_id] = self.preprocess_text(text)
        return preprocessed