from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
import gensim.corpora as corpora
from nltk.tokenize import word_tokenize, sent_tokenize
import heapq

def extract_keywords_with_scores(clean_tokens):
    if len(clean_tokens) < 2:
        return []
    text_str = " ".join(clean_tokens)
    vectorizer = TfidfVectorizer(max_features=10)
    try:
        tfidf_matrix = vectorizer.fit_transform([text_str])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray().flatten()
        return sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
    except:
        return []

def perform_topic_modeling(clean_tokens):
    if len(clean_tokens) < 5:
        return [], None
    id2word = corpora.Dictionary([clean_tokens])
    corpus = [id2word.doc2bow(clean_tokens)]
    try:
        lda_model = gensim.models.LdaModel(corpus=corpus, id2word=id2word, num_topics=3, passes=10)
        coherence = lda_model.log_perplexity(corpus)
        raw_topics = lda_model.show_topics(formatted=False)
        formatted_topics = []
        for _, topic in raw_topics:
            formatted_topics.append([word for word, prob in topic])
        return formatted_topics, coherence
    except:
        return [], None

def generate_summary(text, clean_tokens):
    sentences = sent_tokenize(text)
    if len(sentences) <= 3: return text
    word_frequencies = {}
    for word in clean_tokens:
        word_frequencies[word] = word_frequencies.get(word, 0) + 1
    max_freq = max(word_frequencies.values())
    for word in word_frequencies:
        word_frequencies[word] = word_frequencies[word] / max_freq
    sentence_scores = {}
    for sent in sentences:
        for word in word_tokenize(sent.lower()):
            if word in word_frequencies:
                sentence_scores[sent] = sentence_scores.get(sent, 0) + word_frequencies[word]
    summary_sentences = heapq.nlargest(3, sentence_scores, key=sentence_scores.get)
    return " ".join(summary_sentences)