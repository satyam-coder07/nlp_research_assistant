import streamlit as st
import sys
from pathlib import Path
import logging
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from document_loader import DocumentLoader
from processor import TextPreprocessor
from feature_extractor import FeatureExtractor
from topic_modeler import TopicModeler
from clustering import DocumentClusterer
from keyword_extractor import KeywordExtractor
from summarizer import ExtractiveSummarizer
from evaluator import Evaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="NLP Research Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_session_state():
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'results' not in st.session_state:
        st.session_state.results = {}


def main():
    init_session_state()

    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        h1 {
            color: #1a365d;
            font-weight: 600;
            padding-bottom: 20px;
            border-bottom: 2px solid #e2e8f0;
        }
        div[data-testid="stMetric"] {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .streamlit-expanderHeader {
            background-color: #f1f5f9;
            border-radius: 4px;
            font-weight: 500;
        }
        .stButton>button {
            border-radius: 6px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        th {
            background-color: #1e293b !important;
            color: white !important;
        }
        section[data-testid="stSidebar"] {
            background-color: #f8fafc;
            border-right: 1px solid #e2e8f0;
        }
        h2, h3 {
            color: #2d3748;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Traditional NLP Research Analysis")
    st.markdown("**Milestone-1**: Classical NLP & ML Techniques for Research Document Analysis")

    with st.expander("Project Overview & Methodology", expanded=False):
        st.markdown("""
        ### Abstract
        This system explores the capabilities and limitations of **Traditional Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques applied to research documents.

        Unlike modern Large Language Models (LLMs), this milestone relies entirely on classical, foundational methods:
        - **Bag-of-Words (BoW) & TF-IDF** for feature extraction and text representation.
        - **Latent Dirichlet Allocation (LDA) / K-Means** for unsupervised topic modeling and text clustering.
        - **Heuristics & Statistics** for extractive summarization.

        **Objective**: To establish a baseline of what can be achieved without semantic embeddings and transformer-based reasoning, motivating the necessity of Agentic AI workflows in subsequent milestones.
        """)

    st.markdown("---")

    with st.sidebar:
        st.header("Configuration")
        st.subheader("Topic Modeling")
        num_topics = st.slider("Number of Topics", 2, 10, 5)
        num_keywords = st.slider("Keywords per Topic", 5, 20, 10)
        st.subheader("Summarization")
        num_sentences = st.slider("Summary Sentences", 3, 10, 5)
        summary_method = st.selectbox(
            "Summarization Method",
            ["TF-IDF", "Frequency-based", "Position-weighted"]
        )
        st.subheader("Analysis Method")
        modeling_method = st.selectbox(
            "Topic Extraction",
            ["LDA (Gensim)", "K-Means Clustering"]
        )
        st.markdown("---")
        st.info("This system uses only classical NLP techniques - no LLMs or transformers.")

    st.header("Input Documents")

    input_method = st.radio(
        "Choose input method:",
        ["Upload Documents (PDF/Text)", "Enter Text Directly"]
    )

    documents = {}

    if input_method == "Upload Documents (PDF/Text)":
        uploaded_files = st.file_uploader(
            "Upload research documents",
            type=['txt', 'pdf', 'md'],
            accept_multiple_files=True,
            help="Upload one or more text or PDF files"
        )
        if uploaded_files:
            st.success(f"{len(uploaded_files)} file(s) uploaded")
            loader = DocumentLoader()
            with st.spinner("Loading documents..."):
                documents = loader.load_from_uploaded_files(uploaded_files)
            if documents:
                st.info(f"Loaded {len(documents)} documents")
    else:
        text_input = st.text_area(
            "Paste your research text or keywords:",
            height=200,
            placeholder="Enter research documents, topics, or keywords..."
        )
        if text_input:
            documents['input_text'] = text_input
            st.success("Text received")

    st.markdown("---")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_button = st.button("Run Analysis", type="primary", use_container_width=True)

    if analyze_button and documents:
        with st.status("Processing Document Analysis...", expanded=True) as status:
            try:
                st.write("Initializing analysis pipeline...")
                results = run_analysis(
                    documents, num_topics, num_keywords,
                    num_sentences, summary_method, modeling_method
                )
                status.update(label="Analysis Complete!", state="complete", expanded=False)
                st.session_state.results = results
                st.session_state.analysis_complete = True
            except Exception as e:
                status.update(label="Analysis Failed!", state="error", expanded=True)
                st.error(f"Error details: {str(e)}")
                logger.error(f"Analysis error: {str(e)}", exc_info=True)
    elif analyze_button and not documents:
        st.warning("Please upload documents or enter text first!")

    if st.session_state.analysis_complete and st.session_state.results:
        display_results(st.session_state.results, modeling_method)


def run_analysis(documents, num_topics, num_keywords, num_sentences, summary_method, modeling_method):
    results = {}

    st.write("**Step 1/5**: Preprocessing documents...")
    preprocessor = TextPreprocessor(use_spacy=False)
    preprocessed_docs = {}
    for doc_id, text in documents.items():
        preprocessed_docs[doc_id] = preprocessor.preprocess_text(text)

    combined_text = ' '.join([text for text in documents.values()])
    all_tokens = [doc['lemmatized_tokens'] for doc in preprocessed_docs.values()]
    token_strings = [' '.join(tokens) for tokens in all_tokens]

    total_tokens = sum(len(tokens) for tokens in all_tokens)
    if total_tokens < 10:
        raise ValueError(
            f"Input text is too short! After preprocessing, only {total_tokens} tokens remain. "
            f"Please provide at least 1-2 paragraphs of content (100+ words)."
        )

    st.write("**Step 2/5**: Extracting features (TF-IDF)...")
    feature_extractor = FeatureExtractor()
    tfidf_matrix, feature_names = feature_extractor.extract_tfidf(
        token_strings, max_features=500, ngram_range=(1, 2)
    )
    results['tfidf_matrix'] = tfidf_matrix
    results['feature_names'] = feature_names
    results['feature_stats'] = feature_extractor.get_feature_statistics(tfidf_matrix)

    st.write(f"**Step 3/5**: {modeling_method}...")

    if modeling_method == "LDA (Gensim)":
        topic_modeler = TopicModeler()
        lda_model = topic_modeler.train_lda_model(all_tokens, num_topics=num_topics, passes=15)
        topics = topic_modeler.get_topics(num_words=num_keywords)
        doc_topics = topic_modeler.get_document_topics()
        dominant_topics = topic_modeler.assign_dominant_topic(doc_topics)
        results['method'] = 'LDA'
        results['topics'] = topics
        results['doc_topics'] = doc_topics
        results['dominant_topics'] = dominant_topics
        results['lda_model'] = lda_model
        results['topic_modeler'] = topic_modeler
        coherence_score = topic_modeler.calculate_coherence()
        results['coherence_score'] = coherence_score
    else:
        clusterer = DocumentClusterer()
        kmeans_model = clusterer.perform_kmeans(tfidf_matrix, n_clusters=num_topics)
        cluster_labels = clusterer.get_cluster_labels()
        cluster_terms = clusterer.get_top_terms_per_cluster(tfidf_matrix, feature_names, n_terms=num_keywords)
        cluster_stats = clusterer.get_cluster_statistics()
        results['method'] = 'K-Means'
        results['topics'] = cluster_terms
        results['cluster_labels'] = cluster_labels
        results['cluster_stats'] = cluster_stats
        results['kmeans_model'] = kmeans_model

    st.write("**Step 4/5**: Extracting keywords and themes...")
    keyword_extractor = KeywordExtractor()
    topic_keywords = keyword_extractor.extract_keywords_from_topics(results['topics'], n_keywords=num_keywords)
    theme_labels = keyword_extractor.generate_theme_labels(topic_keywords)
    overall_keywords = keyword_extractor.extract_keywords_tfidf(tfidf_matrix, feature_names, n=20)
    results['topic_keywords'] = topic_keywords
    results['theme_labels'] = theme_labels
    results['overall_keywords'] = overall_keywords

    st.write("**Step 5/5**: Generating extractive summary...")
    summarizer = ExtractiveSummarizer()
    if summary_method == "TF-IDF":
        summary = summarizer.summarize_tfidf(combined_text, num_sentences=num_sentences)
    elif summary_method == "Frequency-based":
        summary = summarizer.summarize_frequency(combined_text, num_sentences=num_sentences)
    else:
        summary = summarizer.summarize_position_weighted(combined_text, num_sentences=num_sentences)

    summary_stats = summarizer.get_summary_statistics(combined_text, summary)
    results['summary'] = summary
    results['summary_stats'] = summary_stats
    results['summary_method'] = summary_method

    if modeling_method == "LDA (Gensim)":
        evaluator = Evaluator()
        evaluation = evaluator.evaluate_topics(lda_model, all_tokens, topic_modeler.dictionary)
        results['evaluation'] = evaluation

    return results


def display_results(results, modeling_method):
    st.markdown("---")
    st.header("Analysis Results")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Topics", "Keywords", "Summary", "Evaluation", "Visualizations", "Limitations"
    ])

    with tab1:
        st.subheader(f"Discovered Topics ({modeling_method})")
        topics = results['topics']
        theme_labels = results['theme_labels']
        for topic_id in sorted(topics.keys()):
            with st.expander(f"**Topic {topic_id}**: {theme_labels.get(topic_id, 'N/A')}", expanded=True):
                topic_words = topics[topic_id]
                if isinstance(topic_words[0], tuple):
                    df = pd.DataFrame(topic_words, columns=['Word', 'Weight'])
                    df['Weight'] = df['Weight'].round(4)
                    styled_df = df.style.background_gradient(subset=['Weight'], cmap='Blues')
                else:
                    df = pd.DataFrame({'Word': topic_words})
                    styled_df = df
                st.dataframe(styled_df, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Overall Top Keywords")
        keywords = results['overall_keywords']
        col1, col2 = st.columns([2, 1])
        with col1:
            kw_df = pd.DataFrame(keywords, columns=['Keyword', 'TF-IDF Score'])
            kw_df['TF-IDF Score'] = kw_df['TF-IDF Score'].round(4)
            st.dataframe(kw_df.style.background_gradient(subset=['TF-IDF Score'], cmap='Purples'),
                        use_container_width=True, hide_index=True)
        with col2:
            st.markdown("**Word Cloud**")
            try:
                wordcloud_dict = {word: score for word, score in keywords}
                generate_wordcloud(wordcloud_dict)
            except Exception as e:
                st.error(f"Could not generate word cloud: {str(e)}")

    with tab3:
        st.subheader(f"Extractive Summary ({results['summary_method']})")
        st.info(results['summary'])
        st.markdown("---")
        st.subheader("Summary Statistics")
        stats = results['summary_stats']
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Original Sentences", stats['original_sentences'])
        with col2:
            st.metric("Summary Sentences", stats['summary_sentences'])
        with col3:
            st.metric("Original Words", stats['original_words'])
        with col4:
            st.metric("Summary Words", stats['summary_words'])
        st.write(f"**Compression Ratio**: {stats['compression_ratio']:.2%}")

    with tab4:
        st.subheader("Evaluation Metrics")
        if results['method'] == 'LDA':
            evaluation = results['evaluation']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Coherence Score (c_v)", f"{evaluation['coherence_score']:.4f}")
            with col2:
                st.metric("Topic Diversity", f"{evaluation['topic_diversity']:.4f}")
            with col3:
                st.metric("Number of Topics", evaluation['num_topics'])
            st.markdown("---")
            st.subheader("Interpretation")
            st.write(evaluation['interpretation'])
            st.markdown("---")
            st.subheader("Coherence per Topic")
            topic_coh_df = pd.DataFrame({
                'Topic ID': range(len(evaluation['coherence_per_topic'])),
                'Coherence': evaluation['coherence_per_topic']
            })
            st.dataframe(topic_coh_df, use_container_width=True, hide_index=True)
        else:
            st.info("K-Means clustering evaluation metrics")
            stats = results.get('cluster_stats', {})
            stats_df = pd.DataFrame([
                {'Cluster': k, 'Documents': v['num_documents'], 'Percentage': f"{v['percentage']:.1f}%"}
                for k, v in stats.items()
            ])
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        st.markdown("---")
        st.subheader("Feature Extraction Statistics")
        feat_stats = results['feature_stats']
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Documents", feat_stats['num_documents'])
        with col2:
            st.metric("Features", feat_stats['num_features'])
        with col3:
            st.metric("Sparsity", f"{feat_stats['sparsity']:.2%}")

    with tab5:
        st.subheader("Visualizations")
        st.markdown("**Top Keywords per Topic**")
        topic_id_to_plot = st.selectbox(
            "Select Topic:",
            options=list(results['topics'].keys()),
            format_func=lambda x: f"Topic {x}: {results['theme_labels'].get(x, 'N/A')}"
        )
        plot_topic_keywords(results['topics'][topic_id_to_plot], topic_id_to_plot)

    with tab6:
        st.subheader("Limitations of Classical NLP")
        st.markdown("""
        This system demonstrates the **capabilities and constraints** of traditional NLP techniques:

        ### Key Limitations

        #### 1. **No Semantic Understanding**
        - Bag-of-words and TF-IDF treat words as independent tokens
        - Cannot understand that "car" and "automobile" are synonyms
        - No comprehension of word order or sentence structure

        #### 2. **Context Blindness**
        - Cannot resolve polysemy (e.g., "bank" = financial institution vs. river bank)
        - Struggles with homonyms and contextual meanings
        - No understanding of discourse relationships

        #### 3. **Topic Coherence is not Meaningfulness**
        - High coherence scores measure statistical co-occurrence, not semantic coherence
        - Topics may group unrelated words that frequently appear together
        - Automatic theme labeling is unreliable and often nonsensical

        #### 4. **Preprocessing Sensitivity**
        - Results heavily depend on tokenization, stop-word lists, and lemmatization choices
        - Different preprocessing pipelines yield vastly different topics
        - No robustness to variations in text quality

        #### 5. **Summarization Limitations**
        - Extractive methods simply copy sentences - no paraphrasing or synthesis
        - Summaries may lack coherence and logical flow
        - Cannot combine information across sentences
        - Position bias (favors earlier sentences)

        #### 6. **No Multi-document Reasoning**
        - Cannot identify conflicting information across documents
        - No synthesis of information from multiple sources
        - Cannot track entities or relationships across documents

        ---

        ### Why This Motivates Agentic AI (Milestone-2)

        **Agentic AI systems** address these limitations by:
        - Using **transformer-based models** for semantic understanding
        - Employing **contextual embeddings** for word meaning disambiguation
        - Implementing **reasoning chains** for multi-step analysis
        - Leveraging **tool use** to access external knowledge
        - Performing **fact-checking** and contradiction detection
        - Generating **abstractive summaries** with true comprehension

        Traditional NLP provides interpretable, deterministic results but lacks the **semantic depth**
        and **reasoning capabilities** needed for advanced research analysis.
        """)


def generate_wordcloud(word_freq_dict):
    try:
        wordcloud = WordCloud(
            width=800, height=400, background_color='white', colormap='viridis'
        ).generate_from_frequencies(word_freq_dict)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.error(f"Word cloud generation failed: {str(e)}")


def plot_topic_keywords(topic_words, topic_id):
    try:
        if isinstance(topic_words[0], tuple):
            words = [w for w, _ in topic_words[:10]]
            weights = [wt for _, wt in topic_words[:10]]
        else:
            words = topic_words[:10]
            weights = [1] * len(words)
        chart_data = pd.DataFrame({"Weight": weights}, index=words)
        st.bar_chart(chart_data, horizontal=True, y_label="Keywords", x_label="Weight", color="#2b6cb0")
    except Exception as e:
        st.error(f"Chart generation failed: {str(e)}")


if __name__ == "__main__":
    main()