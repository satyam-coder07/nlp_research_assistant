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
    page_title="TextAnalysis Pro | Research Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

def init_session_state():
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'results' not in st.session_state:
        st.session_state.results = {}
    if 'theme_mode' not in st.session_state:
        st.session_state.theme_mode = "Device Theme"

def generate_theme_css(theme_mode: str) -> str:
    """
    Return CSS string for the selected theme without using f-strings,
    to avoid any confusion between Python formatting and CSS braces.
    """

    # Base CSS shell, shared across themes
    base_css_prefix = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Essential Reset to avoid theme bleeding */
    [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: var(--bg-page) !important;
        color: var(--text-main) !important;
        font-family: 'Inter', sans-serif !important;
    }

    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

    /* ----------------------------
       Header Section
    ----------------------------- */
    .dashboard-header {
        padding: 1.5rem 0;
        margin-bottom: 2rem;
        border-bottom: 1px solid var(--border);
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
    }

    .header-title-group h1 {
        font-size: 1.875rem;
        font-weight: 700;
        color: var(--text-main);
        margin-bottom: 0.25rem;
    }

    .header-subtitle {
        font-size: 1rem;
        color: var(--text-muted);
        max-width: 600px;
    }

    .header-status {
        display: flex;
        gap: 0.75rem;
    }

    .badge {
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.025em;
    }

    .badge-primary {
        background-color: #dbeafe;
        color: #1e40af;
    }

    .badge-outline {
        border: 1px solid var(--border);
        color: var(--text-muted);
    }

    /* ----------------------------
       Containers & Cards
    ----------------------------- */
    .card {
        background-color: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.5rem;
        box-shadow: var(--shadow);
        margin-bottom: 1.5rem;
    }

    .card-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: var(--text-main);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ----------------------------
       Sidebar Styling
    ----------------------------- */
    [data-testid="stSidebar"] {
        background-color: var(--bg-card) !important;
        border-right: 1px solid var(--border) !important;
        padding: 1rem 0 !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        font-size: 0.875rem;
        color: var(--text-muted);
    }

    .sidebar-section-title {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-muted);
        margin-bottom: 1rem;
        margin-top: 1.5rem;
    }

    /* ----------------------------
       Inputs & Components
    ----------------------------- */
    .stButton > button {
        background-color: var(--primary) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius) !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        transition: all 0.2s ease !important;
        width: 100%;
    }

    .stButton > button:hover {
        background-color: var(--primary-hover) !important;
        box-shadow: var(--shadow-lg) !important;
    }

    /* Text areas and inputs */
    div[data-baseweb="textarea"], div[data-baseweb="input"] {
        border-radius: var(--radius) !important;
        border: 1px solid var(--border) !important;
        background-color: var(--bg-card) !important;
        color: var(--text-main) !important;
    }

    /* File Uploader override */
    [data-testid="stFileUploader"] section {
        background-color: #0206170d !important;
        border: 1px dashed var(--border) !important;
        border-radius: var(--radius) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid var(--border);
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre;
        background-color: transparent !important;
        border: none !important;
        color: var(--text-muted) !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        padding: 0 1.5rem !important;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: var(--primary) !important;
        border-bottom: 2px solid var(--primary) !important;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid var(--border);
        padding: 1rem;
        border-radius: var(--radius);
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.025em;
        color: var(--text-muted);
    }

    div[data-testid="stMetricValue"] {
        font-weight: 700;
        color: var(--text-main);
    }

    /* Success/Info override to match professional / adaptive theme */
    .stAlert {
        background-color: #0f172a0d !important;
        color: var(--text-main) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
    }

    /* Hide Streamlit components that look "generated" */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    """

    # Light theme variables
    light_vars = """
    :root {
        --primary: #2563eb;
        --primary-hover: #1d4ed8;
        --bg-page: #f8fafc;
        --bg-card: #ffffff;
        --text-main: #0f172a;
        --text-muted: #64748b;
        --border: #e2e8f0;
        --radius: 8px;
        --shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    """

    # Dark theme variables
    dark_vars = """
    :root {
        --primary: #3b82f6;
        --primary-hover: #60a5fa;
        --bg-page: #020617;
        --bg-card: #020617;
        --text-main: #e5e7eb;
        --text-muted: #9ca3af;
        --border: #1f2933;
        --radius: 8px;
        --shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.6), 0 1px 2px 0 rgba(0, 0, 0, 0.5);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.6), 0 4px 6px -2px rgba(0, 0, 0, 0.5);
    }
    """

    if theme_mode == "Light":
        root_block = light_vars
    elif theme_mode == "Dark":
        root_block = dark_vars
    else:
        # Device Theme: light by default, dark when system prefers it
        root_block = """
        :root {
            --primary: #2563eb;
            --primary-hover: #1d4ed8;
            --bg-page: #f8fafc;
            --bg-card: #ffffff;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --border: #e2e8f0;
            --radius: 8px;
            --shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --primary: #3b82f6;
                --primary-hover: #60a5fa;
                --bg-page: #020617;
                --bg-card: #020617;
                --text-main: #e5e7eb;
                --text-muted: #9ca3af;
                --border: #1f2933;
                --radius: 8px;
                --shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.6), 0 1px 2px 0 rgba(0, 0, 0, 0.5);
                --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.6), 0 4px 6px -2px rgba(0, 0, 0, 0.5);
            }
        }
        """

    return base_css_prefix + root_block + "\n</style>"

def main():
    init_session_state()

    # ----------------------------
    # Theme Selection (Sidebar)
    # ----------------------------
    with st.sidebar:
        st.markdown('<div class="sidebar-section-title">Appearance</div>', unsafe_allow_html=True)
        theme_mode = st.selectbox(
            "Theme",
            options=["Device Theme", "Light", "Dark"],
            index=["Device Theme", "Light", "Dark"].index(st.session_state.theme_mode),
            help="Device Theme follows your system / browser setting. Light and Dark override it."
        )
        st.session_state.theme_mode = theme_mode

    # Global styling based on selected theme
    st.markdown(generate_theme_css(st.session_state.theme_mode), unsafe_allow_html=True)

    # ----------------------------
    # Dashboard Header Component
    # ----------------------------
    st.markdown("""
        <div class="dashboard-header">
            <div class="header-title-group">
                <h1>Text Intelligence Platform</h1>
                <div class="header-subtitle">
                    Advanced research document analysis using classical natural language processing.
                    Establish an interpretable baseline before agentic transformation.
                </div>
            </div>
            <div class="header-status">
                <span class="badge badge-primary">Milestone 1</span>
                <span class="badge badge-outline">Interpretation Baseline</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar Configuration
    with st.sidebar:
        st.markdown('<div class="sidebar-section-title">System Configuration</div>', unsafe_allow_html=True)
        
        st.markdown("**Topic Modeling**")
        num_topics = st.slider("Target Topics", 2, 10, 5)
        num_keywords = st.slider("Terms per Topic", 5, 20, 10)
        
        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
        st.markdown("**Summarization**")
        num_sentences = st.slider("Length (Sentences)", 3, 10, 5)
        summary_method = st.selectbox(
            "Algorithm",
            ["TF-IDF", "Frequency-based", "Position-weighted"]
        )
        
        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
        st.markdown("**Core Method**")
        modeling_method = st.selectbox(
            "Extraction Engine",
            ["LDA (Gensim)", "K-Means Clustering"]
        )
        
        st.markdown('<div style="margin-top: 2rem;"></div>', unsafe_allow_html=True)
        st.caption("Architecture Focus: This phase implements classical NLP heuristics to ensure complete transparency of results.")

    # ----------------------------
    # Introduction / Methodology
    # ----------------------------
    with st.expander("Methodology & Philosophical Abstract", expanded=False):
        st.markdown("""
        ### Strategic Baseline
        This platform leverages **Foundational Natural Language Processing** to categorize and summarize research corpus. 
        Unlike opaque neural architectures, the methods used here—TF-IDF, Latent Dirichlet Allocation, and K-Means—provide 
        mathematical clarity on feature importance and document relationships.

        **Pipeline Overview:**
        - **Representation**: Statistical tokenization without semantic embedding.
        - **Modeling**: Unsupervised probabilistic distribution for topic discovery.
        - **Summarization**: Rank-based extractive heuristics for content synthesis.
        """)

    # ----------------------------
    # Document Input Section
    # ----------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Document Ingestion</div>', unsafe_allow_html=True)
    
    col_input, col_action = st.columns([2.5, 1])

    documents = {}
    with col_input:
        input_method = st.radio(
            "Select Ingestion Method:",
            ["Bulk Upload (PDF/TXT)", "Direct Script Input"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        if input_method == "Bulk Upload (PDF/TXT)":
            uploaded_files = st.file_uploader(
                "Upload corpus",
                type=['txt', 'pdf', 'md'],
                accept_multiple_files=True,
                help="Supported: PDF, Text, Markdown"
            )
            if uploaded_files:
                loader = DocumentLoader()
                with st.spinner("Extracting text from source..."):
                    documents = loader.load_from_uploaded_files(uploaded_files)
                if documents:
                    st.toast(f"Synchronized {len(documents)} document(s)")
        else:
            text_input = st.text_area(
                "Direct Input",
                height=180,
                placeholder="Paste research text or abstract content here for rapid analysis...",
                label_visibility="hidden"
            )
            if text_input:
                documents['input_text'] = text_input

    with col_action:
        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
        st.markdown("""
            <div style="font-size: 0.875rem; color: #64748b; line-height: 1.5; margin-bottom: 1rem;">
                Upon execution, the system will initiate the preprocessing pipeline, including normalization, 
                lemmatization, and feature vectorization.
            </div>
        """, unsafe_allow_html=True)
        analyze_button = st.button("Initiate Analysis Pipeline", type="primary")

    st.markdown('</div>', unsafe_allow_html=True)

    # ----------------------------
    # Analysis Execution
    # ----------------------------
    if analyze_button:
        if documents:
            with st.status("Executing Analysis Pipeline...", expanded=True) as status:
                try:
                    st.write("Initializing feature extraction...")
                    results = run_analysis(
                        documents, num_topics, num_keywords,
                        num_sentences, summary_method, modeling_method
                    )
                    status.update(label="Pipeline Execution Complete", state="complete", expanded=False)
                    st.session_state.results = results
                    st.session_state.analysis_complete = True
                except Exception as e:
                    status.update(label="Pipeline Failed", state="error", expanded=True)
                    st.error(f"Execution Error: {str(e)}")
                    logger.error(f"Analysis error: {str(e)}", exc_info=True)
        else:
            st.warning("Please provide input documents to begin.")

    # ----------------------------
    # Results Presentation
    # ----------------------------
    if st.session_state.analysis_complete and st.session_state.results:
        display_results(st.session_state.results, modeling_method)

def run_analysis(documents, num_topics, num_keywords, num_sentences, summary_method, modeling_method):
    results = {}

    st.write("Step 1/5: Normalizing and tokenizing documents...")
    preprocessor = TextPreprocessor(use_spacy=False)
    preprocessed_docs = {}
    for doc_id, text in documents.items():
        preprocessed_docs[doc_id] = preprocessor.preprocess_text(text)

    combined_text = ' '.join([text for text in documents.values()])
    all_tokens = [doc['lemmatized_tokens'] for doc in preprocessed_docs.values()]
    token_strings = [' '.join(tokens) for tokens in all_tokens]

    total_tokens = sum(len(tokens) for tokens in all_tokens)
    if total_tokens < 10:
        raise ValueError("Insufficient document volume for statistical modeling.")

    st.write("Step 2/5: Vectorizing features (TF-IDF)...")
    feature_extractor = FeatureExtractor()
    tfidf_matrix, feature_names = feature_extractor.extract_tfidf(
        token_strings, max_features=500, ngram_range=(1, 2)
    )
    results['tfidf_matrix'] = tfidf_matrix
    results['feature_names'] = feature_names
    results['feature_stats'] = feature_extractor.get_feature_statistics(tfidf_matrix)

    st.write(f"Step 3/5: Implementing {modeling_method}...")

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
        results['coherence_score'] = topic_modeler.calculate_coherence()
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

    st.write("Step 4/5: Discerning thematic keywords...")
    keyword_extractor = KeywordExtractor()
    topic_keywords = keyword_extractor.extract_keywords_from_topics(results['topics'], n_keywords=num_keywords)
    theme_labels = keyword_extractor.generate_theme_labels(topic_keywords)
    overall_keywords = keyword_extractor.extract_keywords_tfidf(tfidf_matrix, feature_names, n=20)
    results['topic_keywords'] = topic_keywords
    results['theme_labels'] = theme_labels
    results['overall_keywords'] = overall_keywords

    st.write("Step 5/5: Generating extractive synthesis...")
    summarizer = ExtractiveSummarizer()
    if summary_method == "TF-IDF":
        summary = summarizer.summarize_tfidf(combined_text, num_sentences=num_sentences)
    elif summary_method == "Frequency-based":
        summary = summarizer.summarize_frequency(combined_text, num_sentences=num_sentences)
    else:
        summary = summarizer.summarize_position_weighted(combined_text, num_sentences=num_sentences)

    results['summary'] = summary
    results['summary_stats'] = summarizer.get_summary_statistics(combined_text, summary)
    results['summary_method'] = summary_method

    if modeling_method == "LDA (Gensim)":
        evaluator = Evaluator()
        results['evaluation'] = evaluator.evaluate_topics(lda_model, all_tokens, topic_modeler.dictionary)

    return results

def display_results(results, modeling_method):
    st.markdown('<div class="card" style="margin-top: 2rem;">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Intelligence Report</div>', unsafe_allow_html=True)
    
    tabs = st.tabs([
        "Topics discovered", 
        "Thematic Keywords", 
        "Executive Summary", 
        "Model Integrity", 
        "Visualization", 
        "Foundational Limitations"
    ])

    with tabs[0]:
        st.markdown(f"**Cluster Analysis ({modeling_method})**")
        topics = results['topics']
        theme_labels = results['theme_labels']
        
        for topic_id in sorted(topics.keys()):
            with st.expander(f"Topic Group {topic_id}: {theme_labels.get(topic_id, 'Undefined Pattern')}", expanded=(topic_id==0)):
                topic_words = topics[topic_id]
                if isinstance(topic_words[0], tuple):
                    df = pd.DataFrame(topic_words, columns=['Term', 'Relevance'])
                    df['Relevance'] = df['Relevance'].round(4)
                else:
                    df = pd.DataFrame({'Term': topic_words})
                st.dataframe(df, use_container_width=True, hide_index=True)

    with tabs[1]:
        st.markdown("**Top Global Keywords**")
        keywords = results['overall_keywords']
        col_list, col_cloud = st.columns([1, 1])
        with col_list:
            kw_df = pd.DataFrame(keywords, columns=['Key Term', 'TF-IDF Weight'])
            kw_df['TF-IDF Weight'] = kw_df['TF-IDF Weight'].round(4)
            st.dataframe(kw_df, use_container_width=True, hide_index=True)
        with col_cloud:
            st.markdown('<p style="text-align: center; color:#64748b; font-size: 0.875rem;">Frequency Distribution</p>', unsafe_allow_html=True)
            try:
                generate_wordcloud({word: score for word, score in keywords})
            except:
                st.info("Visual rendering unavailable for this corpus.")

    with tabs[2]:
        st.markdown(f"**Synthesized Content (Method: {results['summary_method']})**")
        st.markdown(f'<div style="background-color: #f8fafc; padding: 1.5rem; border-radius: 8px; border: 1px solid #e2e8f0; line-height: 1.6; color: #1e293b;">{results["summary"]}</div>', unsafe_allow_html=True)
        
        st.markdown('<div style="height: 1.5rem;"></div>', unsafe_allow_html=True)
        stats = results['summary_stats']
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Original Sentences", stats['original_sentences'])
        m2.metric("Summary Sentences", stats['summary_sentences'])
        m3.metric("Reduction Ratio", f"{100-stats['compression_ratio']*100:.1f}%")
        m4.metric("Extracted Word Count", stats['summary_words'])

    with tabs[3]:
        st.markdown("**Core Metrics**")
        if results['method'] == 'LDA':
            evaluation = results['evaluation']
            c1, c2, c3 = st.columns(3)
            c1.metric("Coherence Score (c_v)", f"{evaluation['coherence_score']:.4f}")
            c2.metric("Topic Diversity", f"{evaluation['topic_diversity']:.4f}")
            c3.metric("Discovered Clusters", evaluation['num_topics'])
            
            st.markdown("---")
            st.markdown("**Interpretative Analysis**")
            st.write(evaluation['interpretation'])
        else:
            stats = results.get('cluster_stats', {})
            stats_df = pd.DataFrame([
                {'Cluster': k, 'Docs': v['num_documents'], 'Volume': f"{v['percentage']:.1f}%"}
                for k, v in stats.items()
            ])
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        feat_stats = results['feature_stats']
        f1, f2, f3 = st.columns(3)
        f1.metric("Processed Documents", feat_stats['num_documents'])
        f2.metric("Unique Features", feat_stats['num_features'])
        f3.metric("Matrix Sparsity", f"{feat_stats['sparsity']:.2%}")

    with tabs[4]:
        st.markdown("**Diagnostic Visuals**")
        topic_id_to_plot = st.selectbox(
            "Filter Visualization by Cluster:",
            options=list(results['topics'].keys()),
            format_func=lambda x: f"Group {x}: {results['theme_labels'].get(x, 'N/A')}"
        )
        plot_topic_keywords(results['topics'][topic_id_to_plot], topic_id_to_plot)

    with tabs[5]:
        st.markdown("""
        ### Strategic Constraints
        This foundational analysis highlights the boundaries of non-neural text processing:

        1. **Semantic Gaps**: Without word embeddings, the system cannot recognize that "patient" and "subject" might signify the same entity.
        2. **Context Loss**: Polysemy remains unresolved (e.g., "model" as a statistical artifact vs. a physical prototype).
        3. **Logic Limitations**: Extractive methods identify high-value sentences but cannot reason across documents to resolve contradictions.
        
        ---
        **Transformation Path**: This baseline establishes the quantitative metrics required to evaluate the performance gains of subsequent Large Language Model (LLM) implementations.
        """)

    st.markdown('</div>', unsafe_allow_html=True)

def generate_wordcloud(word_freq_dict):
    try:
        wordcloud = WordCloud(
            width=800, height=400, background_color='white', colormap='Blues',
            font_path=None, prefer_horizontal=0.9
        ).generate_from_frequencies(word_freq_dict)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.error("Visualization layer error.")

def plot_topic_keywords(topic_words, topic_id):
    try:
        if isinstance(topic_words[0], tuple):
            words = [w for w, _ in topic_words[:10]]
            weights = [wt for _, wt in topic_words[:10]]
        else:
            words = topic_words[:10]
            weights = [1] * len(words)
        chart_data = pd.DataFrame({"Weight": weights}, index=words)
        st.bar_chart(chart_data, horizontal=True, color="#2563eb")
    except:
        st.error("Chart rendering error.")

if __name__ == "__main__":
    main()
