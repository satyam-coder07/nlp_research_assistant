import streamlit as st
import pandas as pd
from nltk.tokenize import sent_tokenize
from src.processor import extract_text_from_pdfs, preprocess_text
from src.models import extract_keywords_with_scores, perform_topic_modeling, generate_summary

st.set_page_config(page_title="NLP Research Analyzer", layout="wide")
st.title("Intelligent Research Analysis System")
st.caption("Milestone 1: Traditional NLP & Statistical Modeling")
st.markdown("---")

with st.sidebar:
    st.header("Document Ingestion")
    uploaded_files = st.file_uploader("Upload PDF Collection", type="pdf", accept_multiple_files=True)
    user_text = st.text_area("Or Paste Research Text:", height=200)

if st.button("Perform Analysis"):
    content = extract_text_from_pdfs(uploaded_files) if uploaded_files else user_text
    
    if content.strip():
        tokens = preprocess_text(content)
        
        st.subheader("1. Keywords & Visualisation")
        keyword_data = extract_keywords_with_scores(tokens)
        if keyword_data:
            df_km = pd.DataFrame(keyword_data, columns=["Keyword", "TF-IDF Score"])
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.table(df_km)
            with col_b:
                st.bar_chart(df_km.set_index("Keyword"))
        
        st.markdown("---")
        
        st.subheader("2. Topic Clusters")
        topics, coherence = perform_topic_modeling(tokens)
        t_cols = st.columns(3)
        for i, topic in enumerate(topics):
            with t_cols[i]:
                st.info(f"**Cluster {i+1}**\n\n" + ", ".join(topic))

        st.markdown("---")
        
        st.subheader("3. Research Summary")
        summary = generate_summary(content, tokens)
        st.success(summary)

        st.markdown("---")
        
        st.subheader("4. Technical Evaluation")
        ev1, ev2, ev3 = st.columns(3)
        ev1.metric("Sentences Count", len(sent_tokenize(content)))
        ev2.metric("Vocabulary Size", len(set(tokens)))
        if coherence:
            ev3.metric("Model Log-Perplexity", round(coherence, 2))
            st.caption("Log-Perplexity is a traditional measure of how well the topic model predicts the sample.")

    else:
        st.error("Please provide research documents to begin.")