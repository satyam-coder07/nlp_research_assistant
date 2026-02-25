# NLP Research Assistant

A comprehensive toolkit for document analysis using traditional Natural Language Processing (NLP) and Machine Learning techniques. This project demonstrates classical approaches to text analysis, topic modeling, clustering, and summarization without relying on modern Large Language Models.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project serves as **Milestone-1** in exploring NLP capabilities, focusing exclusively on traditional, interpretable methods. It provides a baseline for understanding what can be achieved without semantic embeddings or transformer-based models, motivating the need for more advanced Agentic AI approaches in subsequent milestones.

The system processes research documents through a complete pipeline: loading, preprocessing, feature extraction, topic modeling, clustering, keyword extraction, and summarization.

## Features

### Core Capabilities

- **Document Loading**: Support for PDF, TXT, and Markdown files
- **Text Preprocessing**: Tokenization, lemmatization, and stop-word removal
- **Feature Extraction**: TF-IDF vectorization with n-gram support
- **Topic Modeling**: 
  - Latent Dirichlet Allocation (LDA) using Gensim
  - K-Means clustering for document grouping
- **Keyword Extraction**: Statistical keyword identification and theme labeling
- **Summarization**: Multiple extractive summarization methods
  - TF-IDF based
  - Frequency-based
  - Position-weighted
- **Evaluation Metrics**: Coherence scores, topic diversity, and compression ratios
- **Visualizations**: Word clouds, topic distributions, and keyword charts

### Interactive Web Interface

Built with Streamlit, the interface provides:
- Real-time document analysis
- Configurable parameters for all algorithms
- Multiple visualization options
- Comprehensive results display with tabs
- Professional, clean UI design

## Tech Stack

### Core Libraries

- **Python 3.8+**: Primary programming language
- **Streamlit**: Web application framework
- **NLTK**: Natural language processing toolkit
- **scikit-learn**: Machine learning algorithms (TF-IDF, K-Means)
- **Gensim**: Topic modeling (LDA)
- **spaCy**: Advanced text processing (optional)

### Supporting Libraries

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib**: Plotting and visualizations
- **WordCloud**: Word cloud generation
- **PyPDF2 / pdfplumber**: PDF document parsing
- **scipy**: Scientific computing utilities

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Setup Steps

1. **Clone the repository**

```bash
git clone <repository-url>
cd nlp_research_assistant
```

2. **Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download NLTK data**

```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

5. **Optional: Download spaCy model**

```bash
python -m spacy download en_core_web_sm
```

## Usage

### Running the Application

Start the Streamlit web interface:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Using the Interface

1. **Configure Parameters** (Sidebar):
   - Set number of topics (2-10)
   - Choose keywords per topic (5-20)
   - Select summary length (3-10 sentences)
   - Pick summarization method
   - Choose topic modeling algorithm (LDA or K-Means)

2. **Input Documents**:
   - Upload PDF/TXT/MD files, or
   - Paste text directly into the text area

3. **Run Analysis**:
   - Click "Run Analysis" button
   - Wait for processing to complete

4. **Explore Results**:
   - **Topics Tab**: View discovered topics with keywords
   - **Keywords Tab**: See overall top keywords and word cloud
   - **Summary Tab**: Read extractive summary with statistics
   - **Evaluation Tab**: Check coherence scores and metrics
   - **Visualizations Tab**: Explore charts and graphs
   - **Limitations Tab**: Understand classical NLP constraints

### Programmatic Usage

You can also use individual modules in your own scripts:

```python
from src.document_loader import DocumentLoader
from src.processor import TextPreprocessor
from src.feature_extractor import FeatureExtractor
from src.topic_modeler import TopicModeler

# Load documents
loader = DocumentLoader()
documents = loader.load_from_directory("data/")

# Preprocess
preprocessor = TextPreprocessor()
processed = preprocessor.preprocess_text(documents['doc1'])

# Extract features
extractor = FeatureExtractor()
tfidf_matrix, features = extractor.extract_tfidf([processed['lemmatized_tokens']])

# Topic modeling
modeler = TopicModeler()
model = modeler.train_lda_model([processed['lemmatized_tokens']], num_topics=5)
topics = modeler.get_topics(num_words=10)
```

## Project Structure

```
nlp_research_assistant/
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
└── src/                        # Core modules
    ├── document_loader.py      # Document loading and parsing
    ├── processor.py            # Text preprocessing pipeline
    ├── feature_extractor.py    # TF-IDF and feature extraction
    ├── topic_modeler.py        # LDA topic modeling
    ├── clustering.py           # K-Means clustering
    ├── keyword_extractor.py    # Keyword extraction and labeling
    ├── summarizer.py           # Extractive summarization
    ├── evaluator.py            # Evaluation metrics
    └── utils.py                # Shared utility functions
```

## Methodology

### Pipeline Overview

1. **Document Loading**: Parse and load documents from various formats
2. **Preprocessing**: 
   - Tokenization
   - Lowercasing
   - Stop-word removal
   - Lemmatization
3. **Feature Extraction**: Convert text to TF-IDF vectors
4. **Topic Modeling**: Apply LDA or K-Means to discover topics
5. **Keyword Extraction**: Identify significant terms and generate theme labels
6. **Summarization**: Extract most important sentences
7. **Evaluation**: Calculate coherence, diversity, and compression metrics

### Algorithms Used

- **TF-IDF**: Term Frequency-Inverse Document Frequency for feature weighting
- **LDA**: Latent Dirichlet Allocation for probabilistic topic modeling
- **K-Means**: Clustering algorithm for document grouping
- **Extractive Summarization**: Sentence ranking based on various heuristics

## Limitations

This project intentionally uses only classical NLP techniques to demonstrate their capabilities and constraints:

### Key Limitations

1. **No Semantic Understanding**: Treats words as independent tokens, cannot understand synonyms or context
2. **Context Blindness**: Cannot resolve polysemy or understand discourse relationships
3. **Statistical Co-occurrence**: High coherence doesn't guarantee meaningful topics
4. **Preprocessing Sensitivity**: Results heavily depend on preprocessing choices
5. **Extractive Summarization**: Can only copy sentences, no paraphrasing or synthesis
6. **No Multi-document Reasoning**: Cannot identify conflicts or synthesize across documents

These limitations motivate the development of more advanced Agentic AI systems in future milestones.

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Make your changes
4. Write clear commit messages
5. Push to your branch (`git push origin feature-name`)
6. Open a Pull Request with a detailed description

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all functions and classes
- Include type hints where appropriate
- Test your changes thoroughly
- Update documentation as needed

## License

This project is open-source. Please add an appropriate LICENSE file (MIT, Apache 2.0, etc.) based on your requirements.

## Contact & Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Provide details about your environment and data
- Include minimal reproducible examples when reporting bugs

---

**Note**: This is Milestone-1 of a multi-phase project exploring NLP capabilities. Future milestones will incorporate transformer-based models and Agentic AI workflows.
