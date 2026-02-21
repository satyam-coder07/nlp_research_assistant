import os
import logging
import re
from typing import List


def setup_logging(log_file="nlp_system.log"):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def validate_file_path(file_path, allowed_extensions=None):
    if not os.path.exists(file_path):
        return False
    if allowed_extensions:
        _, ext = os.path.splitext(file_path)
        return ext.lower() in allowed_extensions
    return True


def ensure_directory_exists(directory_path):
    os.makedirs(directory_path, exist_ok=True)


def truncate_text(text, max_length=1000):
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def format_topic_keywords(keywords, num_words=10):
    keywords = keywords[:num_words]
    formatted = []
    for word, weight in keywords:
        formatted.append(f"{word} ({weight:.3f})")
    return ", ".join(formatted)


def calculate_percentage(part, whole):
    if whole == 0:
        return 0.0
    return (part / whole) * 100
