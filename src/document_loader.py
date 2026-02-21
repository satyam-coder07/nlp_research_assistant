import os
import logging
from typing import List, Dict, Optional
from pathlib import Path

try:
    import PyPDF2
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

from utils import clean_text


class DocumentLoader:

    SUPPORTED_TEXT_EXTENSIONS = ['.txt', '.md', '.text']
    SUPPORTED_PDF_EXTENSIONS = ['.pdf']

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def load_single_document(self, file_path):
        if not os.path.exists(file_path):
            return None
        file_ext = Path(file_path).suffix.lower()
        try:
            if file_ext in self.SUPPORTED_TEXT_EXTENSIONS:
                return self._load_text_file(file_path)
            elif file_ext in self.SUPPORTED_PDF_EXTENSIONS:
                return self._load_pdf_file(file_path)
            else:
                return None
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {str(e)}")
            return None

    def load_multiple_documents(self, file_paths):
        documents = {}
        for file_path in file_paths:
            text = self.load_single_document(file_path)
            if text:
                documents[file_path] = text
        return documents

    def _load_text_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        return clean_text(text)

    def _load_pdf_file(self, file_path):
        if not PDF_SUPPORT:
            raise ImportError("PDF support not available. Install PyPDF2 and pdfplumber.")
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            if text.strip():
                return clean_text(text)
        except Exception:
            pass
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return clean_text(text)
        except Exception as e:
            self.logger.error(f"PDF extraction failed: {str(e)}")
            raise

    def load_from_uploaded_files(self, uploaded_files):
        documents = {}
        for uploaded_file in uploaded_files:
            try:
                file_ext = Path(uploaded_file.name).suffix.lower()
                if file_ext in self.SUPPORTED_TEXT_EXTENSIONS:
                    text = uploaded_file.read().decode('utf-8', errors='ignore')
                    documents[uploaded_file.name] = clean_text(text)
                elif file_ext in self.SUPPORTED_PDF_EXTENSIONS and PDF_SUPPORT:
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name
                    try:
                        text = self._load_pdf_file(tmp_path)
                        documents[uploaded_file.name] = text
                    finally:
                        os.unlink(tmp_path)
            except Exception as e:
                self.logger.error(f"Error processing {uploaded_file.name}: {str(e)}")
        return documents

    @staticmethod
    def get_supported_extensions():
        extensions = DocumentLoader.SUPPORTED_TEXT_EXTENSIONS.copy()
        if PDF_SUPPORT:
            extensions.extend(DocumentLoader.SUPPORTED_PDF_EXTENSIONS)
        return extensions
