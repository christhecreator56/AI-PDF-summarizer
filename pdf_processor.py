import PyPDF2
import pdfplumber
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from typing import List, Dict, Tuple, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    Handles PDF text extraction and preprocessing for fast AI summarization.
    """
    
    def __init__(self):
        """Initialize the PDF processor with necessary NLTK data."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF using multiple methods for better coverage.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as string
        """
        text = ""
        
        # Method 1: PyPDF2
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {e}")
        
        # Method 2: pdfplumber (if PyPDF2 didn't work well)
        if len(text.strip()) < 100:  # If text is too short, try pdfplumber
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except Exception as e:
                logger.warning(f"pdfplumber extraction failed: {e}")
        
        return text.strip()
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text, preserving structure for sectioning and code.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Collapse horizontal whitespace but preserve newlines
        text = re.sub(r'[ \t\r\f\v]+', ' ', text)
        text = re.sub(r' \n', '\n', text)
        
        # Collapse multiple newlines into two for paragraph breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Allow more symbols for code snippets
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\n\=\+\*\/\[\]\{\}\<\>\_\'\"]', '', text)
        
        # Fix common OCR issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'(\d+)([a-zA-Z])', r'\1 \2', text)
        
        # Remove page numbers and headers
        text = re.sub(r'Page\s*\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

        return text.strip()
    
    def split_into_chunks(self, text: str, max_chunk_size: int = 300) -> List[str]:
        """
        Split text into very small chunks for ultra-fast processing.
        
        Args:
            text: Input text
            max_chunk_size: Maximum size of each chunk (reduced for speed)
            
        Returns:
            List of text chunks
        """
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Limit the number of chunks for very large documents (ultra-fast mode)
        if len(chunks) > 20:
            logger.info(f"Large document detected ({len(chunks)} chunks). Limiting to first 20 chunks for ultra-fast processing.")
            chunks = chunks[:20]
        
        return chunks
    
    def extract_code_blocks(self, text: str) -> List[Dict[str, str]]:
        """
        Extract code blocks and their preceding context from the text.
        
        Args:
            text: Input text
            
        Returns:
            List of dictionaries, each with 'context' and 'code'.
        """
        code_blocks = []
        code_pattern = re.compile(
            r'(^\s*(import|from|def|class|for|while|if|else|elif|try|except|with)\s.*)|'
            r'(^\s*[\w\.\_]+\s*=\s*.*)|'
            r'(^\s*[\w\.\_]+\(.*\))|'
            r'(.*\[.*\].*|.*\{.*\}.*)'
        )

        paragraphs = text.split('\n\n')
        
        for i, para in enumerate(paragraphs):
            lines = para.split('\n')
            if not lines:
                continue

            code_lines = [line for line in lines if code_pattern.search(line)]
            
            # If a high percentage of lines are code-like, treat the whole paragraph as code
            if len(lines) > 0 and (len(code_lines) / len(lines)) > 0.4:
                context = ""
                # Try to get the previous paragraph as context, if it's not also code
                if i > 0:
                    prev_para = paragraphs[i-1]
                    prev_lines = prev_para.split('\n')
                    if prev_lines:
                        prev_code_lines = [line for line in prev_lines if code_pattern.search(line)]
                        if (len(prev_code_lines) / len(prev_lines)) < 0.4:
                            context = prev_para

                code_blocks.append({'context': context, 'code': para})
            
        return code_blocks

    def extract_formulas(self, text: str) -> List[Dict[str, str]]:
        """
        Extracts formulas and their preceding context from the text.
        
        Args:
            text: Input text
            
        Returns:
            List of dictionaries, each with 'context' and 'formula'.
        """
        formulas = []
        # A formula typically contains an equals sign and some math operators.
        formula_pattern = re.compile(r'\w+\s*=\s*.*[+\-*/\^%_\[\]]')

        paragraphs = text.split('\n\n')

        for i, para in enumerate(paragraphs):
            if formula_pattern.search(para) and len(para) < 300 and len(para.split('\n')) < 5:
                context = ""
                if i > 0:
                    prev_para = paragraphs[i-1]
                    if not formula_pattern.search(prev_para):
                        context = prev_para
                
                formulas.append({
                    'context': context.strip(),
                    'formula': para.strip()
                })
        return formulas

    def extract_key_sections(self, text: str) -> Dict[str, str]:
        """
        Identify and extract key sections from the text, looking for headers.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of section names and their content
        """
        sections = {}
        
        # More robust section patterns
        section_patterns = [
            r'^(abstract|summary|introduction|conclusion|methodology|results|discussion)\b',
            r'^(chapter|section)\s+\d+',
            r'^(overview|background|related\s+work|future\s+work)\b',
            # Pattern for numbered headings like "1. Introduction"
            r'^\d+(\.\d+)*\s+[A-Z].*'
        ]
        
        lines = text.split('\n')
        current_section = "Introduction"  # Default section
        current_content = []
        
        for line in lines:
            line_strip = line.strip()
            if not line_strip:
                continue

            is_header = False
            # Headers are usually short and have a distinct format
            if len(line_strip.split()) < 10 and len(line_strip) < 100:
                # Check for explicit section headers
                if any(re.match(p, line_strip, re.IGNORECASE) for p in section_patterns):
                    is_header = True
                # Check for all-caps headers, which are common
                elif line_strip.isupper() and len(line_strip.split()) > 1:
                    is_header = True

            if is_header:
                if current_content:
                    # Clean up section name
                    clean_section_name = re.sub(r'^\d+(\.\d+)*\s*', '', current_section).strip()
                    sections[clean_section_name.title()] = '\n'.join(current_content)
                current_section = line_strip
                current_content = []
            else:
                current_content.append(line_strip)
        
        # Add the last section
        if current_content:
            clean_section_name = re.sub(r'^\d+(\.\d+)*\s*', '', current_section).strip()
            sections[clean_section_name.title()] = '\n'.join(current_content)

        # If no sections were found, return the whole text as one section
        if not sections:
            sections['Summary'] = text

        return sections
    
    def get_text_statistics(self, text: str) -> Dict[str, Any]:
        """
        Calculate text statistics for analysis.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of text statistics
        """
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalpha()]
        
        stats = {
            'total_sentences': len(sentences),
            'total_words': len(words),
            'unique_words': len(set(words)),
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'vocabulary_diversity': len(set(words)) / len(words) if words else 0
        }
        
        return stats
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Complete PDF processing pipeline with ultra-fast optimizations.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing processed text and metadata
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Extract text
        raw_text = self.extract_text_from_pdf(pdf_path)
        if not raw_text:
            raise ValueError("No text could be extracted from the PDF")
        
        # Clean text
        cleaned_text = self.clean_text(raw_text)
        
        # Split into very small chunks for ultra-fast processing
        chunks = self.split_into_chunks(cleaned_text, max_chunk_size=300)
        
        # Extract sections, code blocks, and formulas
        sections = self.extract_key_sections(cleaned_text)
        code_blocks = self.extract_code_blocks(cleaned_text)
        formulas = self.extract_formulas(cleaned_text)
        
        # Get statistics
        stats = self.get_text_statistics(cleaned_text)
        
        result = {
            'raw_text': raw_text,
            'cleaned_text': cleaned_text,
            'chunks': chunks,
            'sections': sections,
            'code_blocks': code_blocks,
            'formulas': formulas,
            'statistics': stats,
            'total_chunks': len(chunks)
        }
        
        logger.info(f"PDF processing completed. Extracted {stats['total_words']} words, found {len(code_blocks)} code blocks and {len(formulas)} formulas.")
        
        return result 