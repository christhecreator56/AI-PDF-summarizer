import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import textstat
import re
from typing import List, Dict, Tuple, Any
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import heapq

logger = logging.getLogger(__name__)

class AISummarizer:
    """
    Fast and accurate AI-powered summarization system with student-friendly output.
    """
    
    def __init__(self, model_name: str = "facebook/bart-base"):
        """
        Initialize the AI summarizer with optimized settings.
        
        Args:
            model_name: Name of the transformer model to use
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize summarization pipeline with smaller, faster model
        try:
            self.summarizer = pipeline(
                "summarization",
                model=model_name,
                device=0 if self.device == "cuda" else -1
            )
            logger.info(f"Loaded summarization model: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load {model_name}, using default: {e}")
            self.summarizer = pipeline("summarization", device=0 if self.device == "cuda" else -1)
        
        # Initialize sentence embeddings for similarity
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Language simplification rules
        self.complex_words = {
            'utilize': 'use',
            'implement': 'put into action',
            'facilitate': 'help',
            'methodology': 'method',
            'methodologies': 'methods',
            'subsequently': 'after that',
            'consequently': 'as a result',
            'furthermore': 'also',
            'moreover': 'also',
            'nevertheless': 'however',
            'notwithstanding': 'despite',
            'aforementioned': 'mentioned before',
            'aforementioned': 'mentioned above',
            'subsequently': 'later',
            'preliminary': 'first',
            'subsequent': 'following',
            'preceding': 'before',
            'subsequent': 'next',
            'preliminary': 'early',
            'subsequently': 'then'
        }
    
    def extractive_summarize(self, text: str, num_sentences: int = 5) -> str:
        """
        Fast extractive summarization using TF-IDF and sentence importance.
        
        Args:
            text: Input text
            num_sentences: Number of sentences to extract
            
        Returns:
            Extracted summary
        """
        # Split into sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if len(sentences) <= num_sentences:
            return text
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
        except ValueError:
            return text # Not enough vocab

        # Calculate sentence scores based on TF-IDF
        sentence_scores = []
        for i in range(len(sentences)):
            score = np.sum(tfidf_matrix[i].toarray())
            sentence_scores.append((score, i))
        
        # Get top sentences
        top_sentences = heapq.nlargest(num_sentences, sentence_scores)
        top_sentences.sort(key=lambda x: x[1])  # Sort by original position
        
        # Combine sentences
        summary = '. '.join([sentences[i] for _, i in top_sentences]) + '.'
        
        return summary
    
    def smart_chunk_summarization(self, chunks: List[str], target_length: int = 200) -> str:
        """
        Smart chunk summarization that processes only the most important chunks.
        
        Args:
            chunks: List of text chunks
            target_length: Target summary length
            
        Returns:
            Summarized text
        """
        if len(chunks) <= 3:
            # For small documents, summarize all chunks
            combined_text = ' '.join(chunks)
            return self.extractive_summarize(combined_text, num_sentences=8)
        
        # For large documents, select most important chunks
        chunk_scores = []
        for i, chunk in enumerate(chunks):
            # Score based on length and content
            words = chunk.split()
            score = len(words) * 0.3  # Length factor
            
            # Bonus for chunks with key terms
            key_terms = ['introduction', 'conclusion', 'summary', 'important', 'key', 'main', 'primary']
            for term in key_terms:
                if term.lower() in chunk.lower():
                    score += 10
            
            chunk_scores.append((score, i))
        
        # Select top chunks (limit to 10 for speed)
        top_chunks = heapq.nlargest(min(10, len(chunks)), chunk_scores)
        selected_chunks = [chunks[i] for _, i in sorted(top_chunks, key=lambda x: x[1])]
        
        # Combine and summarize
        combined_text = ' '.join(selected_chunks)
        return self.extractive_summarize(combined_text, num_sentences=10)
    
    def simplify_language(self, text: str) -> str:
        """
        Simplify complex language for student comprehension.
        
        Args:
            text: Input text
            
        Returns:
            Simplified text
        """
        # Replace complex words
        for complex_word, simple_word in self.complex_words.items():
            text = re.sub(r'\b' + complex_word + r'\b', simple_word, text, flags=re.IGNORECASE)
        
        # Break long sentences
        sentences = text.split('. ')
        simplified_sentences = []
        
        for sentence in sentences:
            if len(sentence.split()) > 25:  # Long sentence
                # Try to break at conjunctions
                conjunctions = [' and ', ' but ', ' or ', ' because ', ' however ', ' therefore ']
                for conj in conjunctions:
                    if conj in sentence:
                        parts = sentence.split(conj)
                        if len(parts) > 1:
                            sentence = conj.join(parts[:2]) + '.' + conj.join(parts[2:])
                            break
            
            simplified_sentences.append(sentence)
        
        return '. '.join(simplified_sentences)

    def highlight_key_phrases(self, text: str) -> str:
        """
        Highlight key phrases in the text by making them bold.

        Args:
            text: Input text.

        Returns:
            Text with key phrases in bold markdown.
        """
        if not text.strip():
            return text

        # Use TF-IDF to find important words
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_features=20)
        try:
            vectorizer.fit([text])
            feature_names = vectorizer.get_feature_names_out()
            
            # Sort features by length to bold longer phrases first
            feature_names = sorted(feature_names, key=len, reverse=True)

            for phrase in feature_names:
                # Use word boundaries to avoid matching parts of words
                text = re.sub(r'\b(' + re.escape(phrase) + r')\b', r'**\1**', text, flags=re.IGNORECASE)
        except ValueError:
            # Happens if the text is too short or has no vocabulary
            return text

        return text

    def calculate_readability(self, text: str) -> Dict[str, float]:
        """
        Calculate various readability metrics.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of readability scores
        """
        scores = {
            'flesch_reading_ease': textstat.flesch_reading_ease(text),
            'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
            'gunning_fog': textstat.gunning_fog(text),
            'smog_index': textstat.smog_index(text),
            'automated_readability_index': textstat.automated_readability_index(text),
            'coleman_liau_index': textstat.coleman_liau_index(text),
            'linsear_write_formula': textstat.linsear_write_formula(text),
            'dale_chall_readability_score': textstat.dale_chall_readability_score(text)
        }
        
        return scores
    
    def add_explanations(self, text: str) -> str:
        """
        Add explanations for complex concepts.
        
        Args:
            text: Input text
            
        Returns:
            Text with added explanations
        """
        # Define concepts that might need explanation
        concept_explanations = {
            r'\bAI\b': 'AI (Artificial Intelligence)',
            r'\bML\b': 'ML (Machine Learning)',
            r'\bAPI\b': 'API (Application Programming Interface)',
            r'\bGUI\b': 'GUI (Graphical User Interface)',
            r'\bCPU\b': 'CPU (Central Processing Unit)',
            r'\bRAM\b': 'RAM (Random Access Memory)',
            r'\bHTTP\b': 'HTTP (Hypertext Transfer Protocol)',
            r'\bURL\b': 'URL (Uniform Resource Locator)',
            r'\bHTML\b': 'HTML (Hypertext Markup Language)',
            r'\bCSS\b': 'CSS (Cascading Style Sheets)',
            r'\bSQL\b': 'SQL (Structured Query Language)',
            r'\bJSON\b': 'JSON (JavaScript Object Notation)',
            r'\bXML\b': 'XML (Extensible Markup Language)',
            r'\bREST\b': 'REST (Representational State Transfer)',
            r'\bSOAP\b': 'SOAP (Simple Object Access Protocol)'
        }
        
        for pattern, explanation in concept_explanations.items():
            text = re.sub(pattern, explanation, text)
        
        return text

    def create_student_summary(self, pdf_data: Dict[str, Any], target_grade_level: int = 8) -> Dict[str, Any]:
        """
        Create a fast and accurate student-friendly summary, structured with headings.
        
        Args:
            pdf_data: Dictionary containing chunks and sections from PDFProcessor
            target_grade_level: Target reading grade level
            
        Returns:
            Dictionary containing summary and metadata
        """
        text_chunks = pdf_data.get('chunks', [])
        sections = pdf_data.get('sections', {})
        code_blocks = pdf_data.get('code_blocks', [])
        formulas = pdf_data.get('formulas', [])
        full_text = pdf_data.get('cleaned_text', '')
        
        logger.info(f"Creating structured student summary from {len(text_chunks)} chunks and {len(sections)} sections.")
        
        # Filter out small/irrelevant sections
        meaningful_sections = {k: v for k, v in sections.items() if len(v.split()) > 30}
        
        # Use section-based summary if we have good sections, otherwise fall back to chunk-based
        is_structured = len(meaningful_sections) > 1 and "Summary" not in meaningful_sections
        
        if is_structured:
            logger.info(f"Using section-based summarization with {len(meaningful_sections)} sections.")
            section_summaries = self.summarize_by_sections(meaningful_sections)
            
            structured_summary_parts = []
            for section_name, section_summary in section_summaries.items():
                if section_summary.strip():
                    # Add section heading and summary
                    highlighted_summary = self.highlight_key_phrases(section_summary)
                    structured_summary_parts.append(f"### {section_name}\n\n{highlighted_summary}\n")
                    
                    # Find and append code blocks for this section
                    original_section_content = meaningful_sections.get(section_name, "")
                    for code_block in code_blocks:
                        if code_block['code'] in original_section_content:
                            explanation = ""
                            if code_block['context']:
                                explanation = self.extractive_summarize(code_block['context'], num_sentences=1)
                            
                            if explanation:
                                structured_summary_parts.append(f"_{explanation}_\n")
                            structured_summary_parts.append(f"```python\n{code_block['code']}\n```\n")
            
            final_summary = '\n'.join(structured_summary_parts)
        else:
            logger.info("Using chunk-based summarization as fallback.")
            summary_text = self.smart_chunk_summarization(text_chunks, target_length=300)
            simplified_summary = self.simplify_language(summary_text)
            final_summary = self.highlight_key_phrases(simplified_summary)
            # Add all code blocks at the end if not structured
            if code_blocks:
                final_summary += "\n\n### Code Examples\n\n"
                for code_block in code_blocks:
                    explanation = ""
                    if code_block['context']:
                        explanation = self.extractive_summarize(code_block['context'], num_sentences=1)
                    if explanation:
                        final_summary += f"_{explanation}_\n"
                    final_summary += f"```python\n{code_block['code']}\n```\n\n"

        explained_summary = self.add_explanations(final_summary)

        # For readability calculation, remove all markdown formatting
        plain_summary_text = re.sub(r'```.*?```', '', explained_summary, flags=re.DOTALL)
        plain_summary_text = re.sub(r'(\*\*|_|###\s?.*?\n)', '', plain_summary_text)
        
        # Calculate readability on the plain text
        readability_scores = self.calculate_readability(plain_summary_text)
        
        # Create structured output
        result = {
            'summary': explained_summary,
            'readability_scores': readability_scores,
            'target_grade_level': target_grade_level,
            'word_count': len(plain_summary_text.split()),
            'sentence_count': len(plain_summary_text.split('.')),
            'compression_ratio': len(plain_summary_text) / len(' '.join(text_chunks)) if text_chunks else 0,
            'is_structured': is_structured,
            'has_code': bool(code_blocks),
            'has_formulas': bool(formulas)
        }
        
        # Add recommendations
        result['recommendations'] = self._generate_recommendations(readability_scores, target_grade_level)
        
        logger.info(f"Structured student summary created: {result['word_count']} words, structured: {result['is_structured']}")
        
        return result
    
    def _generate_recommendations(self, readability_scores: Dict[str, float], target_grade: int) -> List[str]:
        """
        Generate recommendations for improving readability.
        
        Args:
            readability_scores: Readability metrics
            target_grade: Target grade level
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        flesch_grade = readability_scores.get('flesch_kincaid_grade', 0)
        
        if flesch_grade > target_grade + 2:
            recommendations.append("Consider simplifying vocabulary for better student comprehension")
        
        if readability_scores.get('gunning_fog', 0) > 12:
            recommendations.append("Break down complex sentences into shorter ones")
        
        if readability_scores.get('smog_index', 0) > 10:
            recommendations.append("Use more common words and fewer technical terms")
        
        if not recommendations:
            recommendations.append("The text is well-suited for student reading level")
        
        return recommendations
    
    def summarize_by_sections(self, sections: Dict[str, str]) -> Dict[str, str]:
        """
        Summarize each section separately using fast extractive method.
        
        Args:
            sections: Dictionary of section names and content
            
        Returns:
            Dictionary of section names and summaries
        """
        section_summaries = {}
        
        for section_name, content in sections.items():
            if len(content.split()) > 50:  # Only summarize if content is substantial
                summary = self.extractive_summarize(content, num_sentences=5)
                simplified_summary = self.simplify_language(summary)
                section_summaries[section_name] = simplified_summary
            else:
                section_summaries[section_name] = content
        
        return section_summaries 