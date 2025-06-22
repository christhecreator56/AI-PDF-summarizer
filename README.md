# AI PDF Summarizer for Students

An intelligent AI-powered system that converts complex PDF documents into simple, student-friendly summaries.

## Features

- **Smart PDF Processing**: Extracts text from PDFs while preserving structure
- **AI-Powered Summarization**: Uses advanced transformer models for intelligent summarization
- **Student-Friendly Output**: Simplifies complex language and concepts
- **Interactive Web Interface**: Easy-to-use Streamlit application
- **Multiple Output Formats**: Export summaries as text, Word documents, or markdown
- **Readability Analysis**: Ensures appropriate reading level for students

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download required NLTK data:
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

## Usage

Run the web application:
```bash
streamlit run app.py
```

Then upload your PDF and get an instant student-friendly summary!

## How It Works

1. **PDF Extraction**: Uses PyPDF2 and pdfplumber to extract text
2. **Text Preprocessing**: Cleans and structures the content
3. **AI Summarization**: Applies transformer models for intelligent summarization
4. **Simplification**: Uses readability metrics and language simplification techniques
5. **Output Generation**: Creates student-friendly summaries with explanations

## Project Structure

- `pdf_processor.py`: PDF text extraction and preprocessing
- `ai_summarizer.py`: AI model for summarization and simplification
- `student_formatter.py`: Formats output for student comprehension
- `app.py`: Streamlit web interface
- `utils.py`: Utility functions and helpers 