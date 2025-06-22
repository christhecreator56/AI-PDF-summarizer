import os
import tempfile
import shutil
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import mimetypes

logger = logging.getLogger(__name__)

def validate_pdf_file(file_path: str) -> bool:
    """
    Validate if the file is a valid PDF.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if valid PDF, False otherwise
    """
    if not os.path.exists(file_path):
        logger.error(f"File does not exist: {file_path}")
        return False
    
    # Check file extension
    if not file_path.lower().endswith('.pdf'):
        logger.error(f"File is not a PDF: {file_path}")
        return False
    
    # Check file size (max 50MB)
    file_size = os.path.getsize(file_path)
    if file_size > 50 * 1024 * 1024:  # 50MB
        logger.error(f"File too large: {file_size / (1024*1024):.1f}MB")
        return False
    
    # Check MIME type
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type != 'application/pdf':
        logger.warning(f"File MIME type is not PDF: {mime_type}")
    
    return True

def create_temp_directory() -> str:
    """
    Create a temporary directory for processing files.
    
    Returns:
        Path to the temporary directory
    """
    temp_dir = tempfile.mkdtemp(prefix="pdf_summarizer_")
    logger.info(f"Created temporary directory: {temp_dir}")
    return temp_dir

def cleanup_temp_directory(temp_dir: str):
    """
    Clean up temporary directory and its contents.
    
    Args:
        temp_dir: Path to the temporary directory
    """
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temp directory {temp_dir}: {e}")

def save_uploaded_file(uploaded_file, temp_dir: str) -> str:
    """
    Save an uploaded file to a temporary directory.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        temp_dir: Temporary directory path
        
    Returns:
        Path to the saved file
    """
    file_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    logger.info(f"Saved uploaded file: {file_path}")
    return file_path

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"

def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get information about a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with file information
    """
    if not os.path.exists(file_path):
        return {}
    
    stat = os.stat(file_path)
    
    return {
        'name': os.path.basename(file_path),
        'size': stat.st_size,
        'size_formatted': format_file_size(stat.st_size),
        'created': stat.st_ctime,
        'modified': stat.st_mtime,
        'path': file_path
    }

def create_output_filename(original_name: str, suffix: str = "", extension: str = "txt") -> str:
    """
    Create an output filename based on the original file name.
    
    Args:
        original_name: Original file name
        suffix: Suffix to add to the name
        extension: File extension
        
    Returns:
        New filename
    """
    base_name = os.path.splitext(original_name)[0]
    if suffix:
        return f"{base_name}_{suffix}.{extension}"
    else:
        return f"{base_name}.{extension}"

def ensure_directory_exists(directory_path: str):
    """
    Ensure a directory exists, create it if it doesn't.
    
    Args:
        directory_path: Path to the directory
    """
    Path(directory_path).mkdir(parents=True, exist_ok=True)

def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip('. ')
    
    # Limit length
    if len(filename) > 200:
        name, ext = os.path.splitext(filename)
        filename = name[:200-len(ext)] + ext
    
    return filename

def get_supported_formats() -> List[str]:
    """
    Get list of supported output formats.
    
    Returns:
        List of supported format names
    """
    return ['txt', 'md', 'html', 'json', 'docx']

def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def validate_grade_level(grade_level: int) -> int:
    """
    Validate and normalize grade level.
    
    Args:
        grade_level: Input grade level
        
    Returns:
        Validated grade level (1-12)
    """
    if grade_level < 1:
        return 1
    elif grade_level > 12:
        return 12
    else:
        return grade_level

def create_progress_callback():
    """
    Create a progress callback function for tracking processing steps.
    
    Returns:
        Progress callback function
    """
    def progress_callback(step: str, progress: float = None):
        if progress is not None:
            logger.info(f"Progress - {step}: {progress:.1%}")
        else:
            logger.info(f"Step: {step}")
    
    return progress_callback

def extract_text_preview(text: str, max_length: int = 200) -> str:
    """
    Extract a preview of text for display.
    
    Args:
        text: Full text
        max_length: Maximum length of preview
        
    Returns:
        Text preview
    """
    if len(text) <= max_length:
        return text
    
    # Try to break at sentence boundary
    preview = text[:max_length]
    last_period = preview.rfind('.')
    
    if last_period > max_length * 0.7:  # If period is in last 30%
        return preview[:last_period + 1] + "..."
    else:
        return preview + "..."

def calculate_processing_time(start_time: float, end_time: float) -> str:
    """
    Calculate and format processing time.
    
    Args:
        start_time: Start timestamp
        end_time: End timestamp
        
    Returns:
        Formatted processing time
    """
    duration = end_time - start_time
    return format_duration(duration)

def create_error_message(error: Exception, context: str = "") -> str:
    """
    Create a user-friendly error message.
    
    Args:
        error: The exception that occurred
        context: Additional context
        
    Returns:
        User-friendly error message
    """
    error_type = type(error).__name__
    
    if "PDF" in error_type or "pdf" in str(error).lower():
        return f"PDF processing error: {str(error)}"
    elif "memory" in str(error).lower():
        return "The PDF is too large to process. Please try a smaller file."
    elif "permission" in str(error).lower():
        return "Permission denied. Please check file access."
    elif "network" in str(error).lower() or "connection" in str(error).lower():
        return "Network error. Please check your internet connection."
    else:
        return f"An error occurred: {str(error)}"

def validate_summary_data(summary_data: Dict[str, Any]) -> bool:
    """
    Validate summary data structure.
    
    Args:
        summary_data: Summary data dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_keys = ['summary', 'readability_scores', 'word_count', 'sentence_count']
    
    for key in required_keys:
        if key not in summary_data:
            logger.error(f"Missing required key in summary data: {key}")
            return False
    
    if not summary_data.get('summary'):
        logger.error("Summary text is empty")
        return False
    
    return True 