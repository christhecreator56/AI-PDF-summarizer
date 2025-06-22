#!/usr/bin/env python3
"""
Test script for the AI PDF Summarizer for Students.
This script demonstrates the system's capabilities using sample text.
"""

import os
import tempfile
from pdf_processor import PDFProcessor
from ai_summarizer import AISummarizer
from student_formatter import StudentFormatter
from utils import create_temp_directory, cleanup_temp_directory

def create_sample_pdf_text():
    """Create sample text that simulates PDF content."""
    return """
    Artificial Intelligence and Machine Learning in Modern Education
    
    Abstract
    
    This comprehensive study examines the transformative impact of artificial intelligence (AI) and machine learning (ML) technologies on contemporary educational methodologies. The research encompasses various aspects including personalized learning systems, automated assessment tools, and intelligent tutoring platforms that are revolutionizing how students engage with educational content.
    
    Introduction
    
    The integration of artificial intelligence and machine learning algorithms into educational frameworks has emerged as a paradigm-shifting development in the field of pedagogy. These advanced computational technologies facilitate the creation of adaptive learning environments that can dynamically adjust to individual student needs, thereby optimizing the educational experience and improving learning outcomes.
    
    Methodology
    
    Our research methodology employed a mixed-methods approach combining quantitative analysis of student performance metrics with qualitative assessment of learning engagement patterns. The study involved 1,200 students across 15 educational institutions, utilizing AI-powered learning platforms over a 12-month period. Data collection encompassed academic performance indicators, engagement metrics, and qualitative feedback from both students and educators.
    
    Results and Discussion
    
    The implementation of AI-driven educational technologies demonstrated significant improvements in student learning outcomes. Personalized learning algorithms achieved a 34% increase in student engagement compared to traditional teaching methods. Automated assessment systems reduced grading time by 67% while maintaining accuracy levels above 95%. Furthermore, intelligent tutoring platforms provided real-time feedback that led to a 28% improvement in problem-solving skills among participating students.
    
    The analysis revealed that machine learning algorithms excel at identifying learning patterns and adapting content delivery accordingly. Students with different learning styles benefited from customized educational pathways that traditional one-size-fits-all approaches could not provide. The adaptive nature of these systems ensured that each student received instruction tailored to their specific needs and pace of learning.
    
    Challenges and Limitations
    
    Despite the promising results, several challenges emerged during the implementation phase. Technical infrastructure requirements posed significant barriers for underfunded educational institutions. Additionally, concerns regarding data privacy and algorithmic bias necessitated careful consideration of ethical implications. The digital divide between socioeconomic groups also highlighted the need for equitable access to AI-powered educational resources.
    
    Future Directions
    
    The future of AI in education appears promising, with emerging technologies such as natural language processing and computer vision offering new possibilities for interactive learning experiences. The development of more sophisticated algorithms that can understand emotional states and learning preferences represents the next frontier in educational technology innovation.
    
    Conclusion
    
    Artificial intelligence and machine learning technologies represent a fundamental shift in educational methodology, offering unprecedented opportunities for personalized and effective learning experiences. While challenges remain, the potential benefits of these technologies justify continued investment and research in this rapidly evolving field. The successful integration of AI into educational frameworks requires careful consideration of technical, ethical, and accessibility factors to ensure equitable benefits for all students.
    
    References
    
    1. Smith, J. et al. (2023). "AI in Education: A Comprehensive Review." Journal of Educational Technology, 45(2), 123-145.
    2. Johnson, A. (2023). "Machine Learning Applications in Personalized Learning." International Journal of AI in Education, 12(3), 67-89.
    3. Brown, M. (2023). "Ethical Considerations in AI-Powered Education." Educational Ethics Quarterly, 8(1), 23-45.
    """

def test_pdf_processing():
    """Test the PDF processing functionality."""
    print("🧪 Testing PDF Processing...")
    
    # Create sample text
    sample_text = create_sample_pdf_text()
    
    # Create a temporary text file to simulate PDF processing
    temp_dir = create_temp_directory()
    temp_file = os.path.join(temp_dir, "sample_content.txt")
    
    try:
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(sample_text)
        
        # Initialize PDF processor
        processor = PDFProcessor()
        
        # Test text cleaning
        cleaned_text = processor.clean_text(sample_text)
        print(f"✅ Text cleaning completed. Original: {len(sample_text)} chars, Cleaned: {len(cleaned_text)} chars")
        
        # Test chunking
        chunks = processor.split_into_chunks(cleaned_text)
        print(f"✅ Text chunking completed. Created {len(chunks)} chunks")
        
        # Test section extraction
        sections = processor.extract_key_sections(cleaned_text)
        print(f"✅ Section extraction completed. Found {len(sections)} sections")
        
        # Test statistics
        stats = processor.get_text_statistics(cleaned_text)
        print(f"✅ Statistics calculated: {stats['total_words']} words, {stats['total_sentences']} sentences")
        
        return {
            'cleaned_text': cleaned_text,
            'chunks': chunks,
            'sections': sections,
            'statistics': stats
        }
        
    finally:
        cleanup_temp_directory(temp_dir)

def test_ai_summarization(processed_data):
    """Test the AI summarization functionality."""
    print("\n🧠 Testing AI Summarization...")
    
    try:
        # Initialize AI summarizer
        summarizer = AISummarizer()
        
        # Test chunk summarization
        print("📝 Creating student-friendly summary...")
        summary_data = summarizer.create_student_summary(
            processed_data['chunks'],
            target_grade_level=8
        )
        
        print(f"✅ Summary created: {summary_data['word_count']} words")
        print(f"✅ Reading level: Grade {summary_data['readability_scores']['flesch_kincaid_grade']:.1f}")
        print(f"✅ Reading ease: {summary_data['readability_scores']['flesch_reading_ease']:.1f}/100")
        
        return summary_data
        
    except Exception as e:
        print(f"❌ AI summarization failed: {e}")
        return None

def test_student_formatting(summary_data):
    """Test the student formatting functionality."""
    print("\n📝 Testing Student Formatting...")
    
    try:
        # Initialize formatter
        formatter = StudentFormatter()
        
        # Test different formats
        formats = formatter.format_summary_for_students(summary_data)
        
        print(f"✅ Created {len(formats)} different formats:")
        for format_name, content in formats.items():
            print(f"   - {format_name}: {len(content)} characters")
        
        # Test Word document creation
        temp_dir = create_temp_directory()
        doc_path = formatter.create_word_document(summary_data, os.path.join(temp_dir, "test_summary.docx"))
        print(f"✅ Word document created: {doc_path}")
        
        cleanup_temp_directory(temp_dir)
        
        return formats
        
    except Exception as e:
        print(f"❌ Student formatting failed: {e}")
        return None

def display_sample_results(summary_data, formats):
    """Display sample results."""
    print("\n" + "="*60)
    print("📚 SAMPLE RESULTS")
    print("="*60)
    
    print(f"\n📖 SUMMARY:")
    print(summary_data['summary'])
    
    print(f"\n📊 READABILITY SCORES:")
    for metric, score in summary_data['readability_scores'].items():
        print(f"   - {metric}: {score:.2f}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    for rec in summary_data['recommendations']:
        print(f"   - {rec}")
    
    print(f"\n📄 SIMPLE TEXT FORMAT PREVIEW:")
    print(formats['simple_text'][:300] + "...")

def main():
    """Main test function."""
    print("🚀 AI PDF Summarizer for Students - System Test")
    print("="*60)
    
    # Test 1: PDF Processing
    processed_data = test_pdf_processing()
    if not processed_data:
        print("❌ PDF processing test failed")
        return
    
    # Test 2: AI Summarization
    summary_data = test_ai_summarization(processed_data)
    if not summary_data:
        print("❌ AI summarization test failed")
        return
    
    # Test 3: Student Formatting
    formats = test_student_formatting(summary_data)
    if not formats:
        print("❌ Student formatting test failed")
        return
    
    # Display results
    display_sample_results(summary_data, formats)
    
    print("\n" + "="*60)
    print("✅ All tests completed successfully!")
    print("🎉 The AI PDF Summarizer system is working correctly.")
    print("="*60)

if __name__ == "__main__":
    main() 