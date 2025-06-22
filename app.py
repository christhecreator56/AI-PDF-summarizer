import streamlit as st
import os
import tempfile
from pdf_processor import PDFProcessor
from ai_summarizer import AISummarizer
from student_formatter import StudentFormatter
import time
import json
import re
from io import BytesIO

# Configure page
st.set_page_config(
    page_title="AI PDF Summarizer - Student Edition",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">📚 AI PDF Summarizer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Transform complex PDFs into student-friendly summaries in seconds!</p>', unsafe_allow_html=True)
    
    # Initialize components
    if 'pdf_processor' not in st.session_state:
        st.session_state.pdf_processor = PDFProcessor()
    
    if 'ai_summarizer' not in st.session_state:
        st.session_state.ai_summarizer = AISummarizer()
    
    if 'formatter' not in st.session_state:
        st.session_state.formatter = StudentFormatter()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Processing mode
        processing_mode = st.selectbox(
            "Processing Mode",
            ["Ultra-Fast (Recommended)", "Fast", "Standard"],
            help="Ultra-Fast: Uses extractive summarization for instant results"
        )
        
        # Target grade level
        target_grade = st.slider(
            "Target Grade Level",
            min_value=1,
            max_value=12,
            value=8,
            help="Choose the reading level for the summary"
        )
        
        # Output format
        output_format = st.selectbox(
            "Output Format",
            ["Text", "Markdown", "HTML", "JSON", "Word Document"],
            help="Choose how you want to download the summary"
        )
        
        # Advanced options
        with st.expander("Advanced Options"):
            show_readability = st.checkbox("Show Readability Metrics", value=True)
            show_recommendations = st.checkbox("Show Recommendations", value=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📄 Upload PDF")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload any PDF document to get a student-friendly summary"
        )
        
        if uploaded_file is not None:
            # Display file info
            file_size = len(uploaded_file.getvalue()) / 1024  # KB
            st.info(f"📁 File: {uploaded_file.name} ({file_size:.1f} KB)")
            
            # Process button
            if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
                process_pdf(uploaded_file, processing_mode, target_grade, output_format, 
                          show_readability, show_recommendations)
    
    with col2:
        st.header("📊 Quick Stats")
        
        if 'last_summary' in st.session_state:
            summary_data = st.session_state.last_summary
            
            st.metric("Words", summary_data.get('word_count', 0))
            st.metric("Sentences", summary_data.get('sentence_count', 0))
            st.metric("Compression", f"{summary_data.get('compression_ratio', 0)*100:.1f}%")
            
            if show_readability and 'readability_scores' in summary_data:
                st.subheader("📖 Readability")
                scores = summary_data['readability_scores']
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Flesch Grade", f"{scores.get('flesch_kincaid_grade', 0):.1f}")
                    st.metric("Gunning Fog", f"{scores.get('gunning_fog', 0):.1f}")
                
                with col_b:
                    st.metric("SMOG Index", f"{scores.get('smog_index', 0):.1f}")
                    st.metric("ARI", f"{scores.get('automated_readability_index', 0):.1f}")

def process_pdf(uploaded_file, processing_mode, target_grade, output_format, 
               show_readability, show_recommendations):
    """Process the uploaded PDF and generate summary."""
    
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Save uploaded file temporarily
        status_text.text("📁 Saving PDF file...")
        progress_bar.progress(10)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name
        
        # Step 2: Extract text from PDF
        status_text.text("📖 Extracting and structuring text from PDF...")
        progress_bar.progress(30)
        
        pdf_data = st.session_state.pdf_processor.process_pdf(pdf_path)
        
        # Step 3: Generate summary
        status_text.text("🤖 Generating structured student-friendly summary...")
        progress_bar.progress(60)
        
        # Pass the entire pdf_data object to the summarizer
        summary_data = st.session_state.ai_summarizer.create_student_summary(
            pdf_data, 
            target_grade_level=target_grade
        )

        # Step 4: Format output
        status_text.text("📝 Formatting output...")
        progress_bar.progress(80)
        
        # Add metadata
        summary_data['original_filename'] = uploaded_file.name
        summary_data['processing_mode'] = processing_mode
        summary_data['target_grade'] = target_grade
        
        # Store in session state
        st.session_state.last_summary = summary_data
        
        # Step 5: Display results
        status_text.text("✅ Summary complete!")
        progress_bar.progress(100)
        
        display_results(summary_data, output_format, show_readability, 
                       show_recommendations, uploaded_file)
        
        # Clean up
        os.unlink(pdf_path)
        
    except Exception as e:
        st.error(f"❌ Error processing PDF: {str(e)}")
        progress_bar.progress(0)
        status_text.text("")

def display_results(summary_data, output_format, show_readability, 
                   show_recommendations, uploaded_file):
    """Display the summary results."""
    
    st.markdown("---")
    st.header("📋 Summary Results")
    
    # Main summary
    st.subheader("📝 Student-Friendly Summary")
    # Use st.markdown to render headings properly from the summary
    st.markdown(summary_data['summary'], unsafe_allow_html=True)

    # Readability metrics
    if show_readability and 'readability_scores' in summary_data:
        st.subheader("📊 Readability Analysis")
        
        scores = summary_data['readability_scores']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Flesch Reading Ease", f"{scores.get('flesch_reading_ease', 0):.1f}")
        with col2:
            st.metric("Flesch-Kincaid Grade", f"{scores.get('flesch_kincaid_grade', 0):.1f}")
        with col3:
            st.metric("Gunning Fog Index", f"{scores.get('gunning_fog', 0):.1f}")
        with col4:
            st.metric("SMOG Index", f"{scores.get('smog_index', 0):.1f}")
        
        # Grade level interpretation
        flesch_grade = scores.get('flesch_kincaid_grade', 0)
        if flesch_grade <= 6:
            st.success("✅ This summary is suitable for elementary school students")
        elif flesch_grade <= 8:
            st.info("📚 This summary is suitable for middle school students")
        elif flesch_grade <= 10:
            st.warning("⚠️ This summary may be challenging for younger students")
        else:
            st.error("❌ This summary is too complex for most students")
    
    # Recommendations
    if show_recommendations and 'recommendations' in summary_data:
        st.subheader("💡 Recommendations")
        for rec in summary_data['recommendations']:
            st.info(f"• {rec}")
    
    # Download options
    st.subheader("💾 Download Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if output_format == "Text":
            # For text download, strip markdown
            plain_text_summary = re.sub(r'(\*\*|###\s?.*?\n|```.*?```)', '', summary_data['summary'], flags=re.DOTALL)
            st.download_button(
                label="📄 Download as Text",
                data=plain_text_summary,
                file_name=f"summary.txt",
                mime="text/plain"
            )
        elif output_format == "Markdown":
            markdown_content = f"# Summary of {summary_data['original_filename']}\n\n{summary_data['summary']}"
            st.download_button(
                label="📝 Download as Markdown",
                data=markdown_content,
                file_name=f"summary.md",
                mime="text/markdown"
            )
        elif output_format == "HTML":
            # Convert markdown to HTML for download
            import markdown
            html_body = markdown.markdown(summary_data['summary'], extensions=['fenced_code', 'tables'])
            html_content = f"""
            <html>
            <head><title>Summary</title></head>
            <body>
                <h1>Summary of {summary_data['original_filename']}</h1>
                {html_body}
            </body>
            </html>
            """
            st.download_button(
                label="🌐 Download as HTML",
                data=html_content,
                file_name=f"summary.html",
                mime="text/html"
            )
    
    with col2:
        if output_format == "JSON":
            st.download_button(
                label="📊 Download as JSON",
                data=json.dumps(summary_data, indent=2),
                file_name=f"summary.json",
                mime="application/json"
            )
        elif output_format == "Word Document":
            # Create Word document in memory
            doc = st.session_state.formatter.create_word_document(summary_data)
            bio = BytesIO()
            doc.save(bio)
            bio.seek(0)
            
            st.download_button(
                label="📄 Download as Word",
                data=bio,
                file_name=f"summary.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
    
    # Success message
    st.markdown('<div class="success-message">🎉 Summary generated successfully! The content above is optimized for student comprehension.</div>', 
                unsafe_allow_html=True)

if __name__ == "__main__":
    main() 