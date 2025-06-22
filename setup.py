#!/usr/bin/env python3
"""
Setup script for AI PDF Summarizer for Students.
This script helps install dependencies and configure the system.
"""

import subprocess
import sys
import os
import platform

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    try:
        # Install from requirements.txt
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def download_nltk_data():
    """Download required NLTK data."""
    print("📚 Downloading NLTK data...")
    
    try:
        import nltk
        
        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        
        print("✅ NLTK data downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to download NLTK data: {e}")
        return False

def test_imports():
    """Test if all modules can be imported."""
    print("🧪 Testing imports...")
    
    modules = [
        'torch',
        'transformers',
        'PyPDF2',
        'pdfplumber',
        'streamlit',
        'numpy',
        'pandas',
        'scikit-learn',
        'nltk',
        'textstat',
        'sentence_transformers',
        'python-docx',
        'markdown'
    ]
    
    failed_imports = []
    
    for module in modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    
    print("✅ All imports successful!")
    return True

def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    
    directories = ['outputs', 'temp', 'logs']
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created directory: {directory}")
        except Exception as e:
            print(f"❌ Failed to create directory {directory}: {e}")
            return False
    
    return True

def run_test():
    """Run the test script to verify installation."""
    print("🧪 Running system test...")
    
    try:
        result = subprocess.run([sys.executable, "test_system.py"], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ System test passed!")
            return True
        else:
            print(f"❌ System test failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ System test timed out")
        return False
    except Exception as e:
        print(f"❌ System test error: {e}")
        return False

def show_usage_instructions():
    """Show usage instructions."""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    print("\n📚 AI PDF Summarizer for Students is ready to use!")
    print("\n🚀 To start the web application:")
    print("   streamlit run app.py")
    print("\n🧪 To run the test script:")
    print("   python test_system.py")
    print("\n📖 For more information, see README.md")
    print("\n" + "="*60)

def main():
    """Main setup function."""
    print("🚀 AI PDF Summarizer for Students - Setup")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Download NLTK data
    if not download_nltk_data():
        return False
    
    # Test imports
    if not test_imports():
        return False
    
    # Create directories
    if not create_directories():
        return False
    
    # Run test
    if not run_test():
        print("⚠️  System test failed, but setup completed. You may need to troubleshoot.")
    
    # Show instructions
    show_usage_instructions()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 