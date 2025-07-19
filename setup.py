#!/usr/bin/env python3
"""
Setup script for the Voice-based Spiritual Chatbot
Creates necessary directories and helps with initial configuration
"""

import os
import sys
from pathlib import Path
import shutil
import subprocess

def create_directories():
    """Create necessary directories for the project"""
    directories = [
        'documents',
        'chroma_db',
        'analysis_results',
        'temp_audio'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def setup_environment():
    """Set up environment file if it doesn't exist"""
    env_file = Path('.env')
    env_example = Path('env.example')
    
    if not env_file.exists() and env_example.exists():
        shutil.copy(env_example, env_file)
        print("✅ Created .env file from template")
        print("⚠️  Please edit .env file and add your Google API key")
    elif env_file.exists():
        print("✅ .env file already exists")
    else:
        print("❌ env.example file not found")

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected")
        print("⚠️  Python 3.8 or higher is required")
        return False
    else:
        print(f"✅ Python {version.major}.{version.minor} detected")
        return True

def install_dependencies():
    """Install required Python packages"""
    print("📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True, capture_output=True, text=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("Try running: pip install -r requirements.txt")
        return False

def check_api_key():
    """Check if Google API key is configured"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv('GOOGLE_API_KEY')
        if api_key and api_key != 'your_google_api_key_here':
            print("✅ Google API key configured")
            return True
        else:
            print("⚠️  Google API key not configured")
            print("Please edit .env file and add your Google API key")
            return False
    except ImportError:
        print("⚠️  Cannot check API key - dotenv not installed")
        return False

def display_next_steps():
    """Display next steps for the user"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Configure your Google API key in the .env file")
    print("2. Run data analysis: python data_analysis.py")
    print("3. Ingest your documents: python ingest.py")
    print("4. Launch the application: python app.py")
    print("\nFor detailed instructions, see README.md")
    print("="*60)

def main():
    """Main setup function"""
    print("🕉️  Voice-based Spiritual Chatbot Setup")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Setup environment file
    setup_environment()
    
    # Install dependencies
    if not install_dependencies():
        print("⚠️  Please install dependencies manually and run setup again")
        sys.exit(1)
    
    # Check API key
    check_api_key()
    
    # Display next steps
    display_next_steps()

if __name__ == "__main__":
    main() 