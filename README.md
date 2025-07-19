# 🕉️ Voice-based Spiritual Chatbot

A sophisticated voice-driven conversational AI assistant that provides spiritual guidance based on Avyakt Murli texts. The chatbot supports both Hindi and English and features a modern web interface with voice input/output capabilities.

## ✨ Features

- **🎙️ Voice Input/Output**: Natural speech interaction with the assistant
- **🌐 Multilingual Support**: Seamless Hindi and English conversation
- **📚 Rich Knowledge Base**: Trained on 2000+ Avyakt Murli spiritual texts (1969-2020+)
- **🔍 Semantic Search**: Advanced RAG (Retrieval-Augmented Generation) system
- **💬 Dual Interface**: Both voice and text chat options
- **🎨 Modern UI**: Clean, responsive web interface built with Gradio
- **📱 Cross-platform**: Works on desktop and mobile browsers

## 🛠️ Technology Stack

- **Language Model**: Google Gemini 1.5 Flash
- **Vector Database**: ChromaDB for semantic search
- **Embeddings**: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
- **Speech-to-Text**: Google Speech Recognition API
- **Text-to-Speech**: Google Text-to-Speech (gTTS)
- **Web Framework**: Gradio for interactive UI
- **Backend**: Python with LangChain framework

## 📊 Dataset Information

The knowledge base contains:
- **Hindi PDFs**: 1,108 documents
- **English PDFs**: 1,075 documents
- **Total**: 2,183+ spiritual texts
- **Time Period**: 1969-2020+
- **Storage Requirements**: ~4-6 GB for full dataset

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Google API Key for Gemini
- 8GB+ RAM recommended for full dataset
- Microphone access for voice input

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd omni

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp env.example .env

# Edit .env file and add your Google API key
nano .env
```

Add your Google API key to the `.env` file:
```
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. Data Analysis (Optional)

Run the data analysis script to understand your dataset:

```bash
python data_analysis.py
```

This will generate insights about your PDF collection and create analysis reports.

### 4. Data Ingestion

Process and index your spiritual texts:

```bash
python ingest.py
```

This will:
- Load PDF documents from the data directory
- Extract text and create embeddings
- Store everything in ChromaDB
- Create a searchable knowledge base

### 5. Launch the Application

```bash
python app.py
```

The application will start on `http://localhost:7860`

## 📁 Project Structure

```
omni/
├── app.py                 # Main application with Gradio interface
├── ingest.py             # Document processing and indexing
├── data_analysis.py      # Dataset analysis and insights
├── requirements.txt      # Python dependencies
├── env.example          # Environment variables template
├── README.md            # This file
├── data/                # Your PDF documents
│   ├── Hindi.csv        # Hindi document metadata
│   ├── English.csv      # English document metadata
│   ├── All Avyakt Vani Hindi 1969 - 2020/
│   └── All Avyakt English Pdf Murli - 1969-2020(1)/
├── documents/           # Additional user documents (optional)
├── chroma_db/          # ChromaDB persistent storage
└── analysis_results/   # Generated analysis reports
```

## 🎯 Usage Guide

### Voice Chat
1. Open the "🎤 Voice Chat" tab
2. Click the microphone button
3. Speak your question clearly
4. Get both text and audio responses

### Text Chat
1. Open the "💬 Text Chat" tab
2. Type your question
3. Click "Get Answer" or press Enter
4. Receive text and audio responses

### Sample Questions

**English:**
- "What is the importance of meditation in spiritual life?"
- "How can I develop divine qualities?"
- "Tell me about the soul's journey"
- "What is the meaning of spiritual transformation?"

**Hindi:**
- "आध्यात्मिक जीवन में मन की शुद्धता क्यों आवश्यक है?"
- "ध्यान कैसे करें?"
- "आत्मा की यात्रा क्या है?"
- "दिव्य गुण कैसे विकसित करें?"

## 🔧 Configuration Options

### Environment Variables

```bash
# Required
GOOGLE_API_KEY=your_google_api_key_here

# Optional
GEMINI_MODEL=gemini-1.5-flash-latest
GEMINI_TEMPERATURE=0.7
CHROMA_PERSIST_DIR=./chroma_db
DOCUMENTS_DIR=./documents
```

### Customization

- **Model Selection**: Change `GEMINI_MODEL` to use different Gemini variants
- **Temperature**: Adjust `GEMINI_TEMPERATURE` for response creativity (0.0-1.0)
- **Chunk Size**: Modify in `ingest.py` for different text processing
- **Retrieval Count**: Adjust in `app.py` for more/fewer context documents

## 🎭 Advanced Features

### Hybrid Retrieval (Future Enhancement)
The system is designed to support hybrid retrieval combining:
- Semantic search (current)
- Keyword search (planned)
- Metadata filtering (planned)

### Multi-tenancy Support
The architecture supports multiple users and applications through:
- Separate collections per user
- Metadata-based filtering
- Access control integration

### Performance Optimization
- Batch processing for large datasets
- Persistent vector storage
- Memory-efficient chunking
- Caching for frequent queries

## 🐛 Troubleshooting

### Common Issues

**"No module named 'pyaudio'"**
```bash
# On Ubuntu/Debian
sudo apt-get install portaudio19-dev
pip install pyaudio

# On macOS
brew install portaudio
pip install pyaudio

# On Windows
pip install pipwin
pipwin install pyaudio
```

**"GOOGLE_API_KEY not found"**
- Ensure your `.env` file exists and contains the API key
- Check that the key is valid and has appropriate permissions

**"No existing collection found"**
- Run `python ingest.py` first to create the knowledge base
- Ensure your data directories contain PDF files

**Memory Issues**
- Reduce batch size in `ingest.py`
- Process documents in smaller chunks
- Consider using a machine with more RAM

### Performance Tips

1. **For Large Datasets**: Process documents in batches
2. **Memory Optimization**: Close other applications during ingestion
3. **Voice Recognition**: Ensure good microphone quality
4. **Network**: Stable internet connection for API calls

## 🔄 Future Enhancements

### Phase 1 (Current)
- ✅ Voice input/output
- ✅ Multilingual support
- ✅ RAG implementation
- ✅ Web interface

### Phase 2 (Planned)
- 🔄 Hybrid retrieval system
- 🔄 Advanced TTS models
- 🔄 Conversation history
- 🔄 User authentication

### Phase 3 (Future)
- 🔄 Mobile app
- 🔄 Offline mode
- 🔄 Custom voice training
- 🔄 API endpoints

## 📜 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details.

## 📧 Support

For issues, questions, or suggestions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the documentation

## 🙏 Acknowledgments

- Avyakt Murli spiritual texts for the knowledge base
- Google for Gemini API and speech services
- ChromaDB team for the vector database
- LangChain community for the RAG framework
- Gradio team for the web interface

---

*May this spiritual assistant guide you on your journey of inner wisdom and divine understanding.* 🕉️
