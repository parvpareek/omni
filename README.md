# 🕉️ Omni - Spiritual Guide

A real-time voice and text chatbot powered by spiritual wisdom from Avyakt Murli texts, built with **Pipecat**, **FastAPI**, and **Streamlit**.

## 🌟 Features

- **Real-time Voice & Text Chat**: WebSocket-based streaming communication
- **Spiritual Guidance**: Personalized advice based on Avyakt Murli wisdom
- **RAG-Powered Responses**: Retrieval-Augmented Generation with ChromaDB
- **Multilingual Support**: Hindi and English
- **Free Tier Services**: Google STT/TTS with generous limits
- **Modern Architecture**: Pipecat pipeline + FastAPI + Streamlit

## 🏗️ Architecture

```
[Streamlit Frontend] ← WebSocket → [FastAPI + Pipecat Pipeline] ←→ [STT/LLM/TTS Services]
```

### Components:
- **Frontend**: Streamlit web interface with real-time chat
- **Backend**: FastAPI server with WebSocket transport
- **Pipeline**: Pipecat framework for streaming audio/text processing
- **RAG**: Custom processor with ChromaDB + Gemini LLM
- **Services**: Google STT/TTS (free tier) or OpenAI services

## 🚀 Quick Start

### Prerequisites

- Python 3.11+ (conda environment recommended)
- Google API key for Gemini LLM and STT/TTS
- ChromaDB with ingested spiritual texts

### Installation

1. **Clone and setup environment**:
```bash
git clone <repository-url>
cd omni
conda create -n omni python=3.11
conda activate omni
pip install -r requirements.txt
```

2. **Configure environment variables**:
```bash
cp env.example .env
# Edit .env with your API keys
```

3. **Ingest spiritual texts** (if not already done):
```bash
python ingest.py
```

### Running the Application

#### Option 1: Full Stack (Recommended)

1. **Start the FastAPI backend**:
```bash
python fastapi_server.py
# Server runs on http://localhost:8000
```

2. **Start the Streamlit frontend** (in another terminal):
```bash
streamlit run streamlit_app.py
# Frontend runs on http://localhost:8501
```

3. **Connect and chat**:
   - Open Streamlit app in browser
   - Click "Connect" to establish WebSocket connection
   - Start asking spiritual questions!

#### Option 2: Test Backend Only

```bash
python fastapi_server.py
# Visit http://localhost:8000/test for WebSocket test page
```

#### Option 3: Direct Pipeline Test

```bash
python pipecat_pipeline.py
# Tests the pipeline components directly
```

## ⚙️ Configuration

### Environment Variables (.env)

```bash
# Required
GOOGLE_API_KEY=your_google_api_key_here

# Optional
USE_CLOUD_CHROMADB=false
USE_OPENAI_SERVICES=false
CHROMA_PERSIST_DIR=./chroma_db
GEMINI_MODEL=gemini-2.0-flash
GEMINI_TEMPERATURE=0.7

# Server configuration
HOST=0.0.0.0
PORT=8000
FASTAPI_URL=ws://localhost:8000/ws
```

### Service Options

#### Google Services (Free Tier - Default)
- **STT**: 60 minutes/month free
- **TTS**: 4 million characters/month free
- **LLM**: Gemini 2.0 Flash (generous free tier)

#### OpenAI Services (Paid)
- Set `USE_OPENAI_SERVICES=true`
- Requires `OPENAI_API_KEY`
- Better quality but costs money

## 📁 Project Structure

```
omni/
├── fastapi_server.py      # FastAPI backend with WebSocket
├── streamlit_app.py       # Streamlit frontend client
├── pipecat_pipeline.py    # Pipecat pipeline definition
├── app.py                 # Original Gradio app (legacy)
├── ingest.py              # Data ingestion script
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .env                  # Environment variables
├── chroma_db/            # Local ChromaDB storage
├── data/                 # Spiritual texts data
└── tests/                # Test files
```

## 🔧 Technical Details

### Pipecat Pipeline Architecture

```python
Pipeline([
    transport.input(),           # WebSocket input
    rtvi_processor,              # Real-time voice interaction
    stt_service,                 # Speech-to-text
    rag_processor,               # Custom RAG logic
    tts_service,                 # Text-to-speech
    transport.output()           # WebSocket output
])
```

### Custom RAG Processor

The `SpiritualRAGProcessor` integrates:
- **ChromaDB**: Vector search for spiritual texts
- **Gemini LLM**: Response generation
- **Custom Prompts**: Spiritual guidance templates
- **Source Citations**: Transparent references

### WebSocket Communication

- **Real-time streaming**: Audio/text frames
- **Bidirectional**: Client ↔ Server
- **Frame-based**: Pipecat's typed message system
- **Error handling**: Graceful disconnections

## 🎯 Usage Guide

### Text Chat
1. Connect to the server
2. Type your spiritual question
3. Receive personalized guidance with sources

### Voice Chat (Future)
- WebSocket supports audio streaming
- Browser microphone integration
- Real-time STT → RAG → TTS pipeline

### Sample Questions
- "What is the importance of meditation in spiritual life?"
- "How can I develop divine qualities?"
- "Tell me about the soul's journey"
- "What is the meaning of spiritual purity?"

## 🧪 Testing

### Run Tests
```bash
pytest tests/test_pipeline.py -v
```

### Test Components
```bash
# Test pipeline directly
python pipecat_pipeline.py

# Test FastAPI server
python fastapi_server.py
# Visit http://localhost:8000/test

# Test Streamlit client
streamlit run streamlit_app.py
```

## 🐛 Troubleshooting

### Common Issues

**Connection Failed**:
- Ensure FastAPI server is running on port 8000
- Check WebSocket URL in Streamlit config
- Verify network connectivity

**No Responses**:
- Check if ChromaDB is populated (`python ingest.py`)
- Verify Google API key is valid
- Check server logs for errors

**Audio Issues**:
- Ensure proper audio format (PCM 16-bit)
- Check microphone permissions
- Verify STT/TTS service credentials

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python fastapi_server.py
```

## 🔮 Future Enhancements

- **Voice Chat**: Browser microphone integration
- **Video Support**: Multimodal conversations
- **User Sessions**: Conversation history
- **Advanced RAG**: Multi-turn conversations
- **Mobile App**: React Native client
- **Cloud Deployment**: Docker + Kubernetes

## 📚 API Reference

### FastAPI Endpoints

- `GET /`: API information
- `GET /health`: Health check
- `GET /test`: WebSocket test page
- `POST /chat/text`: Text-only chat
- `WS /ws`: WebSocket for real-time chat

### WebSocket Messages

```json
// Send text message
{
  "type": "text",
  "text": "Your question here"
}

// Receive response
{
  "type": "text", 
  "text": "Spiritual guidance response"
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Avyakt Murli texts** for spiritual wisdom
- **Pipecat team** for the streaming framework
- **Google** for free tier AI services
- **ChromaDB** for vector database
- **Streamlit** for the web interface

---

**🕉️ May Omni guide you on your spiritual journey! 🕉️**
