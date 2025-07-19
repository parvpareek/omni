#!/usr/bin/env python3
"""
Voice-based Spiritual Chatbot Application
Main application file with Gradio interface for voice-driven conversational AI
Supports both local and cloud ChromaDB with GPU acceleration
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import tempfile
import traceback
import torch

# Core libraries
from dotenv import load_dotenv
import gradio as gr
import chromadb
from chromadb.config import Settings

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate

# Speech and audio processing
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
import io

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SpiritualChatbot:
    """Voice-based spiritual chatbot with RAG capabilities"""
    
    def __init__(self, use_cloud: bool = False):
        self.use_cloud = use_cloud
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        self.chroma_persist_dir = os.getenv('CHROMA_PERSIST_DIR', './chroma_db')
        self.gemini_model = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash-latest')
        self.gemini_temperature = float(os.getenv('GEMINI_TEMPERATURE', '0.7'))
        
        # Detect and configure device (GPU if available)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self._initialize_llm()
        self._initialize_embeddings()
        self._initialize_chromadb()
        self._initialize_speech_recognizer()
        self._setup_rag_chain()
        
        logger.info(f"SpiritualChatbot initialized successfully ({'cloud' if use_cloud else 'local'} mode)")
    
    def _initialize_llm(self):
        """Initialize the Google Gemini LLM"""
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=self.gemini_model,
                temperature=self.gemini_temperature,
                google_api_key=self.google_api_key
            )
            logger.info(f"LLM initialized: {self.gemini_model}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise
    
    def _initialize_embeddings(self):
        """Initialize the embedding model with GPU support"""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
                model_kwargs={'device': self.device}
            )
            logger.info(f"Embeddings model initialized on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {str(e)}")
            raise
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB client and collection (cloud or local)"""
        try:
            if self.use_cloud:
                # Cloud ChromaDB setup
                chroma_api_key = os.getenv('CHROMA_API_KEY')
                if not chroma_api_key:
                    raise ValueError("CHROMA_API_KEY not found in environment variables for cloud mode")
                
                logger.info("Connecting to ChromaDB Cloud...")
                self.chroma_client = chromadb.HttpClient(
                    host="https://api.trychroma.com",
                    headers={"Authorization": f"Bearer {chroma_api_key}"}
                )
            else:
                # Local ChromaDB setup
                logger.info("Connecting to local ChromaDB...")
                self.chroma_client = chromadb.PersistentClient(path=self.chroma_persist_dir)
            
            # Try to get the collection
            try:
                self.collection = self.chroma_client.get_collection("spiritual_texts")
                count = self.collection.count()
                logger.info(f"Connected to existing collection with {count} documents")
            except Exception:
                logger.warning("No existing collection found. You may need to run ingest.py first.")
                self.collection = None
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise
    
    def _initialize_speech_recognizer(self):
        """Initialize speech recognition"""
        try:
            self.recognizer = sr.Recognizer()
            logger.info("Speech recognizer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize speech recognizer: {str(e)}")
            raise
    
    def _setup_rag_chain(self):
        """Set up the RAG chain for question answering"""
        try:
            # Create the prompt template
            system_prompt = """
            You are a knowledgeable spiritual assistant specializing in Avyakt Murli spiritual texts. 
            Your role is to provide helpful, accurate, and compassionate answers based on the spiritual knowledge provided.
            
            Guidelines:
            1. Answer questions based ONLY on the provided context from the spiritual texts
            2. If the answer is not found in the context, politely say "I do not have information on this topic from the provided texts"
            3. Be respectful and compassionate in your responses
            4. If the user asks in Hindi, respond in Hindi. If they ask in English, respond in English
            5. Provide practical spiritual guidance when appropriate
            6. Reference the source material when possible
            
            Context from spiritual texts:
            {context}
            
            Question: {input}
            
            Answer:
            """
            
            self.prompt = ChatPromptTemplate.from_template(system_prompt)
            
            # Create the question answering chain
            self.question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt)
            
            logger.info("RAG chain setup completed")
        except Exception as e:
            logger.error(f"Failed to setup RAG chain: {str(e)}")
            raise
    
    def get_rag_response(self, query: str) -> str:
        """Get response using RAG (Retrieval-Augmented Generation)"""
        try:
            if not self.collection:
                return "Sorry, the knowledge base is not available. Please run the ingestion script first."
            
            # Retrieve relevant documents
            query_embedding = self.embeddings.embed_query(query)
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=3,
                include=['documents', 'metadatas', 'distances']
            )
            
            if not results['documents'][0]:
                return "I do not have information on this topic from the provided texts."
            
            # Create context from retrieved documents
            context_docs = []
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                context_docs.append(Document(
                    page_content=doc,
                    metadata=metadata
                ))
            
            # Generate response using the question answering chain
            response = self.question_answer_chain.invoke({
                "context": context_docs,
                "input": query
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG response: {str(e)}")
            return f"Sorry, I encountered an error while processing your question: {str(e)}"
    
    def speech_to_text(self, audio_file_path: str) -> str:
        """Convert speech to text using speech recognition"""
        try:
            if not audio_file_path:
                return "No audio file provided"
            
            # Load audio file
            with sr.AudioFile(audio_file_path) as source:
                audio = self.recognizer.record(source)
            
            # Convert to text
            text = self.recognizer.recognize_google(audio)
            logger.info(f"Speech recognized: {text}")
            return text
            
        except sr.UnknownValueError:
            logger.warning("Could not understand audio")
            return "Sorry, I could not understand the audio. Please try again."
        except sr.RequestError as e:
            logger.error(f"Speech recognition error: {str(e)}")
            return f"Sorry, there was an error with speech recognition: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in speech to text: {str(e)}")
            return f"Sorry, there was an unexpected error: {str(e)}"
    
    def text_to_speech(self, text: str, language: str = 'en') -> str:
        """Convert text to speech using gTTS"""
        try:
            if not text:
                return None
            
            # Ensure text is a string
            if not isinstance(text, str):
                text = str(text)
            
            # Detect language if not specified
            if language == 'auto':
                # Simple language detection based on script
                if any('\u0900' <= char <= '\u097F' for char in text):
                    language = 'hi'  # Hindi
                else:
                    language = 'en'  # English
            
            # Create TTS object
            tts = gTTS(text=text, lang=language, slow=False)
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            tts.save(temp_file.name)
            
            logger.info(f"Text converted to speech: {temp_file.name}")
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Error in text to speech: {str(e)}")
            return None
    
    def process_voice_query(self, audio_file) -> Tuple[str, str, str]:
        """Process voice query and return transcription, response, and audio response"""
        try:
            if audio_file is None:
                return "No audio provided", "Please provide an audio input", None
            
            # Step 1: Speech to Text
            transcribed_text = self.speech_to_text(audio_file)
            
            if transcribed_text.startswith("Sorry"):
                return transcribed_text, "Could not process your query", None
            
            # Step 2: Get RAG response
            response_text = self.get_rag_response(transcribed_text)
            
            # Step 3: Text to Speech
            audio_response = self.text_to_speech(response_text, language='auto')
            
            return transcribed_text, response_text, audio_response
            
        except Exception as e:
            logger.error(f"Error processing voice query: {str(e)}")
            error_msg = f"An error occurred: {str(e)}"
            return error_msg, error_msg, None
    
    def process_text_query(self, text_input: str) -> Tuple[str, str]:
        """Process text query and return response and audio response"""
        try:
            if not text_input.strip():
                return "Please enter a question", None
            
            # Get RAG response
            response_text = self.get_rag_response(text_input)
            
            # Text to Speech
            audio_response = self.text_to_speech(response_text, language='auto')
            
            return response_text, audio_response
            
        except Exception as e:
            logger.error(f"Error processing text query: {str(e)}")
            error_msg = f"An error occurred: {str(e)}"
            return error_msg, None

def create_gradio_interface(use_cloud: bool = False):
    """Create and configure the Gradio interface"""
    
    # Initialize the chatbot
    try:
        chatbot = SpiritualChatbot(use_cloud=use_cloud)
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {str(e)}")
        raise
    
    # Custom CSS for better appearance
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 2rem;
    }
    .section-header {
        color: #34495e;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin: 1rem 0;
    }
    """
    
    with gr.Blocks(css=custom_css, title="Spiritual Voice Assistant") as interface:
        gr.Markdown(
            """
            # 🕉️ Spiritual Voice Assistant
            
            Welcome to your personal spiritual guide! This AI assistant is trained on Avyakt Murli spiritual texts 
            and can answer your questions in both Hindi and English through voice or text.
            
            ## How to use:
            1. **Voice Input**: Click the microphone button and speak your question
            2. **Text Input**: Type your question in the text box below
            3. The assistant will provide both text and audio responses
            """,
            elem_classes=["main-header"]
        )
        
        with gr.Tab("🎤 Voice Chat"):
            gr.Markdown("### Speak your spiritual question", elem_classes=["section-header"])
            
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label="🎙️ Record your question",
                        elem_id="audio_input"
                    )
                    
                    process_voice_btn = gr.Button(
                        "Process Voice Query",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=2):
                    transcription_output = gr.Textbox(
                        label="📝 Your Question (Transcribed)",
                        lines=2,
                        interactive=False
                    )
                    
                    response_output = gr.Textbox(
                        label="🕉️ Spiritual Assistant's Answer",
                        lines=5,
                        interactive=False
                    )
                    
                    audio_output = gr.Audio(
                        label="🔊 Listen to Response",
                        elem_id="audio_output"
                    )
            
            process_voice_btn.click(
                chatbot.process_voice_query,
                inputs=[audio_input],
                outputs=[transcription_output, response_output, audio_output]
            )
        
        with gr.Tab("💬 Text Chat"):
            gr.Markdown("### Type your spiritual question", elem_classes=["section-header"])
            
            with gr.Row():
                with gr.Column(scale=1):
                    text_input = gr.Textbox(
                        label="✍️ Your Question",
                        lines=3,
                        placeholder="Ask about spiritual wisdom, meditation, or divine knowledge..."
                    )
                    
                    process_text_btn = gr.Button(
                        "Get Answer",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=2):
                    text_response_output = gr.Textbox(
                        label="🕉️ Spiritual Assistant's Answer",
                        lines=6,
                        interactive=False
                    )
                    
                    text_audio_output = gr.Audio(
                        label="🔊 Listen to Response",
                        elem_id="text_audio_output"
                    )
            
            process_text_btn.click(
                chatbot.process_text_query,
                inputs=[text_input],
                outputs=[text_response_output, text_audio_output]
            )
            
            # Allow Enter key to submit
            text_input.submit(
                chatbot.process_text_query,
                inputs=[text_input],
                outputs=[text_response_output, text_audio_output]
            )
        
        with gr.Tab("ℹ️ Information"):
            gr.Markdown(
                """
                ## About This Assistant
                
                This spiritual voice assistant is powered by:
                - **Knowledge Base**: Avyakt Murli spiritual texts (1969-2020+)
                - **Languages**: Hindi and English
                - **AI Model**: Google Gemini 1.5 Flash
                - **Voice Recognition**: Google Speech Recognition
                - **Text-to-Speech**: Google Text-to-Speech
                - **Vector Database**: ChromaDB for semantic search
                
                ## Features
                - 🎙️ Voice input and output
                - 🌐 Multilingual support (Hindi/English)
                - 📚 Comprehensive spiritual knowledge base
                - 🔍 Semantic search for relevant answers
                - 💬 Natural conversation flow
                
                ## Usage Tips
                - Speak clearly for better voice recognition
                - You can ask questions in Hindi or English
                - The assistant will respond in the same language you use
                - For better results, be specific in your questions
                
                ## Sample Questions
                - "What is the importance of meditation in spiritual life?"
                - "How can I develop divine qualities?"
                - "आध्यात्मिक जीवन में मन की शुद्धता क्यों आवश्यक है?"
                - "Tell me about the soul's journey"
                """
            )
    
    return interface

def main():
    """Main function to run the application with command line argument support"""
    parser = argparse.ArgumentParser(description='Voice-based Spiritual Chatbot Application')
    parser.add_argument('--cloud', action='store_true',
                       help='Use ChromaDB Cloud instead of local database')
    parser.add_argument('--local', action='store_true',
                       help='Use local ChromaDB (default)')
    
    args = parser.parse_args()
    
    # Determine which database to use
    use_cloud = args.cloud
    if args.local and args.cloud:
        logger.error("Cannot specify both --cloud and --local. Choose one.")
        sys.exit(1)
    
    logger.info(f"Starting application in {'cloud' if use_cloud else 'local'} mode")
    
    try:
        # Create the Gradio interface
        interface = create_gradio_interface(use_cloud=use_cloud)
        
        # Launch the application
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,
            debug=True
        )
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 