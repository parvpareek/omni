import logging
import asyncio
from fastapi import WebSocket
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.pipeline.task import PipelineParams
from pipecat.processors.frame_processor import FrameDirection
from pipecat.frames.frames import (
    AudioRawFrame,
    TextFrame,
    StartFrame,
    StopFrame,
    Frame,
    TranscriptionFrame,
    TTSAudioRawFrame,
)
from pipecat.services.ai_service import AIService
from pipecat.services.google.stt import GoogleSTTService
from pipecat.services.google.tts import GoogleTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketTransport, FastAPIWebsocketParams
from config import settings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import chromadb
import os
from langchain_core.documents import Document

# New imports for WebRTC transport
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport, SmallWebRTCConnection
from pipecat.audio.vad.silero import SileroVADAnalyzer, VADParams
import torch

logger = logging.getLogger(__name__)

class RAGService(AIService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info("Initializing RAGService...")
        self._llm = self._create_llm()
        logger.info("LLM created")
        # Use the same embedding model that was used to create the collection
        self._embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        logger.info("Embeddings created")
        self._db = self._create_db_client()
        logger.info("DB client created")
        self._chain = self._create_rag_chain()
        logger.info("RAG chain created")

        # Phase 0: Improved utterance-level gating
        self._utterance_buffer: str = ""
        self._last_processed_text: str = ""
        self._debounce_task: asyncio.Task | None = None
        self._debounce_secs: float = 0.4  # Reduced from 0.8s for better responsiveness
        self._rag_running: bool = False
        self._utterance_start_time: float = 0.0
        self._min_utterance_length: int = 3  # Minimum words to consider as complete utterance

    def _create_llm(self):
        logger.info(f"Creating LLM service: {settings.llm_service}")
        try:
            if settings.llm_service == "gemini":
                logger.info(f"Creating Gemini LLM with API key: {settings.google_api_key[:10]}..." if settings.google_api_key else "API key is None")
                llm = ChatGoogleGenerativeAI(model=settings.gemini_model, temperature=settings.gemini_temperature, google_api_key=settings.google_api_key)
                logger.info("Gemini LLM created successfully")
                return llm
            elif settings.llm_service == "openai":
                return ChatOpenAI(model_name=settings.openai_model, api_key=settings.openai_api_key)
            elif settings.llm_service == "local":
                return Ollama(base_url=settings.local_llm_url, model="llama2")
            else:
                raise ValueError(f"Unsupported LLM service: {settings.llm_service}")
        except Exception as e:
            logger.error(f"Error creating LLM: {e}")
            raise

    def _create_db_client(self):
        if settings.use_cloud_chromadb:
            return chromadb.CloudClient(api_key=settings.chroma_api_key, tenant=settings.chroma_tenant, database=settings.chroma_database)
        else:
            return chromadb.PersistentClient(path=settings.chroma_persist_dir)

    def _create_rag_chain(self):
        prompt = ChatPromptTemplate.from_template(
            "You are a helpful assistant. Answer the user's question based on the following context:\n\n{context}\n\nQuestion: {input}"
        )
        return create_stuff_documents_chain(self._llm, prompt)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        logger.debug(f"RAGService received frame: {type(frame).__name__} - {frame}")

        # IMPORTANT: Let the base class handle Start/Stop and any control frames
        if isinstance(frame, (StartFrame, StopFrame)):
            # Phase 0: Reset utterance state on conversation start/stop
            if isinstance(frame, StartFrame):
                self._reset_utterance_state()
            return await super().process_frame(frame, direction)

        # Handle TextFrame from WebRTC pipeline
        if isinstance(frame, TextFrame):
            text = (frame.text or "").strip()
            if not text:
                return
            await self._handle_stt_text(text)
            return

        # Handle raw text data from WebSocket transport (frames with .text attr)
        if getattr(frame, 'text', None):
            logger.info(f"Processing text data: {frame.text}")
            await self._process_text_query(frame.text)
            return

        # For all other frames, delegate to base class
        return await super().process_frame(frame, direction)

    def _reset_utterance_state(self):
        """Phase 0: Reset utterance state for new conversation."""
        logger.info("Resetting utterance state for new conversation")
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        self._utterance_buffer = ""
        self._last_processed_text = ""
        self._rag_running = False
        self._utterance_start_time = 0.0

    def _handle_interruption(self):
        """Phase 0: Handle conversation interruption by clearing current utterance."""
        logger.info("Handling conversation interruption")
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        self._utterance_buffer = ""
        # Don't reset _last_processed_text to maintain conversation context

    async def _handle_stt_text(self, text: str):
        """Phase 0: Improved utterance boundary detection with transcript change tracking."""
        import time
        
        current_time = time.time()
        text = text.strip()
        
        if not text:
            return
            
        # Safety check: prevent processing if RAG is already running
        if self._rag_running:
            logger.debug("RAG already running, buffering text: '{text}'")
            # Still buffer the text but don't start new processing
            if self._utterance_buffer and not self._utterance_buffer.endswith(" "):
                self._utterance_buffer += " "
            self._utterance_buffer += text
            return
            
        # Check if this is a completely new utterance (not just a partial update)
        if self._is_new_utterance(text):
            logger.info(f"New utterance detected: '{text}' (previous: '{self._last_processed_text}')")
            # Cancel any pending debounce task
            if self._debounce_task and not self._debounce_task.done():
                self._debounce_task.cancel()
            
            # Reset buffer and start new utterance
            self._utterance_buffer = text
            self._utterance_start_time = current_time
        else:
            # This is a partial update to the current utterance
            logger.debug(f"Partial update to utterance: '{text}'")
            # Append with space if needed
            if self._utterance_buffer and not self._utterance_buffer.endswith(" "):
                self._utterance_buffer += " "
            self._utterance_buffer += text

        # Reset debounce timer for current utterance
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        loop = asyncio.get_running_loop()
        self._debounce_task = loop.create_task(self._debounce_flush())

    def _is_new_utterance(self, new_text: str) -> bool:
        """Determine if this is a new utterance vs. partial update."""
        if not self._last_processed_text:
            return True
            
        # Phase 0: Check for exact duplicates first, but allow explicit repeat requests
        if new_text.strip().lower() == self._last_processed_text.strip().lower():
            if self.allow_repeat_question(new_text):
                logger.info(f"Explicit repeat request detected: '{new_text}' - allowing")
                return True
            else:
                logger.debug(f"Exact duplicate detected: '{new_text}' - skipping")
                return False
            
        # Check if new text is completely different (new question/topic)
        if new_text != self._last_processed_text:
            # Check if it's a prefix/suffix or completely different
            if (new_text.startswith(self._last_processed_text) or 
                self._last_processed_text.startswith(new_text)):
                return False
            else:
                return True
                
        return False

    def allow_repeat_question(self, text: str) -> bool:
        """Phase 0: Allow user to explicitly repeat a question (e.g., 'Can you repeat that?')"""
        text_lower = text.strip().lower()
        repeat_phrases = [
            'can you repeat that',
            'can you say that again',
            'repeat that',
            'say that again',
            'what was that',
            'i didn\'t hear that',
            'can you explain that again'
        ]
        return any(phrase in text_lower for phrase in repeat_phrases)

    async def _debounce_flush(self):
        """Phase 0: Improved debounce with utterance validation."""
        try:
            await asyncio.sleep(self._debounce_secs)
        except asyncio.CancelledError:
            return
            
        if not self._utterance_buffer:
            return
            
        # Phase 0: Validate utterance before processing
        if not self._should_process_utterance():
            logger.debug(f"Skipping utterance '{self._utterance_buffer}' - too short or incomplete")
            self._utterance_buffer = ""
            return
            
        if self._rag_running:
            # If a query is in-flight, reschedule a short delay
            logger.debug("RAG already running, rescheduling flush")
            await asyncio.sleep(0.2)  # Reduced from 0.3s
            return await self._debounce_flush()
            
        # Phase 0: Mark this utterance as being processed
        self._rag_running = True
        query = self._utterance_buffer.strip()
        self._utterance_buffer = ""
        
        logger.info(f"Processing utterance: '{query}' (length: {len(query)} chars, {len(query.split())} words)")
        
        try:
            await self._process_text_query(query)
            # Phase 0: Update last processed text to prevent duplicates
            self._last_processed_text = query
            logger.info(f"Successfully processed utterance: '{query}'")
        except Exception as e:
            logger.error(f"Error processing utterance '{query}': {e}")
            # Phase 0: Provide user-friendly error message
            error_msg = "I'm having trouble processing your request right now. Please try again in a moment."
            await self.push_frame(TextFrame(error_msg))
        finally:
            self._rag_running = False
            logger.debug("RAG processing completed, service ready for next utterance")

    def _should_process_utterance(self) -> bool:
        """Phase 0: Validate if utterance should be processed."""
        if not self._utterance_buffer:
            return False
            
        # Check minimum length (words)
        word_count = len(self._utterance_buffer.split())
        if word_count < self._min_utterance_length:
            logger.debug(f"Utterance too short: {word_count} words < {self._min_utterance_length}")
            return False
            
        # Check if utterance seems complete (ends with punctuation or question words)
        text = self._utterance_buffer.strip().lower()
        if text.endswith(('.', '?', '!', '...')):
            logger.debug(f"Utterance ends with punctuation: '{text}'")
            return True
            
        # Check for common question patterns
        question_starters = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'can', 'could', 'would', 'will', 'do', 'does', 'is', 'are', 'was', 'were']
        if any(text.startswith(starter) for starter in question_starters):
            # For questions, check if they seem complete
            if word_count >= 4:  # Questions should have some content
                logger.debug(f"Question utterance detected: '{text}'")
                return True
            else:
                logger.debug(f"Question too short: '{text}'")
                return False
            
        # Check utterance duration (if it's been building for too long, process it)
        import time
        utterance_duration = time.time() - self._utterance_start_time
        if utterance_duration > 3.0:  # Max 3 seconds to build utterance
            logger.debug(f"Utterance timeout reached: {utterance_duration:.1f}s")
            return True
            
        # Check if utterance seems like a complete thought (not just filler words)
        filler_words = {'um', 'uh', 'ah', 'er', 'hmm', 'well', 'so', 'like', 'you know'}
        meaningful_words = [w for w in text.split() if w.lower() not in filler_words]
        if len(meaningful_words) >= 2:  # At least 2 meaningful words
            logger.debug(f"Utterance has meaningful content: '{text}'")
            return True
            
        logger.debug(f"Utterance validation failed: '{text}' (duration: {utterance_duration:.1f}s, words: {word_count})")
        return False

    async def force_process_utterance(self):
        """Phase 0: Force process current utterance if it has content."""
        if self._utterance_buffer and not self._rag_running:
            logger.info(f"Force processing utterance: '{self._utterance_buffer}'")
            query = self._utterance_buffer.strip()
            self._utterance_buffer = ""
            self._last_processed_text = query
            await self._process_text_query(query)
        elif self._rag_running:
            logger.debug("RAG already running, cannot force process")
        else:
            logger.debug("No utterance to force process")

    def get_status(self) -> dict:
        """Phase 0: Get current status of the RAG service for debugging."""
        import time
        return {
            "utterance_buffer": self._utterance_buffer,
            "last_processed_text": self._last_processed_text,
            "rag_running": self._rag_running,
            "utterance_start_time": self._utterance_start_time,
            "debounce_task_active": self._debounce_task is not None and not self._debounce_task.done(),
            "current_time": time.time(),
            "utterance_duration": time.time() - self._utterance_start_time if self._utterance_start_time > 0 else 0
        }

    async def _process_text_query(self, query_text: str):
        """Process a text query through RAG"""
        try:
            logger.info(f"Starting RAG processing for query: '{query_text}'")
            
            collection = self._db.get_collection("spiritual_texts")

            # Generate embeddings for the query text
            logger.debug("Generating query embeddings...")
            query_embeddings = self._embeddings.embed_query(query_text)

            # Vector-only retrieval using embeddings directly
            logger.debug("Performing vector search...")
            vector_results = collection.query(
                query_embeddings=[query_embeddings],
                n_results=4,
                include=['documents', 'metadatas', 'distances']
            )

            # Create proper Document objects
            documents = vector_results["documents"][0]
            metadatas = vector_results["metadatas"][0]

            # Filter out None values and create Document objects
            docs = []
            for doc, meta in zip(documents, metadatas):
                if doc is not None:
                    document = Document(
                        page_content=doc,
                        metadata=meta if meta else {}
                    )
                    docs.append(document)

            if not docs:
                logger.warning("No relevant documents found for query")
                await self.push_frame(TextFrame("I don't have enough information to answer that question. Could you please rephrase or ask something else?"))
                return

            logger.info(f"Found {len(docs)} relevant documents, generating response...")

            # Streaming Response using the RAG chain
            async for chunk in self._chain.astream({"context": docs, "input": query_text}):
                # Extract the text content from the chunk
                if hasattr(chunk, 'content'):
                    chunk_text = chunk.content
                else:
                    chunk_text = str(chunk)

                # Push each chunk as a TextFrame for streaming
                await self.push_frame(TextFrame(chunk_text))

            logger.info("RAG processing completed successfully")

        except Exception as e:
            logger.error(f"Error processing RAG query: {e}")
            # Phase 0: Provide user-friendly error message
            error_msg = "I'm having trouble processing your request right now. Please try again in a moment."
            await self.push_frame(TextFrame(error_msg))

def create_stt_service():
    logger.info(f"Creating STT service: {settings.stt_service}")
    try:
        if settings.stt_service == "google":
            logger.info(f"Using Google credentials path: {settings.google_credentials_path}")
            logger.info("Creating GoogleSTTService...")
            service = GoogleSTTService(credentials_path=settings.google_credentials_path)
            logger.info("GoogleSTTService created successfully")
            return service
        elif settings.stt_service == "openai":
            return OpenAISTTService(api_key=settings.openai_api_key)
        elif settings.stt_service == "local":
            return OpenAISTTService(api_base=settings.local_stt_url)
        else:
            raise ValueError(f"Unsupported STT service: {settings.stt_service}")
    except Exception as e:
        logger.error(f"Error creating STT service: {e}")
        raise

def create_tts_service():
    logger.info(f"Creating TTS service: {settings.tts_service}")
    try:
        if settings.tts_service == "google":
            logger.info(f"Using Google credentials path for TTS: {settings.google_credentials_path}")
            logger.info("Creating GoogleTTSService...")
            service = GoogleTTSService(credentials_path=settings.google_credentials_path)
            logger.info("GoogleTTSService created successfully")
            return service
        elif settings.tts_service == "openai":
            return OpenAITTSService(api_key=settings.openai_api_key)
        elif settings.tts_service == "local":
            return OpenAITTSService(api_base=settings.local_tts_url)
        else:
            raise ValueError(f"Unsupported TTS service: {settings.tts_service}")
    except Exception as e:
        logger.error(f"Error creating TTS service: {e}")
        raise

async def create_pipeline(websocket: WebSocket):
    logger.info("Creating pipeline...")
    try:
        transport = FastAPIWebsocketTransport(
            websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=False,
                audio_out_enabled=False,
            ),
        )
        logger.info("Transport created")

        # Text-only path for /ws to validate RAG+LLM
        rag = RAGService()
        logger.info("RAG service created")
    except Exception as e:
        logger.error(f"Error in pipeline creation: {e}")
        raise

    # Simple text pipeline without RTVI complexity
    pipeline = Pipeline([
        transport.input(),
        rag,
        transport.output(),
    ])
    logger.info("Pipeline created")

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=False,
            enable_metrics=False,
        ),
    )
    logger.info("Pipeline task created")

    @transport.event_handler("on_client_connected")
    async def on_client_connected(*_args, **_kwargs):
        logger.info("WS client connected -> sending StartFrame")
        await task.queue_frame(StartFrame())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, websocket, reason=None):
        logger.info(f"Client disconnected: {reason}")
        await task.cancel()

    logger.info("Creating pipeline runner...")
    runner = PipelineRunner()
    logger.info("Starting pipeline runner...")
    await runner.run(task)

# New: WebRTC-run bot for continuous demo with interruption support
async def run_webrtc_bot(connection: SmallWebRTCConnection):
    """Run the spiritual bot with WebRTC transport and interruption support."""
    
    # Create transport params for audio input/output with optimized VAD
    transport_params = TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_sample_rate=16000,
        audio_out_sample_rate=24000,
        vad_enabled=True,
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(
                stop_secs=0.5,  # Phase 0: Reduced from 0.8s for better responsiveness
                start_secs=0.2,  # Phase 0: Added start_secs for better speech detection
                threshold=0.5,
                min_volume=0.3   # Phase 0: Added min_volume for better noise handling
            )
        )
    )
    
    # Create the transport
    transport = SmallWebRTCTransport(
        webrtc_connection=connection,
        params=transport_params
    )

    # RTVI for readiness/metrics
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    # Services
    stt = create_stt_service()
    rag = RAGService()
    tts = create_tts_service()
    
    pipeline = Pipeline([
        transport.input(),
        rtvi,
        stt,
        rag,
        tts,
        transport.output()
    ])
    
    runner = PipelineRunner()
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=False
        ),
        observers=[RTVIObserver(rtvi)]
    )

    @transport.event_handler("on_connected")
    async def on_transport_connected(*_args, **_kwargs):
        logger.info("WebRTC transport connected -> sending StartFrame")
        await task.queue_frame(StartFrame())

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.info("Client ready event received")
        await rtvi.set_bot_ready()
        logger.info("Bot ready sent to client")

    await runner.run(task)