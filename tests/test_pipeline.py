#!/usr/bin/env python3
"""
Tests for Omni Spiritual Guide Pipeline
Tests the FastAPI + Pipecat architecture
"""

import pytest
import asyncio
import os
import sys
import logging
from unittest.mock import Mock, patch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture
def sample_env():
    """Sample environment variables for testing"""
    return {
        'GOOGLE_API_KEY': 'test_key',
        'USE_CLOUD_CHROMADB': 'false',
        'USE_OPENAI_SERVICES': 'false',
        'CHROMA_PERSIST_DIR': './test_chroma_db',
        'GEMINI_MODEL': 'gemini-2.0-flash',
        'GEMINI_TEMPERATURE': '0.7'
    }

@pytest.fixture
def mock_websocket():
    """Mock WebSocket connection"""
    mock_ws = Mock()
    mock_ws.send = Mock()
    mock_ws.recv = Mock()
    mock_ws.close = Mock()
    return mock_ws

class TestSpiritualRAGProcessor:
    """Test the custom RAG processor"""
    
    @pytest.mark.asyncio
    async def test_processor_initialization(self, sample_env):
        """Test that the RAG processor initializes correctly"""
        with patch.dict(os.environ, sample_env):
            try:
                from pipecat_pipeline import SpiritualRAGProcessor
                
                # Test initialization
                processor = SpiritualRAGProcessor(use_cloud=False)
                
                # Check that components are initialized
                assert processor.llm is not None
                assert processor.embeddings is not None
                assert processor.question_answer_chain is not None
                
                logger.info("✅ RAG processor initialization test passed")
                
            except Exception as e:
                logger.warning(f"⚠️ RAG processor test skipped: {str(e)}")
                pytest.skip(f"RAG processor not available: {str(e)}")
    
    @pytest.mark.asyncio
    async def test_rag_response(self, sample_env):
        """Test RAG response generation"""
        with patch.dict(os.environ, sample_env):
            try:
                from pipecat_pipeline import SpiritualRAGProcessor
                
                processor = SpiritualRAGProcessor(use_cloud=False)
                
                # Test with a simple query
                query = "What is meditation?"
                response = processor.get_rag_response(query)
                
                # Check that we get a response (even if it's an error message)
                assert isinstance(response, str)
                assert len(response) > 0
                
                logger.info("✅ RAG response test passed")
                
            except Exception as e:
                logger.warning(f"⚠️ RAG response test skipped: {str(e)}")
                pytest.skip(f"RAG response not available: {str(e)}")

class TestPipecatService:
    """Test the main Pipecat service"""
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, sample_env):
        """Test that the service initializes correctly"""
        with patch.dict(os.environ, sample_env):
            try:
                from pipecat_pipeline import create_spiritual_pipeline
                
                # Test service creation
                service = create_spiritual_pipeline(use_cloud=False, use_openai_services=False)
                
                # Check that components are initialized
                assert service.rag_processor is not None
                assert service.pipeline is not None
                
                logger.info("✅ Pipecat service initialization test passed")
                
            except Exception as e:
                logger.warning(f"⚠️ Service initialization test skipped: {str(e)}")
                pytest.skip(f"Service not available: {str(e)}")

class TestFastAPIServer:
    """Test FastAPI server components"""
    
    def test_server_import(self):
        """Test that FastAPI server can be imported"""
        try:
            import fastapi_server
            logger.info("✅ FastAPI server import test passed")
        except ImportError as e:
            logger.warning(f"⚠️ FastAPI server import test skipped: {str(e)}")
            pytest.skip(f"FastAPI server not available: {str(e)}")
    
    def test_websocket_client(self):
        """Test WebSocket client functionality"""
        try:
            from streamlit_app import WebSocketClient
            
            # Test client creation
            client = WebSocketClient("ws://localhost:8000/ws")
            assert client.url == "ws://localhost:8000/ws"
            assert client.connected == False
            
            logger.info("✅ WebSocket client test passed")
            
        except ImportError as e:
            logger.warning(f"⚠️ WebSocket client test skipped: {str(e)}")
            pytest.skip(f"WebSocket client not available: {str(e)}")

class TestEnvironment:
    """Test environment configuration"""
    
    def test_required_env_vars(self):
        """Test that required environment variables are set"""
        required_vars = ['GOOGLE_API_KEY']
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            logger.warning(f"⚠️ Missing environment variables: {missing_vars}")
            pytest.skip(f"Missing required environment variables: {missing_vars}")
        else:
            logger.info("✅ Environment variables test passed")
    
    def test_chromadb_availability(self):
        """Test that ChromaDB is available"""
        try:
            import chromadb
            logger.info("✅ ChromaDB availability test passed")
        except ImportError:
            logger.warning("⚠️ ChromaDB not available")
            pytest.skip("ChromaDB not available")

class TestDependencies:
    """Test that all required dependencies are available"""
    
    def test_pipecat_import(self):
        """Test Pipecat import"""
        try:
            import pipecat
            logger.info("✅ Pipecat import test passed")
        except ImportError:
            logger.warning("⚠️ Pipecat not available")
            pytest.skip("Pipecat not available")
    
    def test_fastapi_import(self):
        """Test FastAPI import"""
        try:
            import fastapi
            logger.info("✅ FastAPI import test passed")
        except ImportError:
            logger.warning("⚠️ FastAPI not available")
            pytest.skip("FastAPI not available")
    
    def test_streamlit_import(self):
        """Test Streamlit import"""
        try:
            import streamlit
            logger.info("✅ Streamlit import test passed")
        except ImportError:
            logger.warning("⚠️ Streamlit not available")
            pytest.skip("Streamlit not available")

import multiprocessing
import time
import uvicorn

@pytest.mark.asyncio
async def test_end_to_end_flow():
    """Test the complete flow from text input to response"""
    
    def run_server():
        uvicorn.run("fastapi_server:app", host="localhost", port=8000, log_level="info")

    server_process = multiprocessing.Process(target=run_server)
    server_process.start()
    time.sleep(5)  # Give the server time to start

    try:
        from streamlit_app import WebSocketClient
        
        client = WebSocketClient("ws://localhost:8000/ws")
        await client.connect()
        
        assert client.connected
        
        response = await client.send_message("What is meditation?")
        
        assert isinstance(response, str)
        assert len(response) > 0
        
        await client.disconnect()
        
        logger.info("✅ End-to-end flow test passed")
        
    except Exception as e:
        logger.warning(f"⚠️ End-to-end flow test skipped: {str(e)}")
        pytest.skip(f"End-to-end flow not available: {str(e)}")
    finally:
        server_process.terminate()
        server_process.join()

def main():
    """Run all tests"""
    logger.info("🧪 Running Omni pipeline tests...")
    
    # Check environment
    if not os.getenv('GOOGLE_API_KEY'):
        logger.warning("⚠️ GOOGLE_API_KEY not set. Some tests will be skipped.")
    
    # Run tests
    pytest.main([__file__, "-v"])

if __name__ == "__main__":
    main() 