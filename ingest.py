#!/usr/bin/env python3
"""
Document Ingestion Script for Spiritual Texts
Processes PDF documents and stores them in ChromaDB vector database
Supports both local and cloud ChromaDB with GPU acceleration
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import re
import torch

# Core libraries
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SpiritualTextsIngestor:
    """Handles ingestion of spiritual texts into ChromaDB"""
    
    def __init__(self, use_cloud: bool = False):
        self.use_cloud = use_cloud
        self.chroma_persist_dir = os.getenv('CHROMA_PERSIST_DIR', './chroma_db')
        self.documents_dir = os.getenv('DOCUMENTS_DIR', './documents')
        
        # Detect and configure device (GPU if available)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # Initialize ChromaDB client (cloud or local)
        self._initialize_chromadb()
        
        # Initialize embedding model with GPU support
        self.embedding_model = HuggingFaceEmbeddings(
            model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
            model_kwargs={'device': self.device}
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        logger.info(f"SpiritualTextsIngestor initialized ({'cloud' if use_cloud else 'local'} mode)")
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB client (cloud or local)"""
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
    
    def extract_metadata_from_path(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from file path and name"""
        path = Path(file_path)
        metadata = {
            'file_name': path.name,
            'file_path': str(path),
            'file_size': path.stat().st_size if path.exists() else 0,
            'language': 'unknown',
            'year': 'unknown',
            'date': 'unknown',
            'source': 'avyakt_murli'
        }
        
        # Determine language from path
        if 'hindi' in file_path.lower() or 'हिंदी' in file_path.lower():
            metadata['language'] = 'hindi'
        elif 'english' in file_path.lower():
            metadata['language'] = 'english'
        
        # Extract year from path
        year_match = re.search(r'(\d{4})', str(path.parent))
        if year_match:
            metadata['year'] = year_match.group(1)
        
        # Extract date from filename (e.g., AV-E-18.01.1969.pdf)
        date_match = re.search(r'(\d{1,2}\.\d{1,2}\.\d{4})', path.name)
        if date_match:
            metadata['date'] = date_match.group(1)
        
        return metadata
    
    def load_documents_from_data_dir(self) -> List[Document]:
        """Load documents from the data directory structure"""
        documents = []
        
        # Define paths to the main directories
        hindi_dir = Path('data/All Avyakt Vani Hindi 1969 - 2020')
        english_dir = Path('data/All Avyakt English Pdf Murli - 1969-2020(1)')
        
        # Process Hindi documents
        if hindi_dir.exists():
            logger.info(f"Loading Hindi documents from: {hindi_dir}")
            hindi_docs = self._load_pdfs_from_directory(hindi_dir, language='hindi')
            documents.extend(hindi_docs)
            logger.info(f"Loaded {len(hindi_docs)} Hindi documents")
        
        # Process English documents
        if english_dir.exists():
            logger.info(f"Loading English documents from: {english_dir}")
            english_docs = self._load_pdfs_from_directory(english_dir, language='english')
            documents.extend(english_docs)
            logger.info(f"Loaded {len(english_docs)} English documents")
        
        return documents
    
    def _load_pdfs_from_directory(self, directory: Path, language: str) -> List[Document]:
        """Load PDFs from a directory and its subdirectories"""
        documents = []
        
        # Walk through all subdirectories
        for subdir in directory.iterdir():
            if subdir.is_dir():
                logger.info(f"Processing {language} subdirectory: {subdir.name}")
                
                # Load PDFs from this subdirectory
                pdf_files = list(subdir.glob('*.pdf'))
                for pdf_file in pdf_files:
                    try:
                        # Load PDF
                        loader = PyPDFLoader(str(pdf_file))
                        pdf_docs = loader.load()
                        
                        # Add metadata to each document
                        for doc in pdf_docs:
                            metadata = self.extract_metadata_from_path(str(pdf_file))
                            metadata['page_number'] = doc.metadata.get('page', 1)
                            metadata['year_dir'] = subdir.name
                            doc.metadata.update(metadata)
                            documents.append(doc)
                        
                        logger.debug(f"Loaded {len(pdf_docs)} pages from {pdf_file.name}")
                    
                    except Exception as e:
                        logger.error(f"Error loading {pdf_file}: {str(e)}")
                        continue
        
        return documents
    
    def load_documents_from_documents_dir(self) -> List[Document]:
        """Load documents from the documents directory (for user-provided texts)"""
        documents = []
        documents_path = Path(self.documents_dir)
        
        if not documents_path.exists():
            logger.warning(f"Documents directory not found: {documents_path}")
            return documents
        
        # Load all PDF files from documents directory
        try:
            loader = DirectoryLoader(
                self.documents_dir,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader
            )
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} documents from {self.documents_dir}")
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
        
        return documents
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process documents by splitting them into chunks"""
        logger.info(f"Processing {len(documents)} documents...")
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['chunk_size'] = len(chunk.page_content)
            chunk.metadata['ingestion_timestamp'] = datetime.now().isoformat()
        
        logger.info(f"Created {len(chunks)} chunks from documents")
        return chunks
    
    def store_in_chromadb(self, chunks: List[Document], collection_name: str = "spiritual_texts"):
        """Store document chunks in ChromaDB"""
        logger.info(f"Storing {len(chunks)} chunks in ChromaDB collection: {collection_name}")
        
        try:
            # Create or get collection
            collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            # Prepare data for ChromaDB
            documents_text = [chunk.page_content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            ids = [f"chunk_{i}" for i in range(len(chunks))]
            
            # Generate embeddings and store in batches
            batch_size = 100
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            logger.info(f"Processing {len(chunks)} chunks in {total_batches} batches...")
            
            for i in range(0, len(chunks), batch_size):
                batch_end = min(i + batch_size, len(chunks))
                current_batch = i // batch_size + 1
                
                # Generate embeddings for this batch
                logger.info(f"Generating embeddings for batch {current_batch}/{total_batches}...")
                batch_documents = documents_text[i:batch_end]
                batch_embeddings = self.embedding_model.embed_documents(batch_documents)
                
                # Store in ChromaDB
                collection.add(
                    documents=batch_documents,
                    embeddings=batch_embeddings,
                    metadatas=metadatas[i:batch_end],
                    ids=ids[i:batch_end]
                )
                
                logger.info(f"Stored batch {current_batch}/{total_batches}")
            
            logger.info(f"Successfully stored {len(chunks)} chunks in ChromaDB")
            
            # Display collection stats
            count = collection.count()
            logger.info(f"Collection '{collection_name}' now contains {count} documents")
            
        except Exception as e:
            logger.error(f"Error storing documents in ChromaDB: {str(e)}")
            raise
    
    def ingest_all(self):
        """Main ingestion method"""
        logger.info("Starting document ingestion process...")
        
        # Load documents from both data directory and documents directory
        all_documents = []
        
        # Load from data directory (main spiritual texts)
        data_docs = self.load_documents_from_data_dir()
        all_documents.extend(data_docs)
        
        # Load from documents directory (user-provided texts)
        user_docs = self.load_documents_from_documents_dir()
        all_documents.extend(user_docs)
        
        if not all_documents:
            logger.warning("No documents found to process")
            return
        
        logger.info(f"Total documents loaded: {len(all_documents)}")
        
        # Process documents
        chunks = self.process_documents(all_documents)
        
        # Store in ChromaDB
        self.store_in_chromadb(chunks)
        
        logger.info("Document ingestion completed successfully!")
    
    def create_sample_documents(self):
        """Create sample documents for testing when no PDFs are available"""
        logger.info("Creating sample documents for testing...")
        
        sample_texts = [
            {
                'content': "Welcome to the spiritual journey. This is a sacred text about inner wisdom and divine knowledge. The path to enlightenment requires dedication and understanding.",
                'metadata': {
                    'language': 'english',
                    'year': '2024',
                    'source': 'sample_text',
                    'file_name': 'sample1.txt'
                }
            },
            {
                'content': "आध्यात्मिक ज्ञान एक अनमोल खजाना है। यह हमारे अंतर्मन की शुद्धता और दिव्य चेतना के बारे में है। सच्चा ज्ञान हमें शांति और आनंद का अनुभव कराता है।",
                'metadata': {
                    'language': 'hindi',
                    'year': '2024',
                    'source': 'sample_text',
                    'file_name': 'sample2.txt'
                }
            },
            {
                'content': "Meditation is the key to spiritual growth. Through regular practice, we can connect with our inner self and experience divine peace. This sacred knowledge transforms our understanding of life.",
                'metadata': {
                    'language': 'english',
                    'year': '2024',
                    'source': 'sample_text',
                    'file_name': 'sample3.txt'
                }
            }
        ]
        
        # Create Document objects
        documents = []
        for text_data in sample_texts:
            doc = Document(
                page_content=text_data['content'],
                metadata=text_data['metadata']
            )
            documents.append(doc)
        
        # Process and store
        chunks = self.process_documents(documents)
        self.store_in_chromadb(chunks)
        
        logger.info("Sample documents created and stored successfully!")

def main():
    """Main function with command line argument support"""
    parser = argparse.ArgumentParser(description='Ingest spiritual texts into ChromaDB')
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
    
    logger.info(f"Starting ingestion in {'cloud' if use_cloud else 'local'} mode")
    
    # Initialize ingestor
    ingestor = SpiritualTextsIngestor(use_cloud=use_cloud)
    
    # Check if data directories exist
    hindi_dir = Path('data/All Avyakt Vani Hindi 1969 - 2020')
    english_dir = Path('data/All Avyakt English Pdf Murli - 1969-2020(1)')
    documents_dir = Path(ingestor.documents_dir)
    
    if not hindi_dir.exists() and not english_dir.exists() and not documents_dir.exists():
        logger.warning("No data directories found. Creating sample documents for testing...")
        ingestor.create_sample_documents()
    else:
        # Run full ingestion
        ingestor.ingest_all()

if __name__ == "__main__":
    main() 