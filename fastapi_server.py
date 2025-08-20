import logging
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from config import settings
from pipecat_pipeline import run_webrtc_bot, RAGService

from langchain_core.documents import Document

from pipecat.transports.network.small_webrtc import SmallWebRTCConnection

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Omni Spiritual Guide",
    version="1.0.0",
    description="A real-time voice and text chat application powered by Pipecat and FastAPI."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Omni Spiritual Guide API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established.")
    # Simple text-only RAG streaming loop (bypasses Pipecat transport for reliability)
    rag = RAGService()
    try:
        while True:
            try:
                user_text = await websocket.receive_text()
            except WebSocketDisconnect:
                logger.info("WebSocket connection closed by client.")
                break
            user_text = (user_text or "").strip()
            if not user_text:
                continue

            # Retrieve context
            collection = rag._db.get_collection("spiritual_texts")
            query_embeddings = rag._embeddings.embed_query(user_text)
            vector_results = collection.query(
                query_embeddings=[query_embeddings],
                n_results=4,
                include=['documents', 'metadatas', 'distances']
            )
            documents = vector_results.get("documents", [[]])[0]
            metadatas = vector_results.get("metadatas", [[]])[0]
            docs = [
                Document(page_content=doc, metadata=meta or {})
                for doc, meta in zip(documents, metadatas)
                if doc is not None
            ]
            if not docs:
                await websocket.send_text("I couldn't find anything relevant in the knowledge base.")
                continue

            # Stream RAG response
            async for chunk in rag._chain.astream({"context": docs, "input": user_text}):
                chunk_text = getattr(chunk, 'content', None)
                if not chunk_text:
                    chunk_text = str(chunk)
                await websocket.send_text(chunk_text)

    except Exception as e:
        logger.error(f"WS/RAG error: {e}")
        try:
            await websocket.send_text("Sorry, an error occurred while processing your request.")
        except Exception:
            pass
    finally:
        logger.info("WebSocket connection and RAG loop terminated.")

@app.post("/api/offer")
async def webrtc_offer(offer: dict, background_tasks: BackgroundTasks):
    """Handle WebRTC offer from frontend and return SDP answer."""
    pc_id = offer.get("pc_id")
    sdp = offer.get("sdp")
    type_ = offer.get("type")

    # Create WebRTC connection with ICE servers for better connectivity
    ice_servers = [
        "stun:stun.l.google.com:19302",
        "stun:stun1.l.google.com:19302",
    ]
    
    webrtc_connection = SmallWebRTCConnection(ice_servers=ice_servers)
    await webrtc_connection.initialize(sdp=sdp, type=type_)

    # Start the bot in the background
    background_tasks.add_task(run_webrtc_bot, webrtc_connection)

    answer = webrtc_connection.get_answer()
    return answer

if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port)