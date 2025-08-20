# Omni Minimalist React Frontend

This is a minimalist React frontend for the Omni Spiritual Guide, supporting both text and voice chat with a FastAPI+Pipecat backend.

## Features
- Text chat (input, chat history, send button)
- Voice chat (mic button, live transcription, streaming TTS playback)
- WebSocket connection to backend (`/ws`)

## Prerequisites
- Node.js 18+
- Backend running at `http://localhost:8000` (see project root for FastAPI+Pipecat setup)

## Setup
```bash
cd frontend
npm install
```

## Running the App
```bash
npm run dev
```
- Open [http://localhost:5173](http://localhost:5173) in your browser.
- The frontend will connect to the backend WebSocket at `ws://localhost:8000/ws` by default.

## Configuration
- To change the backend WebSocket URL, create a `.env` file in `frontend/`:
  ```env
  VITE_WS_URL=ws://localhost:8000/ws
  ```

## File Structure
```
frontend/
├── src/
│   ├── App.jsx           # Main app with tab UI
│   ├── components/
│   │   ├── Chat.js       # Text chat component
│   │   └── VoiceChat.js  # Voice chat component
│   └── index.js          # Entry point
├── public/
│   └── index.html        # HTML template
├── package.json
└── README.md
```

## Notes
- Voice chat uses the browser microphone and streams audio to the backend via WebSocket.
- The backend must support:
  - Receiving raw audio chunks (e.g., WebM/PCM)
  - Sending back partial transcriptions and TTS audio (base64-encoded WAV/PCM)
- For production, build with `npm run build` and serve the static files.

---

**Questions?** See the main project README or ask your engineering team.
