import React, { useState, useEffect, useRef } from 'react';
import { PipecatClient } from "@pipecat-ai/client-js";
import { SmallWebRTCTransport } from "@pipecat-ai/small-webrtc-transport";

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

function VoiceChat() {
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState('');
  const [transcription, setTranscription] = useState('');
  const [isListening, setIsListening] = useState(false);
  const pcClientRef = useRef(null);
  const audioRef = useRef(null);

  const handleBotMessage = (message) => {
    if (message.type === 'text') {
      setTranscription(prev => prev + message.text);
    }
  };

  const handleTransportStateChanged = (state) => {
    console.log("Transport state changed:", state);
    if (state === 'connected') {
      setIsConnected(true);
      setIsConnecting(false);
    } else if (state === 'disconnected') {
      setIsConnected(false);
      setIsConnecting(false);
    }
  };

  const handleBotReady = async () => {
    console.log("Bot is ready to chat!");
    setIsListening(pcClientRef.current?.isMicEnabled ?? false);
  };

  const handleError = (message) => {
    console.error("Pipecat client error:", message);
    setError(message.message || 'An error occurred');
    setIsConnecting(false);
  };

  const handleTrackStarted = (track, participant) => {
    if (participant && !participant.local && track.kind === "audio") {
      if (audioRef.current) {
        audioRef.current.srcObject = new MediaStream([track]);
        audioRef.current.play().catch(console.error);
      }
    }
  };

  const startConnection = async () => {
    try {
      setError('');
      setIsConnecting(true);
      setTranscription('');

      const pcClient = new PipecatClient({
        transport: new SmallWebRTCTransport({
          iceServers: [
            { urls: "stun:stun.l.google.com:19302" },
            { urls: "stun:stun1.l.google.com:19302" }
          ],
        }),
        enableMic: false, // we will enable on user gesture below
        enableCam: false,
        callbacks: {
          onMessage: handleBotMessage,
          onTransportStateChanged: handleTransportStateChanged,
          onBotReady: handleBotReady,
          onError: handleError,
          onTrackStarted: handleTrackStarted,
        },
      });

      pcClientRef.current = pcClient;

      // On user gesture: initialize devices then immediately enable mic to prompt permission
      await pcClient.initDevices();
      pcClient.enableMic(true);

      // Establish WebRTC connection
      await pcClient.connect({
        webrtcUrl: `${API_BASE}/api/offer`,
      });

      console.log("Pipecat client connected successfully");
      setIsListening(pcClient.isMicEnabled);

    } catch (err) {
      console.error("Connection error:", err);
      setError(err.message || 'Failed to connect');
      setIsConnecting(false);
    }
  };

  const stopConnection = async () => {
    try {
      if (pcClientRef.current) {
        await pcClientRef.current.disconnect();
        pcClientRef.current = null;
      }
      setIsConnected(false);
      setIsListening(false);
      setTranscription('');
      if (audioRef.current) {
        audioRef.current.srcObject = null;
      }
    } catch (err) {
      console.error("Disconnect error:", err);
    }
  };

  useEffect(() => {
    return () => {
      if (pcClientRef.current) {
        pcClientRef.current.disconnect();
      }
    };
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '20px' }}>
      <h2 style={{ marginBottom: '20px', color: '#333' }}>Voice Chat with Spiritual Guide</h2>
      <div style={{ marginBottom: '20px' }}>
        {!isConnected ? (
          <button onClick={startConnection} disabled={isConnecting} style={{ padding: '12px 24px', fontSize: '16px', backgroundColor: isConnecting ? '#ccc' : '#1976d2', color: 'white', border: 'none', borderRadius: '8px', cursor: isConnecting ? 'not-allowed' : 'pointer', minWidth: '120px' }}>
            {isConnecting ? 'Connecting...' : 'Start Voice Chat'}
          </button>
        ) : (
          <button onClick={stopConnection} style={{ padding: '12px 24px', fontSize: '16px', backgroundColor: '#d32f2f', color: 'white', border: 'none', borderRadius: '8px', cursor: 'pointer', minWidth: '120px' }}>
            Stop Voice Chat
          </button>
        )}
      </div>
      <div style={{ marginBottom: '20px', padding: '12px', borderRadius: '8px', backgroundColor: isConnected ? '#e8f5e8' : '#fff3cd', border: `1px solid ${isConnected ? '#4caf50' : '#ffc107'}`, color: isConnected ? '#2e7d32' : '#856404', textAlign: 'center', minWidth: '200px' }}>
        {isConnecting && '🔄 Connecting...'}
        {isConnected && !isListening && '✅ Connected - Enable mic if prompted'}
        {isConnected && isListening && '🎤 Connected - Speak now!'}
        {!isConnected && !isConnecting && '⏸️ Not connected'}
      </div>
      <audio ref={audioRef} autoPlay playsInline style={{ display: 'none' }} />
      {error && (
        <div style={{ marginBottom: '20px', padding: '12px', borderRadius: '8px', backgroundColor: '#ffebee', border: '1px solid #f44336', color: '#c62828', textAlign: 'center', maxWidth: '400px' }}>
          <strong>Error:</strong> {error}
        </div>
      )}
      {transcription && (
        <div style={{ marginTop: '20px', padding: '16px', backgroundColor: '#f5f5f5', borderRadius: '8px', maxWidth: '500px', width: '100%', minHeight: '100px', border: '1px solid #ddd' }}>
          <h4 style={{ margin: '0 0 12px 0', color: '#333' }}>Bot Response:</h4>
          <div style={{ lineHeight: '1.6', color: '#555', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{transcription}</div>
        </div>
      )}
    </div>
  );
}

export default VoiceChat;