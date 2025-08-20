import React, { useState } from 'react';
import Chat from './components/Chat.jsx';
import VoiceChat from './components/VoiceChat.jsx';

function App() {
  const [tab, setTab] = useState('voice');

  return (
    <div style={{ maxWidth: 600, margin: '2rem auto', fontFamily: 'sans-serif' }}>
      <h1 style={{ textAlign: 'center' }}>🕉️ Omni Spiritual Guide</h1>
      <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 16 }}>
        <button
          onClick={() => setTab('text')}
          style={{
            padding: '8px 24px',
            marginRight: 8,
            background: tab === 'text' ? '#1976d2' : '#eee',
            color: tab === 'text' ? '#fff' : '#333',
            border: 'none',
            borderRadius: 4,
            cursor: 'pointer',
          }}
        >
          Text Chat
        </button>
        <button
          onClick={() => setTab('voice')}
          style={{
            padding: '8px 24px',
            background: tab === 'voice' ? '#1976d2' : '#eee',
            color: tab === 'voice' ? '#fff' : '#333',
            border: 'none',
            borderRadius: 4,
            cursor: 'pointer',
          }}
        >
          Voice Chat
        </button>
      </div>
      <div style={{ border: '1px solid #ddd', borderRadius: 8, padding: 24, minHeight: 400 }}>
        {tab === 'text' ? <Chat /> : <VoiceChat />}
      </div>
      <footer style={{ marginTop: 32, textAlign: 'center', color: '#888', fontSize: 14 }}>
        Powered by Pipecat, FastAPI, and React
      </footer>
    </div>
  );
}

export default App;
